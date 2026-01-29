"""Global track manager for cross-camera person tracking.

Manages global tracks that span multiple cameras, handling:
1. Creation of global tracks from local tracks
2. Camera handover for overlapping views
3. Re-ID matching for non-overlapping cameras
4. Track lifecycle management
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from src.config import CameraTopologyConfig, ReIDConfig, ZonesConfig
from src.database.repository import Repository
from src.detection.person_detector import compute_iou
from src.recognition.identity_linker import IdentityLinker
from src.tracking.track import GlobalTrack, LocalTrack, TrackState
from src.tracking.zone_manager import ZoneManager
from src.utils.profiler import profiler

logger = logging.getLogger(__name__)


@dataclass
class CameraHandoverZone:
    """Defines an overlap zone between two cameras."""

    camera1_id: str
    camera2_id: str
    cam1_exit_zone: tuple[float, float, float, float]  # Normalized (x1, y1, x2, y2)
    cam2_entry_zone: tuple[float, float, float, float]
    max_handover_sec: float = 3.0


@dataclass
class PendingHandover:
    """Track pending handover between cameras."""

    global_track_id: str
    from_camera: str
    zone_name: str  # Zone where track was lost
    exit_time: float
    max_handover_sec: float
    reid_embedding: Optional[np.ndarray] = None
    last_bbox: Optional[tuple[float, float, float, float]] = None


@dataclass
class GlobalTrackingResult:
    """Result of global tracking update."""

    active_tracks: list[GlobalTrack]
    new_global_tracks: list[str]
    handovers_completed: list[tuple[str, str, str]]  # (track_id, from_cam, to_cam)
    tracks_lost: list[str]


class GlobalTrackManager:
    """Manages global tracks across all cameras.

    Coordinates between:
    - Local ByteTrack instances (per-camera)
    - Identity linker (face recognition + Re-ID)
    - Camera handover logic
    - Database persistence
    """

    def __init__(
        self,
        topology_config: CameraTopologyConfig,
        reid_config: ReIDConfig,
        identity_linker: IdentityLinker,
        repository: Repository,
        zones_config: Optional[ZonesConfig] = None,
    ):
        """Initialize global track manager.

        Args:
            topology_config: Camera topology configuration
            reid_config: Re-ID configuration
            identity_linker: Identity linker instance
            repository: Database repository
            zones_config: Optional polygon zones configuration for zone-based handover
        """
        self.topology_config = topology_config
        self.reid_config = reid_config
        self.identity_linker = identity_linker
        self.repository = repository

        # Zone manager for polygon-based handover (preferred)
        self.zone_manager: Optional[ZoneManager] = None
        if zones_config and zones_config.zones:
            self.zone_manager = ZoneManager(zones_config)
            logger.info("Using zone-based handover")

        # Global tracks: global_track_id -> GlobalTrack
        self._tracks: dict[str, GlobalTrack] = {}

        # Mapping: (camera_id, local_track_id) -> global_track_id
        self._local_to_global: dict[tuple[str, int], str] = {}

        # Pending handovers waiting to be matched
        self._pending_handovers: list[PendingHandover] = []

        # Handover zones (legacy rectangle-based)
        self._handover_zones: list[CameraHandoverZone] = []
        if not self.zone_manager:
            self._load_handover_zones()

        # Track ID counter - load from database to avoid duplicates
        self._next_track_id = self._load_next_track_id()

        # Store frame dimensions per camera for zone checks
        self._frame_dimensions: dict[str, tuple[int, int]] = {}

        # Store last known bbox for each local track (for zone-based handover)
        self._last_bboxes: dict[tuple[str, int], tuple[float, float, float, float]] = {}

    def _load_next_track_id(self) -> int:
        """Load the next track ID from database to avoid duplicates.

        Returns:
            Next available track ID (max existing ID + 1, or 1 if no tracks)
        """
        try:
            with self.repository.get_session() as session:
                from src.database.models import Track
                result = session.query(Track.id).all()
                if not result:
                    return 1

                # Extract numeric IDs from "global_N" format
                max_id = 0
                for (track_id,) in result:
                    if track_id and track_id.startswith("global_"):
                        try:
                            num = int(track_id.replace("global_", ""))
                            max_id = max(max_id, num)
                        except ValueError:
                            pass
                next_id = max_id + 1
                logger.info(f"Loaded next track ID from database: {next_id}")
                return next_id
        except Exception as e:
            logger.warning(f"Failed to load track ID from database: {e}, starting from 1")
            return 1

    def _load_handover_zones(self):
        """Load camera handover zones from config."""
        for overlap in self.topology_config.overlaps:
            zone = CameraHandoverZone(
                camera1_id=overlap.cameras[0],
                camera2_id=overlap.cameras[1],
                cam1_exit_zone=tuple(overlap.cam1_exit_zone),
                cam2_entry_zone=tuple(overlap.cam2_entry_zone),
                max_handover_sec=overlap.max_handover_sec,
            )
            self._handover_zones.append(zone)

            # Also add reverse direction
            reverse_zone = CameraHandoverZone(
                camera1_id=overlap.cameras[1],
                camera2_id=overlap.cameras[0],
                cam1_exit_zone=tuple(overlap.cam2_entry_zone),
                cam2_entry_zone=tuple(overlap.cam1_exit_zone),
                max_handover_sec=overlap.max_handover_sec,
            )
            self._handover_zones.append(reverse_zone)

        logger.info(f"Loaded {len(self._handover_zones)} handover zones")

    def _generate_track_id(self) -> str:
        """Generate a new global track ID."""
        track_id = f"global_{self._next_track_id}"
        self._next_track_id += 1
        return track_id

    def _find_overlapping_track_ids(
        self,
        local_tracks: list[LocalTrack],
        iou_threshold: float,
    ) -> set[int]:
        """Find track IDs that have overlapping bounding boxes with other tracks.

        Args:
            local_tracks: List of local tracks
            iou_threshold: IoU threshold above which tracks are considered overlapping

        Returns:
            Set of track IDs that have significant overlap with at least one other track
        """
        if iou_threshold <= 0 or len(local_tracks) < 2:
            return set()

        overlapping_ids = set()

        for i, track1 in enumerate(local_tracks):
            for track2 in local_tracks[i + 1:]:
                iou = compute_iou(track1.bbox, track2.bbox)
                if iou > iou_threshold:
                    overlapping_ids.add(track1.track_id)
                    overlapping_ids.add(track2.track_id)
                    logger.debug(
                        f"Overlapping tracks: {track1.track_id} and {track2.track_id} "
                        f"(IoU={iou:.3f})"
                    )

        return overlapping_ids

    def process_local_tracks(
        self,
        camera_id: str,
        local_tracks: list[LocalTrack],
        frame: Optional[np.ndarray],
        new_track_ids: list[int],
        lost_track_ids: list[int],
        has_motion: bool = True,
    ) -> GlobalTrackingResult:
        """Process local tracks from a camera and update global state.

        Args:
            camera_id: Camera identifier
            local_tracks: Current local tracks from ByteTrack
            frame: Current frame for Re-ID extraction (can be None in testing if Re-ID not needed)
            new_track_ids: IDs of newly created local tracks
            lost_track_ids: IDs of lost local tracks
            has_motion: Whether motion was detected in this frame

        Returns:
            GlobalTrackingResult with updated state
        """
        current_time = time.time()
        result = GlobalTrackingResult(
            active_tracks=[],
            new_global_tracks=[],
            handovers_completed=[],
            tracks_lost=[],
        )

        # Store frame dimensions for zone checks
        if frame is not None and frame.shape[0] > 10: # Only update if it looks like a real frame
            frame_h, frame_w = frame.shape[:2]
            self._frame_dimensions[camera_id] = (frame_w, frame_h)
        else:
            # Try to get from cache or use defaults for testing
            frame_w, frame_h = self._frame_dimensions.get(camera_id, (1920, 1080))

        # Clean up expired pending handovers
        self._cleanup_pending_handovers(current_time)

        # Find tracks with overlapping bounding boxes (skip Re-ID for these)
        overlapping_track_ids = self._find_overlapping_track_ids(
            local_tracks, self.reid_config.skip_overlapping_iou
        )

        # Process new local tracks
        for local_track_id in new_track_ids:
            local_track = self._find_local_track(local_tracks, local_track_id)
            if local_track is None:
                continue

            # Check if this is a handover match
            handover_match = self._try_handover_match(
                camera_id, local_track, frame, current_time, local_tracks, frame_w, frame_h
            )

            if handover_match:
                # Link to existing global track
                global_track_id, from_camera = handover_match
                self._link_local_to_global(camera_id, local_track_id, global_track_id)

                # Update global track location
                global_track = self._tracks[global_track_id]
                global_track.update_location(camera_id, local_track_id)
                global_track.state = TrackState.TRACKED

                # Transfer identity via linker
                self.identity_linker.transfer_identity(global_track_id, global_track_id)

                result.handovers_completed.append(
                    (global_track_id, from_camera, camera_id)
                )
                logger.info(
                    f"Handover completed: {global_track_id} from {from_camera} to {camera_id}"
                )

            else:
                # Try Re-ID match against recent lost tracks
                with profiler.measure("GlobalTracker.reid_match"):
                    reid_match = self._try_reid_match(camera_id, local_track, frame)

                if reid_match:
                    global_track_id = reid_match
                    self._link_local_to_global(camera_id, local_track_id, global_track_id)

                    global_track = self._tracks[global_track_id]
                    global_track.update_location(camera_id, local_track_id)
                    global_track.state = TrackState.TRACKED

                    logger.info(f"Re-ID match: {global_track_id} reappeared on {camera_id}")

                else:
                    # Create new global track
                    global_track_id = self._create_global_track(
                        camera_id,
                        local_track,
                        frame,
                        has_overlapping_bbox=local_track.track_id in overlapping_track_ids,
                    )
                    result.new_global_tracks.append(global_track_id)

        # Process active local tracks (update Re-ID, face recognition)
        for local_track in local_tracks:
            # Store last bbox for zone-based handover
            self._last_bboxes[(camera_id, local_track.track_id)] = local_track.bbox

            global_track_id = self._local_to_global.get((camera_id, local_track.track_id))
            if global_track_id is None:
                continue

            global_track = self._tracks.get(global_track_id)
            if global_track is None:
                continue

            # Update location and time
            global_track.last_seen = datetime.now()

            # Process for identification
            if local_track.last_crop is not None:
                # Skip Re-ID if this track overlaps with another (num_persons > 1)
                num_persons = 2 if local_track.track_id in overlapping_track_ids else 1
                with profiler.measure("IdentityLinker.process_track"):
                    self.identity_linker.process_track(
                        global_track_id,
                        frame,
                        local_track.last_crop,
                        local_track.bbox,
                        num_persons_in_frame=num_persons,
                        precomputed_reid=(local_track.last_reid_embedding, local_track.last_reid_quality)
                        if local_track.last_reid_embedding is not None else None
                    )

        # Process lost local tracks
        for local_track_id in lost_track_ids:
            self._handle_lost_local_track(camera_id, local_track_id, current_time)

        # Build result
        for global_track in self._tracks.values():
            if global_track.state in (TrackState.TRACKED, TrackState.NEW):
                result.active_tracks.append(global_track)
            elif global_track.state == TrackState.REMOVED:
                result.tracks_lost.append(global_track.track_id)

        return result

    def _find_local_track(
        self, local_tracks: list[LocalTrack], track_id: int
    ) -> Optional[LocalTrack]:
        """Find a local track by ID."""
        for track in local_tracks:
            if track.track_id == track_id:
                return track
        return None

    def _link_local_to_global(
        self, camera_id: str, local_track_id: int, global_track_id: str
    ):
        """Link a local track to a global track."""
        self._local_to_global[(camera_id, local_track_id)] = global_track_id

    def _create_global_track(
        self,
        camera_id: str,
        local_track: LocalTrack,
        frame: np.ndarray,
        has_overlapping_bbox: bool = False,
    ) -> str:
        """Create a new global track.

        Args:
            camera_id: Camera where track was created
            local_track: Local track from ByteTrack
            frame: Current frame
            has_overlapping_bbox: Whether this track overlaps with another

        Returns:
            New global track ID
        """
        global_track_id = self._generate_track_id()

        # Create global track
        global_track = GlobalTrack(
            track_id=global_track_id,
            state=TrackState.TRACKED,
            current_camera_id=camera_id,
            current_local_track_id=local_track.track_id,
            cameras_seen=[camera_id],
        )
        self._tracks[global_track_id] = global_track

        # Link local to global
        self._link_local_to_global(camera_id, local_track.track_id, global_track_id)

        # Register with identity linker
        self.identity_linker.register_track(global_track_id)

        # Initial Re-ID extraction (skip if overlapping with another track)
        if local_track.last_crop is not None:
            num_persons = 2 if has_overlapping_bbox else 1
            self.identity_linker.process_track(
                global_track_id,
                frame,
                local_track.last_crop,
                local_track.bbox,
                num_persons_in_frame=num_persons,
            )

        # Save to database
        self.repository.create_track(global_track_id, camera_id)
        self.repository.create_event(
            camera_id=camera_id,
            event_type="track_created",
            track_id=global_track_id,
        )

        logger.info(f"Created global track: {global_track_id} on {camera_id}")

        return global_track_id

    def _handle_lost_local_track(
        self, camera_id: str, local_track_id: int, current_time: float
    ):
        """Handle a local track being lost.

        Determines if this could be a camera handover or track end.
        """
        global_track_id = self._local_to_global.get((camera_id, local_track_id))
        if global_track_id is None:
            # Clean up bbox
            self._last_bboxes.pop((camera_id, local_track_id), None)
            return

        global_track = self._tracks.get(global_track_id)
        if global_track is None:
            # Clean up bbox
            self._last_bboxes.pop((camera_id, local_track_id), None)
            return

        # Get identity state for Re-ID embedding
        identity_state = self.identity_linker.get_track_state(global_track_id)
        reid_embedding = None
        if identity_state and len(identity_state.reid_gallery) > 0:
            reid_embedding = identity_state.reid_gallery.get_average_embedding()

        # Get last known bbox
        last_bbox = self._last_bboxes.get((camera_id, local_track_id))

        # Zone-based handover (preferred)
        if self.zone_manager and last_bbox:
            frame_dims = self._frame_dimensions.get(camera_id)
            if frame_dims:
                frame_w, frame_h = frame_dims
                zone_name = self.zone_manager.get_person_zone(
                    camera_id, last_bbox, frame_w, frame_h
                )

                if zone_name:
                    zone_config = self.zone_manager.get_zone_config(zone_name)
                    other_cameras = self.zone_manager.get_other_cameras_for_zone(
                        zone_name, camera_id
                    )

                    if other_cameras:
                        # Create pending handover for zone-based system
                        pending = PendingHandover(
                            global_track_id=global_track_id,
                            from_camera=camera_id,
                            zone_name=zone_name,
                            exit_time=current_time,
                            max_handover_sec=zone_config.max_handover_sec if zone_config else 5.0,
                            reid_embedding=reid_embedding,
                            last_bbox=last_bbox,
                        )
                        self._pending_handovers.append(pending)
                        global_track.state = TrackState.LOST
                        logger.debug(
                            f"Track {global_track_id} pending zone handover from {camera_id} "
                            f"in zone '{zone_name}' (potential cameras: {other_cameras})"
                        )

                        # Clean up mappings
                        del self._local_to_global[(camera_id, local_track_id)]
                        self._last_bboxes.pop((camera_id, local_track_id), None)
                        return

        # Legacy rectangle-based handover (fallback)
        if not self.zone_manager:
            exit_zone = self._check_exit_zone(camera_id)
            if exit_zone:
                pending = PendingHandover(
                    global_track_id=global_track_id,
                    from_camera=camera_id,
                    zone_name=exit_zone,  # Use camera name as zone name for legacy
                    exit_time=current_time,
                    max_handover_sec=3.0,
                    reid_embedding=reid_embedding,
                    last_bbox=last_bbox,
                )
                self._pending_handovers.append(pending)
                global_track.state = TrackState.LOST
                logger.debug(
                    f"Track {global_track_id} pending handover from {camera_id} to {exit_zone}"
                )

                del self._local_to_global[(camera_id, local_track_id)]
                self._last_bboxes.pop((camera_id, local_track_id), None)
                return

        # Mark as lost (may be re-identified via Re-ID later)
        global_track.state = TrackState.LOST

        # Remove local-to-global mapping
        del self._local_to_global[(camera_id, local_track_id)]
        self._last_bboxes.pop((camera_id, local_track_id), None)

    def _check_exit_zone(self, camera_id: str) -> Optional[str]:
        """Check if camera has a handover zone and return target camera.

        TODO: This should check actual track position against zones.
        For now, returns the first connected camera if any.
        """
        for zone in self._handover_zones:
            if zone.camera1_id == camera_id:
                return zone.camera2_id
        return None

    def _try_handover_match(
        self,
        camera_id: str,
        local_track: LocalTrack,
        frame: np.ndarray,
        current_time: float,
        all_local_tracks: list[LocalTrack],
        frame_w: int,
        frame_h: int,
    ) -> Optional[tuple[str, str]]:
        """Try to match new track against pending handovers.

        Args:
            camera_id: Camera identifier
            local_track: New local track to match
            frame: Current frame
            current_time: Current timestamp
            all_local_tracks: All local tracks on this camera (for single-person check)
            frame_w: Frame width in pixels
            frame_h: Frame height in pixels

        Returns:
            Tuple of (global_track_id, from_camera) if matched, else None
        """
        if not self._pending_handovers:
            return None

        # Zone-based handover matching
        if self.zone_manager and self.topology_config.use_zones:
            # Check what zone the new track is in
            zone_name = self.zone_manager.get_person_zone(
                camera_id, local_track.bbox, frame_w, frame_h
            )

            if zone_name:
                # Check if alone in zone (required for handover)
                persons_in_zone = self.zone_manager.get_persons_in_zone(
                    zone_name, camera_id, all_local_tracks, frame_w, frame_h
                )
                if len(persons_in_zone) > 1:
                    logger.debug(
                        f"Multiple persons ({len(persons_in_zone)}) in zone '{zone_name}', "
                        f"skipping handover"
                    )
                    return None

                # Find matching pending handover for this zone
                for pending in self._pending_handovers:
                    if pending.zone_name != zone_name:
                        continue
                    if pending.from_camera == camera_id:
                        continue  # Can't handover to same camera

                    # Check time window
                    time_since_exit = current_time - pending.exit_time
                    if time_since_exit > pending.max_handover_sec:
                        continue

                    # Match using Re-ID if available
                    if pending.reid_embedding is not None:
                        # Use pre-computed Re-ID from local track if available
                        if local_track.last_reid_embedding is not None:
                            embedding = local_track.last_reid_embedding
                            quality = local_track.last_reid_quality
                        elif local_track.last_crop is not None:
                            # Fallback to extraction if not pre-computed (should be rare)
                            with profiler.measure("GlobalTracker.handover_reid"):
                                embedding, quality = self.identity_linker.reid_extractor.extract(
                                    local_track.last_crop, return_quality=True
                                )
                        else:
                            continue

                        if quality >= self.reid_config.min_visibility:
                            from src.recognition.reid_extractor import cosine_similarity

                            similarity = cosine_similarity(embedding, pending.reid_embedding)

                            if similarity > self.reid_config.similarity_threshold:
                                # Match found
                                logger.info(
                                    f"Zone handover match: zone='{zone_name}', "
                                    f"similarity={similarity:.2f}"
                                )
                                self._pending_handovers.remove(pending)
                                return (pending.global_track_id, pending.from_camera)
                    else:
                        # No Re-ID available, accept based on timing and zone alone
                        self._pending_handovers.remove(pending)
                        logger.info(
                            f"Zone handover match (no Re-ID): zone='{zone_name}'"
                        )
                        return (pending.global_track_id, pending.from_camera)

            return None

        # Legacy rectangle-based handover matching
        for pending in self._pending_handovers:
            # For legacy, zone_name contains the target camera ID
            if pending.zone_name != camera_id:
                continue

            # Check time window
            time_since_exit = current_time - pending.exit_time
            zone = self._find_handover_zone(pending.from_camera, camera_id)
            if zone and time_since_exit > zone.max_handover_sec:
                continue

            # Match using Re-ID if available
            if pending.reid_embedding is not None:
                # Use pre-computed Re-ID from local track if available
                if local_track.last_reid_embedding is not None:
                    embedding = local_track.last_reid_embedding
                    quality = local_track.last_reid_quality
                elif local_track.last_crop is not None:
                    with profiler.measure("GlobalTracker.handover_reid"):
                        embedding, quality = self.identity_linker.reid_extractor.extract(
                            local_track.last_crop, return_quality=True
                        )
                else:
                    continue

                if quality >= self.reid_config.min_visibility:
                    from src.recognition.reid_extractor import cosine_similarity

                    similarity = cosine_similarity(embedding, pending.reid_embedding)

                    if similarity > self.reid_config.similarity_threshold:
                        self._pending_handovers.remove(pending)
                        return (pending.global_track_id, pending.from_camera)

            else:
                # No Re-ID available, accept based on timing alone
                self._pending_handovers.remove(pending)
                return (pending.global_track_id, pending.from_camera)

        return None

    def _find_handover_zone(
        self, from_camera: str, to_camera: str
    ) -> Optional[CameraHandoverZone]:
        """Find handover zone between two cameras."""
        for zone in self._handover_zones:
            if zone.camera1_id == from_camera and zone.camera2_id == to_camera:
                return zone
        return None

    def _try_reid_match(
        self,
        camera_id: str,
        local_track: LocalTrack,
        frame: np.ndarray,
    ) -> Optional[str]:
        """Try to match new track against recently lost tracks using Re-ID.

        Returns:
            Global track ID if matched, else None
        """
        if local_track.last_crop is None:
            return None

        # Get recently lost tracks
        candidate_track_ids = []
        for global_track_id, global_track in self._tracks.items():
            if (
                global_track.state == TrackState.LOST
            ):
                # Check if track was lost recently
                time_since_lost = (
                    datetime.now() - global_track.last_seen
                ).total_seconds()
                if time_since_lost < self.reid_config.global_id_grace_period:
                    candidate_track_ids.append(global_track_id)

        if not candidate_track_ids:
            return None

        # Use identity linker's Re-ID matching
        match = self.identity_linker.match_reid_cross_camera(
            f"temp_{camera_id}_{local_track.track_id}",
            local_track.last_crop,
            candidate_track_ids,
            precomputed_reid=(local_track.last_reid_embedding, local_track.last_reid_quality)
            if local_track.last_reid_embedding is not None else None
        )

        if match:
            return match[0]  # Return matched global track ID

        return None

    def _cleanup_pending_handovers(self, current_time: float):
        """Remove expired pending handovers."""
        to_remove = []
        for pending in self._pending_handovers:
            # Use per-handover timeout (from zone config)
            if current_time - pending.exit_time > pending.max_handover_sec:
                to_remove.append(pending)
                # Mark the global track as truly lost
                if pending.global_track_id in self._tracks:
                    self._tracks[pending.global_track_id].state = TrackState.REMOVED

        for pending in to_remove:
            self._pending_handovers.remove(pending)
            logger.debug(
                f"Expired pending handover for {pending.global_track_id} "
                f"in zone '{pending.zone_name}'"
            )

    def get_global_track(self, global_track_id: str) -> Optional[GlobalTrack]:
        """Get a global track by ID."""
        return self._tracks.get(global_track_id)

    def get_global_track_for_local(
        self, camera_id: str, local_track_id: int
    ) -> Optional[GlobalTrack]:
        """Get global track for a local track."""
        global_track_id = self._local_to_global.get((camera_id, local_track_id))
        if global_track_id:
            return self._tracks.get(global_track_id)
        return None

    def get_active_tracks(self) -> list[GlobalTrack]:
        """Get all active global tracks."""
        return [
            t for t in self._tracks.values()
            if t.state in (TrackState.TRACKED, TrackState.NEW)
        ]

    def has_active_tracks(self, camera_id: str) -> bool:
        """Check if there are active tracks on a specific camera.

        Args:
            camera_id: Camera identifier

        Returns:
            True if there are active tracks on this camera
        """
        for track in self._tracks.values():
            if (
                track.state in (TrackState.TRACKED, TrackState.NEW)
                and track.current_camera_id == camera_id
            ):
                return True
        return False

    def get_occupancy(self) -> dict:
        """Get current occupancy information.

        Returns:
            Dictionary with occupancy info
        """
        active_tracks = self.get_active_tracks()
        identified = []
        unknown = 0

        for track in active_tracks:
            state = self.identity_linker.get_track_state(track.track_id)
            if state and state.is_identified:
                person = self.repository.get_person(state.person_id)
                if person:
                    identified.append(person.name)
            else:
                unknown += 1

        return {
            "home": len(active_tracks) > 0,
            "persons": list(set(identified)),
            "unknown_count": unknown,
            "total_tracks": len(active_tracks),
        }

    def cleanup_old_tracks(self, max_age_hours: int = 24):
        """Archive old tracks."""
        to_remove = []
        cutoff = datetime.now()

        for track_id, track in self._tracks.items():
            if track.state == TrackState.REMOVED:
                to_remove.append(track_id)
            elif track.state == TrackState.LOST:
                hours_since_seen = (cutoff - track.last_seen).total_seconds() / 3600
                if hours_since_seen > max_age_hours:
                    to_remove.append(track_id)

        for track_id in to_remove:
            self.identity_linker.unregister_track(track_id)
            del self._tracks[track_id]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old global tracks")
