"""Global track manager for cross-camera person tracking.

Manages global tracks that span multiple cameras, handling:
1. Creation of global tracks from local tracks
2. Re-ID gallery matching for cross-camera identification
3. Zone-based identity propagation for simultaneous views
4. Track lifecycle management
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from src.config import CameraTopologyConfig, ReIDConfig, SnapshotConfig, ZonesConfig
from src.database.repository import Repository
from src.detection.person_detector import compute_iou
from src.recognition.identity_linker import IdentityLinker
from src.recognition.identification_manager import CrossCameraMatch
from src.tracking.track import GlobalTrack, LocalTrack, TrackState
from src.tracking.zone_manager import ZoneManager
from src.utils.profiler import profiler

logger = logging.getLogger(__name__)


@dataclass
class GlobalTrackingResult:
    """Result of global tracking update."""

    active_tracks: list[GlobalTrack]
    new_global_tracks: list[str]
    tracks_lost: list[str]


class GlobalTrackManager:
    """Manages global tracks across all cameras.

    Coordinates between:
    - Local ByteTrack instances (per-camera)
    - Identity linker (face recognition + Re-ID)
    - Zone-based identity propagation
    - Database persistence
    """

    def __init__(
        self,
        topology_config: CameraTopologyConfig,
        reid_config: ReIDConfig,
        identity_linker: IdentityLinker,
        repository: Repository,
        zones_config: Optional[ZonesConfig] = None,
        snapshot_config: Optional[SnapshotConfig] = None,
    ):
        """Initialize global track manager.

        Args:
            topology_config: Camera topology configuration
            reid_config: Re-ID configuration
            identity_linker: Identity linker instance
            repository: Database repository
            zones_config: Optional polygon zones configuration for zone-based propagation
            snapshot_config: Optional snapshot configuration for event images
        """
        self.topology_config = topology_config
        self.reid_config = reid_config
        self.identity_linker = identity_linker
        self.repository = repository
        self.snapshot_config = snapshot_config

        # Zone manager for zone-based identity propagation
        self.zone_manager: Optional[ZoneManager] = None
        if zones_config and zones_config.zones:
            self.zone_manager = ZoneManager(zones_config)
            logger.info("Using zone-based identity propagation")

        # Global tracks: global_track_id -> GlobalTrack
        self._tracks: dict[str, GlobalTrack] = {}

        # Mapping: (camera_id, local_track_id) -> global_track_id
        self._local_to_global: dict[tuple[str, int], str] = {}

        # Track ID counter - load from database to avoid duplicates
        self._next_track_id = self._load_next_track_id()

        # Store frame dimensions per camera for zone checks
        self._frame_dimensions: dict[str, tuple[int, int]] = {}

        # Store last known bbox for each local track (for zone propagation)
        self._last_bboxes: dict[tuple[str, int], tuple[float, float, float, float]] = {}

        # Index of recently-lost track IDs for O(1) lookup in Re-ID matching
        # Avoids O(N) iteration over all tracks
        self._recently_lost_tracks: set[str] = set()

        # Current zone per global track (zone tracking)
        self._track_zones: dict[str, Optional[str]] = {}

        # Throttle for cross-camera zone identity propagation
        self._last_cross_camera_check: float = 0.0

    def _save_event_snapshot(self, crop: np.ndarray, event_type: str, label: str = "") -> Optional[str]:
        """Save a person crop as an event snapshot.

        Returns:
            Path to saved snapshot or None if disabled/failed.
        """
        if not self.snapshot_config or not self.snapshot_config.enabled:
            return None
        if not self.snapshot_config.save_on_new_track and event_type == "track_created":
            return None

        try:
            import uuid
            from pathlib import Path
            from src.utils.image_utils import write_jpeg

            snapshot_dir = Path(self.snapshot_config.path) / event_type
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            unique_id = uuid.uuid4().hex[:8]
            label_part = f"_{label}" if label else ""
            filename = f"{timestamp}{label_part}_{unique_id}.jpg"

            write_jpeg(str(snapshot_dir / filename), crop)
            return str(snapshot_dir / filename)
        except Exception:
            logger.debug(f"Failed to save {event_type} snapshot", exc_info=True)
            return None

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
            tracks_lost=[],
        )

        # Store frame dimensions for zone checks
        if frame is not None and frame.shape[0] > 10: # Only update if it looks like a real frame
            frame_h, frame_w = frame.shape[:2]
            self._frame_dimensions[camera_id] = (frame_w, frame_h)
        else:
            # Try to get from cache or use defaults for testing
            frame_w, frame_h = self._frame_dimensions.get(camera_id, (1920, 1080))

        # Clean up stale LOST tracks
        self._cleanup_lost_tracks(current_time)

        # Find tracks with overlapping bounding boxes (skip Re-ID for these)
        overlapping_track_ids = self._find_overlapping_track_ids(
            local_tracks, self.reid_config.skip_overlapping_iou
        )

        # Process new local tracks
        for local_track_id in new_track_ids:
            local_track = self._find_local_track(local_tracks, local_track_id)
            if local_track is None:
                continue

            # Try gallery match for immediate identity assignment
            with profiler.measure("GlobalTracker.gallery_match"):
                gallery_match = self._try_gallery_match(camera_id, local_track)

            if gallery_match and gallery_match.person_name:
                # Gallery match — create new track with immediate identity
                global_track_id = self._create_global_track(
                    camera_id,
                    local_track,
                    frame,
                    has_overlapping_bbox=local_track.track_id in overlapping_track_ids,
                )
                result.new_global_tracks.append(global_track_id)

                # Assign identity from gallery match
                person = self.repository.get_person_by_name(gallery_match.person_name)
                if person:
                    state = self.identity_linker.get_track_state(global_track_id)
                    if state:
                        state.confirm_identity(
                            person.id, gallery_match.similarity, "reid_gallery"
                        )
                        # Update database
                        self.repository.update_track(
                            global_track_id, person_id=person.id
                        )

                logger.info(
                    f"Re-ID gallery: new track {global_track_id} on {camera_id} "
                    f"identified as {gallery_match.person_name} (sim={gallery_match.similarity:.2f})"
                )

            else:
                # No match — create fresh unidentified track
                global_track_id = self._create_global_track(
                    camera_id,
                    local_track,
                    frame,
                    has_overlapping_bbox=local_track.track_id in overlapping_track_ids,
                )
                result.new_global_tracks.append(global_track_id)

        # Process active local tracks (update Re-ID, face recognition)
        for local_track in local_tracks:
            # Store last bbox for zone propagation
            self._last_bboxes[(camera_id, local_track.track_id)] = local_track.bbox

            global_track_id = self._local_to_global.get((camera_id, local_track.track_id))
            if global_track_id is None:
                continue

            global_track = self._tracks.get(global_track_id)
            if global_track is None:
                continue

            # If ByteTrack is reporting this track but we marked it LOST,
            # detection recovered — restore to TRACKED
            if global_track.state == TrackState.LOST:
                global_track.state = TrackState.TRACKED
                self._recently_lost_tracks.discard(global_track_id)
                if self.reid_config.log_track_recovery:
                    self.repository.create_event(
                        camera_id=camera_id,
                        event_type="track_recovered",
                        track_id=global_track_id,
                        person_id=global_track.person_id,
                        extra_data={
                            "reason": "bytetrack_re_detected",
                        }
                    )

                logger.info(
                    f"Track {global_track_id} recovered on {camera_id} "
                    f"(ByteTrack re-detected after brief loss)"
                )

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
                        if local_track.last_reid_embedding is not None else None,
                        camera_id=camera_id,
                    )

        # Process lost local tracks
        for local_track_id in lost_track_ids:
            self._handle_lost_local_track(camera_id, local_track_id, current_time)

        # Zone tracking + cross-camera identity propagation
        if self.zone_manager and current_time - self._last_cross_camera_check > self.reid_config.cross_camera_interval:
            self._last_cross_camera_check = current_time
            zone_camera_tracks = self._update_track_zones()
            if self.topology_config.enable_cross_camera_propagation:
                self._try_zone_identity_propagation(zone_camera_tracks)

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

    def _cleanup_mappings_for_track(self, global_track_id: str):
        """Remove all local-to-global mappings pointing to this global track."""
        keys_to_remove = [
            key for key, gid in self._local_to_global.items()
            if gid == global_track_id
        ]
        for key in keys_to_remove:
            del self._local_to_global[key]
            self._last_bboxes.pop(key, None)

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
                camera_id=camera_id,
            )

        # Save to database
        snapshot_path = None
        if local_track.last_crop is not None:
            snapshot_path = self._save_event_snapshot(
                local_track.last_crop, "track_created", global_track_id
            )

        # Compute initial zone
        initial_zone = None
        if self.zone_manager:
            frame_dims = self._frame_dimensions.get(camera_id)
            if frame_dims:
                frame_w, frame_h = frame_dims
                initial_zone = self.zone_manager.get_person_zone(
                    camera_id, local_track.bbox, frame_w, frame_h
                )
                self._track_zones[global_track_id] = initial_zone

        self.repository.create_track(global_track_id, camera_id)

        # Create track sighting with entry zone
        try:
            self.repository.create_track_sighting(
                global_track_id, camera_id, entry_zone=initial_zone
            )
        except Exception as e:
            logger.debug(f"Failed to create sighting for {global_track_id}: {e}")

        # Persist initial zone to extra_data
        if initial_zone:
            try:
                self.repository.update_track(
                    global_track_id, extra_data={"zone": initial_zone}
                )
            except Exception:
                pass

        self.repository.create_event(
            camera_id=camera_id,
            event_type="track_created",
            track_id=global_track_id,
            snapshot_path=snapshot_path,
            extra_data={
                "origin": "detection",
                "local_track_id": local_track.track_id
            }
        )

        logger.info(f"Created global track: {global_track_id} on {camera_id}")

        return global_track_id

    def _handle_lost_local_track(
        self, camera_id: str, local_track_id: int, current_time: float
    ):
        """Handle a local track being lost.

        Flushes Re-ID gallery to shared gallery and marks the track as LOST.
        The mapping is kept alive so ByteTrack re-detection can recover the track.
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

        # Flush track data to shared Re-ID gallery so it's available for
        # cross-camera matching immediately
        identity_state = self.identity_linker.get_track_state(global_track_id)
        was_face_identified = (
            identity_state is not None
            and identity_state.is_identified
            and identity_state.identified_by == "face"
        )
        if self.identity_linker.reid_gallery_manager:
            track_id_num = hash(global_track_id) % (10**9)
            self.identity_linker.reid_gallery_manager.on_track_lost(
                track_id_num, was_face_identified=was_face_identified
            )
            if was_face_identified:
                logger.debug(
                    f"Flushed Re-ID gallery for face-identified track {global_track_id}"
                )

        # End track sighting with exit zone and set estimated location
        last_zone = self._track_zones.get(global_track_id)
        try:
            self.repository.end_track_sighting(
                global_track_id, camera_id, exit_zone=last_zone
            )
        except Exception as e:
            logger.debug(f"Failed to end sighting for {global_track_id}: {e}")

        # Compute estimated zone (exit destination if configured, else last zone)
        estimated_zone = last_zone
        if last_zone and self.zone_manager:
            exit_dest = self.zone_manager.get_exit_destination(last_zone)
            if exit_dest:
                estimated_zone = exit_dest
        if estimated_zone:
            try:
                self.repository.update_track(
                    global_track_id, extra_data={"estimated_zone": estimated_zone}
                )
            except Exception as e:
                logger.debug(f"Failed to persist estimated_zone for {global_track_id}: {e}")

        # Mark as lost — keep _local_to_global mapping alive so ByteTrack
        # re-detection of the same local ID can recover the global track.
        # Mapping is cleaned up when the track transitions to REMOVED.
        global_track.state = TrackState.LOST
        self._recently_lost_tracks.add(global_track_id)

    def _try_gallery_match(
        self,
        camera_id: str,
        local_track: LocalTrack,
    ) -> Optional[CrossCameraMatch]:
        """Try to match new track against the shared Re-ID gallery.

        Returns:
            CrossCameraMatch if matched, else None
        """
        if local_track.last_crop is None:
            return None

        if not self.identity_linker.reid_gallery_manager:
            return None

        with profiler.measure("ReIDGallery.match"):
            match_result = self.identity_linker.reid_gallery_manager.match_new_track(
                local_track.last_crop, 1
            )

        if not match_result.matched:
            return None

        # Emit reid_match event
        if self.repository:
            person = self.repository.get_person_by_name(match_result.person_name)
            person_id = person.id if person else None
            snapshot_path = self._save_event_snapshot(
                local_track.last_crop, "reid_match", match_result.person_name
            )
            self.repository.create_event(
                camera_id=camera_id,
                event_type="reid_match",
                track_id=f"temp_{camera_id}_{local_track.track_id}",
                person_id=person_id,
                confidence=match_result.score,
                reid_embedding_id=match_result.db_id,
                snapshot_path=snapshot_path,
                extra_data={
                    "match_policy": "gallery",
                    "threshold": self.reid_config.similarity_threshold,
                    "top1_score": match_result.best_score,
                    "original_reid_embedding_id": match_result.db_id,
                }
            )

        return CrossCameraMatch(
            matched_track_id=None,
            similarity=match_result.score,
            person_name=match_result.person_name,
        )

    def _cleanup_lost_tracks(self, current_time: float):
        """Retire stale LOST tracks that exceeded the grace period.

        Any LOST global track that hasn't reappeared within the Re-ID grace
        period (reid.global_id_grace_period) is marked REMOVED so it can be
        fully cleaned up by periodic cleanup.
        """
        if not self._tracks:
            return

        cutoff = datetime.now()
        grace = self.reid_config.global_id_grace_period
        for track_id, track in list(self._tracks.items()):
            if track.state == TrackState.LOST:
                time_since_seen = (cutoff - track.last_seen).total_seconds()
                if time_since_seen > grace:
                    track.state = TrackState.REMOVED
                    self._recently_lost_tracks.discard(track_id)
                    self._track_zones.pop(track_id, None)
                    self._cleanup_mappings_for_track(track_id)

                    # Emit track_lost event
                    self.repository.create_event(
                        camera_id=track.current_camera_id or "unknown",
                        event_type="track_lost",
                        track_id=track_id,
                        person_id=track.person_id,
                        extra_data={
                            "reason": "stale",
                            "age_sec": time_since_seen
                        }
                    )

                    # Sync to database immediately so UI reflects it
                    try:
                        self.repository.update_track(track_id, status="archived")

                        # Emit track_archived event
                        self.repository.create_event(
                            camera_id=track.current_camera_id or "unknown",
                            event_type="track_archived",
                            track_id=track_id,
                            person_id=track.person_id,
                            extra_data={
                                "age_sec": time_since_seen
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update track status in DB: {e}")

                    logger.debug(
                        f"Track {track_id} exceeded LOST grace ({time_since_seen:.1f}s > {grace:.1f}s), marked REMOVED and archived in DB"
                    )

    def _update_track_zones(self) -> dict[str, dict[str, list[tuple[str, tuple]]]]:
        """Update zone for each active track and persist on zone changes.

        Iterates active local-to-global mappings, computes zone via ZoneManager,
        updates _track_zones, and persists to Track.extra_data["zone"] on change.

        Returns:
            zone_camera_tracks map for use by identity propagation
        """
        zone_camera_tracks: dict[str, dict[str, list[tuple[str, tuple]]]] = {}

        if not self.zone_manager:
            return zone_camera_tracks

        for (camera_id, local_track_id), global_track_id in self._local_to_global.items():
            global_track = self._tracks.get(global_track_id)
            if global_track is None or global_track.state not in (TrackState.TRACKED, TrackState.NEW):
                continue

            bbox = self._last_bboxes.get((camera_id, local_track_id))
            if bbox is None:
                continue

            frame_dims = self._frame_dimensions.get(camera_id)
            if frame_dims is None:
                continue

            frame_w, frame_h = frame_dims
            zone_name = self.zone_manager.get_person_zone(camera_id, bbox, frame_w, frame_h)

            # Update in-memory zone and persist on change
            prev_zone = self._track_zones.get(global_track_id)
            if zone_name != prev_zone:
                self._track_zones[global_track_id] = zone_name
                if zone_name is not None:
                    try:
                        self.repository.update_track(
                            global_track_id, extra_data={"zone": zone_name}
                        )
                    except Exception as e:
                        logger.debug(f"Failed to persist zone for {global_track_id}: {e}")

            if zone_name is None:
                continue

            if zone_name not in zone_camera_tracks:
                zone_camera_tracks[zone_name] = {}
            if camera_id not in zone_camera_tracks[zone_name]:
                zone_camera_tracks[zone_name][camera_id] = []
            zone_camera_tracks[zone_name][camera_id].append((global_track_id, bbox))

        return zone_camera_tracks

    def get_track_zone(self, global_track_id: str) -> Optional[str]:
        """Get the current zone for a global track."""
        return self._track_zones.get(global_track_id)

    def _try_zone_identity_propagation(
        self, zone_camera_tracks: dict[str, dict[str, list[tuple[str, tuple]]]]
    ):
        """Propagate identity between simultaneously active tracks in shared zones.

        For each zone visible on multiple cameras, if camera A sees exactly 1 person
        and camera B sees exactly 1 person, and one is face-identified while the other
        is unidentified, transfer identity via handover method.
        """
        if not self.zone_manager:
            return

        # For each zone, check camera pairs
        for zone_name, cameras in zone_camera_tracks.items():
            camera_ids = list(cameras.keys())
            for i in range(len(camera_ids)):
                for j in range(i + 1, len(camera_ids)):
                    cam_a = camera_ids[i]
                    cam_b = camera_ids[j]
                    tracks_a = cameras[cam_a]
                    tracks_b = cameras[cam_b]

                    # Both cameras must see exactly 1 person in this zone
                    if len(tracks_a) != 1 or len(tracks_b) != 1:
                        continue

                    track_a_id = tracks_a[0][0]
                    track_b_id = tracks_b[0][0]

                    state_a = self.identity_linker.get_track_state(track_a_id)
                    state_b = self.identity_linker.get_track_state(track_b_id)
                    if state_a is None or state_b is None:
                        continue

                    # Determine which is identified and which is not
                    a_identified = state_a.is_identified
                    b_identified = state_b.is_identified

                    if a_identified and not b_identified:
                        self.identity_linker.transfer_identity(track_a_id, track_b_id)
                        # Update DB
                        if state_a.person_id is not None:
                            self.repository.update_track(track_b_id, person_id=state_a.person_id)
                        self.repository.create_event(
                            camera_id=cam_b,
                            event_type="cross_camera_propagation",
                            track_id=track_b_id,
                            person_id=state_a.person_id,
                            extra_data={
                                "zone": zone_name,
                                "from_track": track_a_id,
                                "from_camera": cam_a,
                            }
                        )
                        logger.info(
                            f"Zone identity propagation: zone='{zone_name}' "
                            f"{track_a_id} -> {track_b_id} (person_id={state_a.person_id})"
                        )
                    elif b_identified and not a_identified:
                        self.identity_linker.transfer_identity(track_b_id, track_a_id)
                        # Update DB
                        if state_b.person_id is not None:
                            self.repository.update_track(track_a_id, person_id=state_b.person_id)
                        self.repository.create_event(
                            camera_id=cam_a,
                            event_type="cross_camera_propagation",
                            track_id=track_a_id,
                            person_id=state_b.person_id,
                            extra_data={
                                "zone": zone_name,
                                "from_track": track_b_id,
                                "from_camera": cam_b,
                            }
                        )
                        logger.info(
                            f"Zone identity propagation: zone='{zone_name}' "
                            f"{track_b_id} -> {track_a_id} (person_id={state_b.person_id})"
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
        """Check if there are active or recently-lost tracks on a camera.

        Returns True when detection should keep running, including:
        - TRACKED/NEW tracks on this camera
        - LOST tracks within the grace period (so ByteTrack can re-detect)

        Args:
            camera_id: Camera identifier

        Returns:
            True if detection should keep running on this camera
        """
        now = datetime.now()
        grace = self.reid_config.global_id_grace_period

        for track in self._tracks.values():
            if track.current_camera_id != camera_id:
                continue
            if track.state in (TrackState.TRACKED, TrackState.NEW):
                return True
            if track.state == TrackState.LOST:
                time_since_seen = (now - track.last_seen).total_seconds()
                if time_since_seen < grace:
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
            self._cleanup_mappings_for_track(track_id)
            del self._tracks[track_id]
            self._recently_lost_tracks.discard(track_id)
            self._track_zones.pop(track_id, None)

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old global tracks")
