"""Shared preview functionality for camera visualization.

This module provides common classes for track rendering, face recognition,
Re-ID management, and zone visualization used by both single and multi-camera previews.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Callable

import cv2
import numpy as np

from src.config import Config, CameraConfig
from src.detection.motion_detector import MotionDetector
from src.detection.person_detector import PersonDetector, DetectionResult
from src.recognition.face_recognizer import FaceRecognizer
from src.recognition.reid_extractor import ReIDExtractor
from src.recognition.reid_gallery import ReIDGalleryManager
from src.tracking.byte_tracker import ByteTracker
from src.tracking.zone_manager import ZoneManager


@dataclass
class TrackIdentity:
    """Identity information for a track."""
    person_name: Optional[str] = None
    confidence: float = 0.0
    is_face_identified: bool = False
    is_reid_identified: bool = False
    reid_score: float = -1.0
    face_info: Optional[tuple[str, float]] = None  # (closest_name, confidence)


class IdentificationManager:
    """Manages face recognition and Re-ID for tracks."""

    def __init__(self, config: Config, face_gallery: list[tuple[int, str, np.ndarray]]):
        """Initialize identification manager.

        Args:
            config: Application configuration
            face_gallery: List of (person_id, person_name, embedding) tuples
        """
        self.config = config
        self.face_gallery = face_gallery

        # Face recognizer
        self.face_recognizer: Optional[FaceRecognizer] = None
        if config.face_recognition.enabled and face_gallery:
            self.face_recognizer = FaceRecognizer(config.face_recognition)

        # Re-ID gallery manager (optional - can be disabled to save memory)
        self.reid_enabled = config.reid.enabled
        self.reid_extractor: Optional[ReIDExtractor] = None
        self.reid_gallery_manager: Optional[ReIDGalleryManager] = None
        if self.reid_enabled:
            self.reid_extractor = ReIDExtractor(config.reid)
            self.reid_gallery_manager = ReIDGalleryManager(
                reid_extractor=self.reid_extractor,
                similarity_threshold=config.reid.similarity_threshold,
                max_reappear_time_sec=config.reid.max_reappear_time_sec,
                max_embeddings_per_person=config.reid.gallery_size,
            )

        # Track identity state (keyed by track_id - can be local or global)
        self._identities: dict[str, TrackIdentity] = {}

        # Configuration
        self.face_check_interval = config.face_recognition.detection_interval
        self.reid_confirm_interval = config.face_recognition.reid_confirmation_interval
        self.reid_embedding_interval = 30
        self.face_threshold = config.face_recognition.similarity_threshold

    def warmup(self):
        """Warm up models."""
        if self.reid_extractor:
            self.reid_extractor.warmup()

    def get_identity(self, track_id: str) -> TrackIdentity:
        """Get or create identity for a track."""
        if track_id not in self._identities:
            self._identities[track_id] = TrackIdentity()
        return self._identities[track_id]

    def is_identified(self, track_id: str) -> bool:
        """Check if track is identified."""
        identity = self._identities.get(track_id)
        return identity is not None and identity.person_name is not None

    def on_track_lost(self, track_id: str, local_track_id: int):
        """Handle track being lost - store Re-ID for later matching."""
        if not self.reid_gallery_manager:
            return
        identity = self._identities.get(track_id)
        was_face_identified = identity.is_face_identified if identity else False
        self.reid_gallery_manager.on_track_lost(local_track_id, was_face_identified)

    def on_track_removed(self, track_id: str):
        """Clean up identity state when track is removed."""
        self._identities.pop(track_id, None)

    def try_reid_match(self, track_id: str, crop: np.ndarray, num_persons: int) -> bool:
        """Try to match a new track against Re-ID gallery.

        Returns:
            True if matched
        """
        if not self.reid_gallery_manager or crop.size == 0:
            return False

        match_result = self.reid_gallery_manager.match_new_track(crop, num_persons)
        identity = self.get_identity(track_id)

        if match_result.score >= 0:
            identity.reid_score = match_result.score

        if match_result.matched:
            identity.person_name = match_result.person_name
            identity.confidence = match_result.score
            identity.is_reid_identified = True
            identity.is_face_identified = False
            return True

        return False

    def try_face_recognition(self, track_id: str, person_crop: np.ndarray,
                             local_track_id: int, num_persons: int) -> bool:
        """Try face recognition on a track.

        Returns:
            True if face identified
        """
        if not self.face_recognizer or not self.face_gallery:
            return False

        if person_crop.size == 0:
            return False

        identity = self.get_identity(track_id)

        # Skip if already face-identified
        if identity.is_face_identified:
            return False

        face_result = self.face_recognizer.detect_faces(person_crop)
        if not face_result.faces:
            return False

        face = face_result.faces[0]
        if face.embedding is None:
            return False

        # Find best match
        best_score = 0.0
        best_name = None
        for pid, name, emb in self.face_gallery:
            score = self.face_recognizer.compare_embeddings(face.embedding, emb)
            if score > best_score:
                best_score = score
                best_name = name

        # Store face info even if below threshold
        if best_name:
            identity.face_info = (best_name, best_score)

        # Check if above threshold
        if best_score >= self.face_threshold:
            identity.person_name = best_name
            identity.confidence = best_score
            identity.is_face_identified = True
            identity.is_reid_identified = False
            identity.face_info = None

            # Update Re-ID gallery with face-confirmed identity
            self.reid_gallery_manager.update_track_embedding(
                local_track_id, person_crop, best_name, num_persons
            )
            return True

        return False

    def update_reid_embedding(self, track_id: str, local_track_id: int,
                              crop: np.ndarray, num_persons: int):
        """Update Re-ID embedding for a face-identified track."""
        if not self.reid_gallery_manager:
            return
        identity = self._identities.get(track_id)
        if identity and identity.is_face_identified and crop.size > 0:
            self.reid_gallery_manager.update_track_embedding(
                local_track_id, crop, identity.person_name, num_persons
            )

    def cleanup_expired(self):
        """Clean up expired Re-ID gallery entries."""
        if self.reid_gallery_manager:
            self.reid_gallery_manager.cleanup_expired()

    @property
    def gallery_size(self) -> int:
        """Get Re-ID gallery size."""
        if self.reid_gallery_manager:
            return self.reid_gallery_manager.gallery_size
        return 0


class TrackRenderer:
    """Renders tracks with labels and bounding boxes."""

    # Colors
    COLOR_FACE_IDENTIFIED = (255, 150, 0)    # Blue for face identified
    COLOR_REID_IDENTIFIED = (255, 100, 100)  # Light blue for Re-ID identified
    COLOR_UNIDENTIFIED = (0, 255, 0)         # Green for unidentified
    COLOR_MULTI_CAMERA = (0, 255, 255)       # Yellow for multi-camera
    COLOR_STATIONARY = (0, 200, 255)         # Orange for stationary

    def __init__(self, zone_manager: Optional[ZoneManager] = None):
        """Initialize track renderer.

        Args:
            zone_manager: Optional zone manager for zone labels
        """
        self.zone_manager = zone_manager

    def get_track_label_and_color(
        self,
        track_id: str,
        identity: TrackIdentity,
        camera_id: str,
        bbox: tuple[float, float, float, float],
        frame_w: int,
        frame_h: int,
        stationary_time: Optional[int] = None,
        cameras_seen: Optional[list[str]] = None,
        show_reid_score: bool = True,
    ) -> tuple[str, tuple[int, int, int]]:
        """Get label and color for a track.

        Args:
            track_id: Track identifier (local or global)
            identity: Track identity information
            camera_id: Current camera ID
            bbox: Bounding box
            frame_w: Frame width
            frame_h: Frame height
            stationary_time: Seconds stationary (None if not stationary)
            cameras_seen: List of cameras this track has been seen on
            show_reid_score: Whether to show Re-ID score for unidentified tracks

        Returns:
            Tuple of (label, color)
        """
        is_multi_camera = cameras_seen and len(cameras_seen) > 1

        if identity.person_name:
            # Identified
            if identity.is_reid_identified:
                label = f"{identity.person_name} (R:{identity.confidence:.2f})"
                color = self.COLOR_REID_IDENTIFIED
            else:
                label = f"{identity.person_name} (F:{identity.confidence:.2f})"
                color = self.COLOR_FACE_IDENTIFIED
        else:
            # Unidentified
            # Clean track ID for display
            display_id = track_id.replace('global_', '') if 'global_' in str(track_id) else str(track_id)
            label = f"#{display_id}"

            # Add face info (closest match even if below threshold)
            if identity.face_info:
                closest_name, face_conf = identity.face_info
                label += f" ~{closest_name}({face_conf:.2f})"

            # Always show Re-ID score if available (for debugging/monitoring)
            if identity.reid_score >= 0:
                label += f" R:{identity.reid_score:.2f}"

            color = self.COLOR_UNIDENTIFIED

        # Add zone info
        if self.zone_manager:
            track_zone = self.zone_manager.get_person_zone(camera_id, bbox, frame_w, frame_h)
            if track_zone:
                label += f" [{track_zone}]"

        # Add multi-camera indicator
        if is_multi_camera:
            label += " *"
            if not identity.person_name:
                color = self.COLOR_MULTI_CAMERA

        # Add stationary indicator
        if stationary_time is not None:
            label += f" S:{stationary_time}s"
            if not identity.person_name:
                color = self.COLOR_STATIONARY

        return label, color

    def draw_track(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
        label: str,
        color: tuple[int, int, int],
    ):
        """Draw a track bounding box and label on the frame.

        Args:
            frame: Frame to draw on (modified in place)
            bbox: Bounding box (x1, y1, x2, y2)
            label: Label text
            color: BGR color tuple
        """
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


class ZoneRenderer:
    """Renders polygon zones on frames."""

    ZONE_COLORS = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]

    def __init__(self, zone_manager: ZoneManager):
        """Initialize zone renderer.

        Args:
            zone_manager: Zone manager instance
        """
        self.zone_manager = zone_manager

    def draw_zones(self, frame: np.ndarray, camera_id: str, alpha: float = 0.2):
        """Draw all zones for a camera on the frame.

        Args:
            frame: Frame to draw on (modified in place)
            camera_id: Camera identifier
            alpha: Transparency for zone fill (0-1)
        """
        frame_h, frame_w = frame.shape[:2]
        zone_visualizations = self.zone_manager.visualize_zones(camera_id, frame_w, frame_h)

        for i, (polygon_points, zone_name) in enumerate(zone_visualizations):
            color = self.ZONE_COLORS[i % len(self.ZONE_COLORS)]
            pts = np.array(polygon_points, np.int32).reshape((-1, 1, 2))

            # Draw filled polygon with transparency
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Draw polygon outline
            cv2.polylines(frame, [pts], True, color, 2)

            # Draw zone name at centroid
            if polygon_points:
                centroid_x = int(sum(p[0] for p in polygon_points) / len(polygon_points))
                centroid_y = int(sum(p[1] for p in polygon_points) / len(polygon_points))
                cv2.putText(frame, zone_name, (centroid_x - 30, centroid_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


class StationaryTracker:
    """Tracks stationary state of tracks."""

    def __init__(self, timeout: float = 0):
        """Initialize stationary tracker.

        Args:
            timeout: Seconds before removing stationary track (0 = no timeout)
        """
        self.timeout = timeout
        self._stationary_since: dict[tuple[str, int], float] = {}  # (camera_id, track_id) -> timestamp

    def update(self, camera_id: str, track_ids: list[int], has_motion: bool, current_time: float):
        """Update stationary state for tracks.

        Args:
            camera_id: Camera identifier
            track_ids: List of current track IDs
            has_motion: Whether motion was detected
            current_time: Current timestamp
        """
        if has_motion:
            # Motion detected - clear stationary timers for these tracks
            for track_id in track_ids:
                self._stationary_since.pop((camera_id, track_id), None)
        else:
            # No motion - mark tracks as stationary
            for track_id in track_ids:
                if (camera_id, track_id) not in self._stationary_since:
                    self._stationary_since[(camera_id, track_id)] = current_time

    def get_stationary_time(self, camera_id: str, track_id: int, current_time: float) -> Optional[int]:
        """Get how long a track has been stationary.

        Returns:
            Seconds stationary, or None if not stationary
        """
        key = (camera_id, track_id)
        if key in self._stationary_since:
            return int(current_time - self._stationary_since[key])
        return None

    def is_stationary(self, camera_id: str, track_id: int) -> bool:
        """Check if a track is stationary."""
        return (camera_id, track_id) in self._stationary_since

    def cleanup_track(self, camera_id: str, track_id: int):
        """Remove stationary tracking for a track."""
        self._stationary_since.pop((camera_id, track_id), None)

    def cleanup_expired(self, current_time: float) -> list[tuple[str, int]]:
        """Remove expired stationary tracks.

        Returns:
            List of (camera_id, track_id) that were removed
        """
        if self.timeout <= 0:
            return []

        expired = []
        for key, start_time in list(self._stationary_since.items()):
            if current_time - start_time > self.timeout:
                expired.append(key)
                del self._stationary_since[key]

        return expired


def draw_status_bar(frame: np.ndarray, text: str, position: str = "top"):
    """Draw a status bar on the frame.

    Args:
        frame: Frame to draw on
        text: Status text
        position: "top" or "bottom"
    """
    if position == "top":
        y = 30
    else:
        y = frame.shape[0] - 10

    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def draw_motion_indicator(frame: np.ndarray, has_motion: bool):
    """Draw motion indicator circle on frame.

    Args:
        frame: Frame to draw on
        has_motion: Whether motion is detected
    """
    frame_h, frame_w = frame.shape[:2]
    color = (0, 255, 0) if has_motion else (128, 128, 128)
    cv2.circle(frame, (frame_w - 20, 20), 10, color, -1)


@dataclass
class ProcessedTrack:
    """A processed track with identity information."""
    track_id: str  # Can be local or global track ID
    local_track_id: int  # ByteTracker's local ID
    bbox: tuple[float, float, float, float]
    identity: TrackIdentity
    stationary_time: Optional[int] = None
    cameras_seen: Optional[list[str]] = None


@dataclass
class MatchEvent:
    """A face or Re-ID match event."""
    match_type: str  # "face" or "reid"
    track_id: str
    person_name: str
    confidence: float
    camera_id: str
    bbox: tuple[float, float, float, float]
    crop: np.ndarray


@dataclass
class FrameResult:
    """Result of processing a single frame."""
    has_motion: bool
    tracks: list[ProcessedTrack]
    track_count: int
    identified_count: int
    new_track_ids: list[int] = field(default_factory=list)
    lost_track_ids: list[int] = field(default_factory=list)
    match_events: list[MatchEvent] = field(default_factory=list)


class CameraProcessor:
    """Processes frames from a single camera.

    Encapsulates motion detection, person detection, tracking, and identification
    for one camera. Used by both single and multi-camera previews.
    """

    def __init__(
        self,
        camera_id: str,
        config: Config,
        person_detector: PersonDetector,
        id_manager: IdentificationManager,
        exclusion_zones: Optional[list] = None,
        get_global_track_id: Optional[Callable[[str, int], Optional[str]]] = None,
        get_cameras_seen: Optional[Callable[[str], Optional[list[str]]]] = None,
    ):
        """Initialize camera processor.

        Args:
            camera_id: Camera identifier
            config: Application configuration
            person_detector: Shared person detector instance
            id_manager: Shared identification manager
            exclusion_zones: Optional list of exclusion zones for this camera
            get_global_track_id: Optional callback to get global track ID from (camera_id, local_track_id).
                                 If None, local track IDs are used as identity keys.
            get_cameras_seen: Optional callback to get cameras_seen for a global track ID.
        """
        self.camera_id = camera_id
        self.config = config
        self.person_detector = person_detector
        self.id_manager = id_manager
        self.exclusion_zones = exclusion_zones or []

        # Callbacks for global tracking (multi-camera mode)
        self._get_global_track_id = get_global_track_id
        self._get_cameras_seen = get_cameras_seen

        # Per-camera components
        self.motion_detector = MotionDetector(config.motion)
        self.tracker = ByteTracker(camera_id, config.tracking)
        self.stationary_tracker = StationaryTracker(config.tracking.stationary_timeout)

        # Per-camera state
        self._frame_counter = 0
        self._last_tracks: list = []
        self._previous_track_ids: set[int] = set()
        self._warmup_frames = 2  # Run detection on first N frames regardless of motion
        self._detection_skip = config.detection.frame_skip  # Skip detection every N frames
        self._last_detections = None  # Cache last detection result for skipped frames

    def _is_in_exclusion_zone(self, bbox: tuple, frame_w: int, frame_h: int) -> bool:
        """Check if detection center is in any exclusion zone."""
        if not self.exclusion_zones:
            return False
        cx = (bbox[0] + bbox[2]) / 2 / frame_w
        cy = (bbox[1] + bbox[3]) / 2 / frame_h
        for zone in self.exclusion_zones:
            if zone.x1 <= cx <= zone.x2 and zone.y1 <= cy <= zone.y2:
                return True
        return False

    def _get_crop(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        """Extract person crop from frame."""
        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        return frame[max(0, y1):min(frame_h, y2), max(0, x1):min(frame_w, x2)]

    def _get_track_key(self, local_track_id: int) -> str:
        """Get identity key for a track (global ID if available, else local ID)."""
        if self._get_global_track_id:
            global_id = self._get_global_track_id(self.camera_id, local_track_id)
            if global_id:
                return global_id
        return str(local_track_id)

    def process_frame(self, frame: np.ndarray, log_callback: Optional[Callable[[str], None]] = None) -> FrameResult:
        """Process a single frame.

        Args:
            frame: Input frame (BGR)
            log_callback: Optional callback for logging events (e.g., face identification)

        Returns:
            FrameResult with processed tracks and statistics
        """
        current_time = time.time()
        self._frame_counter += 1
        frame_h, frame_w = frame.shape[:2]

        # Motion detection
        motion_result = self.motion_detector.detect(frame)
        has_active_tracks = len(self._last_tracks) > 0

        # During warmup, always run detection to catch stationary objects
        in_warmup = self._frame_counter <= self._warmup_frames

        # No motion and no active tracks - return early (unless in warmup)
        if not motion_result.has_motion and not has_active_tracks and not in_warmup:
            self._last_tracks = []
            return FrameResult(
                has_motion=False,
                tracks=[],
                track_count=0,
                identified_count=0,
            )

        # Person detection (with frame skip to reduce CPU usage)
        # Always detect during warmup, or on scheduled frames, or if we have active tracks
        should_detect = (
            in_warmup or
            (self._frame_counter % self._detection_skip == 0) or
            has_active_tracks
        )

        if should_detect:
            detections = self.person_detector.detect(frame)
            # Filter exclusion zones
            if self.exclusion_zones:
                filtered = [d for d in detections.detections
                            if not self._is_in_exclusion_zone(d.bbox, frame_w, frame_h)]
                detections = DetectionResult(detections=filtered, frame_shape=detections.frame_shape)
            self._last_detections = detections
        else:
            # Use cached detections (or empty if none)
            detections = self._last_detections or DetectionResult(detections=[], frame_shape=frame.shape)

        # Tracking
        track_result = self.tracker.update(detections, frame)
        current_track_ids = {t.track_id for t in track_result.tracks}
        num_persons = len(track_result.tracks)

        # Handle lost tracks
        lost_ids = self._previous_track_ids - current_track_ids
        for lost_id in lost_ids:
            track_key = self._get_track_key(lost_id)
            self.id_manager.on_track_lost(track_key, lost_id)
            self.stationary_tracker.cleanup_track(self.camera_id, lost_id)

        # Collect match events
        match_events = []

        # Process each track
        for local_track in track_result.tracks:
            track_key = self._get_track_key(local_track.track_id)
            crop = self._get_crop(frame, local_track.bbox)

            if crop.size == 0:
                continue

            # Re-ID matching for unidentified tracks
            # - For new tracks: always try to match
            # - For existing tracks: periodically try to match (in case gallery was updated)
            is_new_track = local_track.track_id in track_result.new_track_ids
            should_try_reid = is_new_track or (self._frame_counter % self.id_manager.reid_confirm_interval == 0)

            if should_try_reid and not self.id_manager.is_identified(track_key):
                if self.id_manager.try_reid_match(track_key, crop, num_persons):
                    identity = self.id_manager.get_identity(track_key)
                    if log_callback:
                        log_callback(f"[RE-ID] {identity.person_name} ({track_key}) score={identity.confidence:.2f} on {self.camera_id}")
                    match_events.append(MatchEvent(
                        match_type="reid",
                        track_id=track_key,
                        person_name=identity.person_name,
                        confidence=identity.confidence,
                        camera_id=self.camera_id,
                        bbox=local_track.bbox,
                        crop=crop.copy(),
                    ))

            # Face recognition (periodic, per-camera frame counter)
            if self._frame_counter % self.id_manager.face_check_interval == 0:
                if self.id_manager.try_face_recognition(track_key, crop, local_track.track_id, num_persons):
                    identity = self.id_manager.get_identity(track_key)
                    if log_callback:
                        log_callback(f"[FACE] {identity.person_name} ({track_key}) score={identity.confidence:.2f} on {self.camera_id}")
                    match_events.append(MatchEvent(
                        match_type="face",
                        track_id=track_key,
                        person_name=identity.person_name,
                        confidence=identity.confidence,
                        camera_id=self.camera_id,
                        bbox=local_track.bbox,
                        crop=crop.copy(),
                    ))

            # Update Re-ID embeddings (periodic)
            if self._frame_counter % self.id_manager.reid_embedding_interval == 0:
                self.id_manager.update_reid_embedding(track_key, local_track.track_id, crop, num_persons)

        # Update stationary state
        track_ids = [t.track_id for t in track_result.tracks]
        self.stationary_tracker.update(self.camera_id, track_ids, motion_result.has_motion, current_time)

        # Update state
        if motion_result.has_motion or track_result.tracks:
            self._last_tracks = track_result.tracks
        self._previous_track_ids = current_track_ids

        # Build processed tracks with identity info
        processed_tracks = []
        tracks_to_process = track_result.tracks if track_result.tracks else self._last_tracks
        identified_count = 0

        for local_track in tracks_to_process:
            track_key = self._get_track_key(local_track.track_id)
            identity = self.id_manager.get_identity(track_key)
            stationary_time = self.stationary_tracker.get_stationary_time(
                self.camera_id, local_track.track_id, current_time)

            cameras_seen = None
            if self._get_cameras_seen and self._get_global_track_id:
                global_id = self._get_global_track_id(self.camera_id, local_track.track_id)
                if global_id:
                    cameras_seen = self._get_cameras_seen(global_id)

            if identity.person_name:
                identified_count += 1

            processed_tracks.append(ProcessedTrack(
                track_id=track_key,
                local_track_id=local_track.track_id,
                bbox=local_track.bbox,
                identity=identity,
                stationary_time=stationary_time,
                cameras_seen=cameras_seen,
            ))

        return FrameResult(
            has_motion=motion_result.has_motion,
            tracks=processed_tracks,
            track_count=len(processed_tracks),
            identified_count=identified_count,
            new_track_ids=track_result.new_track_ids,
            lost_track_ids=list(lost_ids),
            match_events=match_events,
        )

    def get_track_result(self):
        """Get the last ByteTracker result for global tracking integration."""
        return self.tracker.get_last_result() if hasattr(self.tracker, 'get_last_result') else None

    @property
    def last_tracks(self) -> list:
        """Get the last processed tracks (ByteTracker tracks)."""
        return self._last_tracks

    @property
    def frame_counter(self) -> int:
        """Get the current frame counter."""
        return self._frame_counter


class SnapshotSaver:
    """Saves snapshots for face matches, Re-ID matches, and handover events."""

    def __init__(self, base_path: str, enabled: bool = True):
        """Initialize snapshot saver.

        Args:
            base_path: Base directory for snapshots
            enabled: Whether to save snapshots
        """
        self.base_path = base_path
        self.enabled = enabled

        if enabled:
            from pathlib import Path
            Path(base_path).mkdir(parents=True, exist_ok=True)
            # Create subdirectories
            (Path(base_path) / "face_matches").mkdir(exist_ok=True)
            (Path(base_path) / "reid_matches").mkdir(exist_ok=True)
            (Path(base_path) / "handovers").mkdir(exist_ok=True)

    def _get_timestamp(self) -> str:
        """Get formatted timestamp for filenames."""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _draw_label_on_image(self, image: np.ndarray, label: str, position: str = "top") -> np.ndarray:
        """Draw a label on an image.

        Args:
            image: Image to draw on (copied)
            label: Label text
            position: "top" or "bottom"

        Returns:
            Image with label
        """
        result = image.copy()
        h, w = result.shape[:2]

        # Add padding at top/bottom for label
        padding = 40
        if position == "top":
            padded = np.zeros((h + padding, w, 3), dtype=np.uint8)
            padded[padding:, :] = result
            cv2.putText(padded, label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            return padded
        else:
            padded = np.zeros((h + padding, w, 3), dtype=np.uint8)
            padded[:h, :] = result
            cv2.putText(padded, label, (10, h + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            return padded

    def save_match(self, match_event: MatchEvent, frame: np.ndarray) -> Optional[str]:
        """Save a face or Re-ID match snapshot.

        Args:
            match_event: The match event
            frame: Full frame with bounding box drawn

        Returns:
            Path to saved image, or None if disabled
        """
        if not self.enabled:
            return None

        from pathlib import Path

        timestamp = self._get_timestamp()
        match_type = match_event.match_type
        subdir = "face_matches" if match_type == "face" else "reid_matches"

        # Create image with person crop and full frame
        crop = match_event.crop
        crop_h, crop_w = crop.shape[:2]

        # Resize crop to reasonable size
        target_crop_h = 200
        scale = target_crop_h / crop_h if crop_h > 0 else 1
        crop_resized = cv2.resize(crop, (int(crop_w * scale), target_crop_h))

        # Add label to crop
        label = f"{match_event.person_name} ({match_type.upper()}:{match_event.confidence:.2f})"
        crop_labeled = self._draw_label_on_image(crop_resized, label)

        # Draw bbox on frame copy
        frame_copy = frame.copy()
        x1, y1, x2, y2 = map(int, match_event.bbox)
        color = (255, 150, 0) if match_type == "face" else (255, 100, 100)
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame_copy, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Resize frame to fit
        frame_h, frame_w = frame_copy.shape[:2]
        target_frame_h = crop_labeled.shape[0]
        frame_scale = target_frame_h / frame_h
        frame_resized = cv2.resize(frame_copy, (int(frame_w * frame_scale), target_frame_h))

        # Combine side by side
        combined = np.hstack([crop_labeled, frame_resized])

        # Add camera label
        camera_label = f"Camera: {match_event.camera_id} | {timestamp}"
        combined = self._draw_label_on_image(combined, camera_label, "bottom")

        # Save
        safe_name = match_event.person_name.replace(" ", "_").replace("/", "_")
        filename = f"{timestamp}_{safe_name}_{match_type}_{match_event.confidence:.2f}.jpg"
        filepath = Path(self.base_path) / subdir / filename

        cv2.imwrite(str(filepath), combined)
        return str(filepath)

    def save_handover(
        self,
        track_id: str,
        person_name: str,
        from_camera: str,
        to_camera: str,
        from_frame: np.ndarray,
        to_frame: np.ndarray,
        from_bbox: Optional[tuple] = None,
        to_bbox: Optional[tuple] = None,
    ) -> Optional[str]:
        """Save a handover snapshot with both camera frames.

        Args:
            track_id: Global track ID
            person_name: Person name (or "Unknown")
            from_camera: Source camera ID
            to_camera: Destination camera ID
            from_frame: Frame from source camera
            to_frame: Frame from destination camera
            from_bbox: Optional bounding box on source frame
            to_bbox: Optional bounding box on destination frame

        Returns:
            Path to saved image, or None if disabled
        """
        if not self.enabled:
            return None

        from pathlib import Path

        timestamp = self._get_timestamp()

        # Draw bboxes if provided
        from_copy = from_frame.copy()
        to_copy = to_frame.copy()
        color = (0, 255, 255)  # Yellow for handover

        if from_bbox:
            x1, y1, x2, y2 = map(int, from_bbox)
            cv2.rectangle(from_copy, (x1, y1), (x2, y2), color, 3)
            cv2.putText(from_copy, person_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if to_bbox:
            x1, y1, x2, y2 = map(int, to_bbox)
            cv2.rectangle(to_copy, (x1, y1), (x2, y2), color, 3)
            cv2.putText(to_copy, person_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Add camera labels
        from_labeled = self._draw_label_on_image(from_copy, f"FROM: {from_camera}")
        to_labeled = self._draw_label_on_image(to_copy, f"TO: {to_camera}")

        # Resize to same height
        from_h, from_w = from_labeled.shape[:2]
        to_h, to_w = to_labeled.shape[:2]
        target_h = max(from_h, to_h)

        if from_h != target_h:
            scale = target_h / from_h
            from_labeled = cv2.resize(from_labeled, (int(from_w * scale), target_h))
        if to_h != target_h:
            scale = target_h / to_h
            to_labeled = cv2.resize(to_labeled, (int(to_w * scale), target_h))

        # Combine side by side
        combined = np.hstack([from_labeled, to_labeled])

        # Add handover info
        info = f"HANDOVER: {person_name} ({track_id}) | {from_camera} -> {to_camera} | {timestamp}"
        combined = self._draw_label_on_image(combined, info, "bottom")

        # Save
        safe_name = person_name.replace(" ", "_").replace("/", "_")
        filename = f"{timestamp}_handover_{safe_name}_{from_camera}_to_{to_camera}.jpg"
        filepath = Path(self.base_path) / "handovers" / filename

        cv2.imwrite(str(filepath), combined)
        return str(filepath)
