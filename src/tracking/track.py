"""Track data classes for person tracking."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np


class TrackState(Enum):
    """State of a track."""

    NEW = "new"  # Just created, not confirmed
    TRACKED = "tracked"  # Actively being tracked
    GRACE = "grace"  # Recently lost; zone still treated as non-estimated during grace period
    LOST = "lost"  # Lost and grace period expired; zone is now estimated
    REMOVED = "removed"  # Marked for removal


@dataclass
class LocalTrack:
    """Track within a single camera.

    This is managed by ByteTrack for each camera independently.
    """

    track_id: int  # Local track ID (per camera)
    camera_id: str
    state: TrackState = TrackState.NEW

    # Bounding box in xyxy format
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    confidence: float = 0.0

    # Kalman filter state
    mean: Optional[np.ndarray] = None  # State mean
    covariance: Optional[np.ndarray] = None  # State covariance

    # Tracking stats
    hits: int = 0  # Number of consecutive hits
    age: int = 0  # Total frames since track started
    time_since_update: int = 0  # Frames since last detection match

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Crop for Re-ID (stored temporarily)
    last_crop: Optional[np.ndarray] = None
    last_reid_embedding: Optional[np.ndarray] = None
    last_reid_quality: float = 0.0
    # The embedding came with the last matched detection (a low-confidence one gets none,
    # person_detector.detect_for_tracking): only then does it describe last_crop
    last_reid_fresh: bool = False
    # Pose detector keypoints of the last matched detection ((17, 3) frame px, Detection.keypoints)
    last_keypoints: Optional[np.ndarray] = None
    # Re-ID appearance EMA and when it last changed (attribution.update_appearance)
    appearance: Optional[np.ndarray] = None
    appearance_changed_at: float = 0.0
    # Re-ID EMA of samples taken while the box overlapped no other live box (ByteTracker un-swap)
    clean_appearance: Optional[np.ndarray] = None
    # Who this box is is open: it overlaps a track that lost its box (merged / hidden), so ByteTrack
    # may hand them over crosswise; cleared after isolated Re-ID samples (ByteTracker._check_swaps)
    uncertain: bool = False

    @property
    def is_confirmed(self) -> bool:
        """Check if track is confirmed (enough consecutive hits)."""
        return self.hits >= 3

    @property
    def center(self) -> tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def width(self) -> float:
        """Get width of bounding box."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        """Get height of bounding box."""
        return self.bbox[3] - self.bbox[1]

    def to_tlwh(self) -> np.ndarray:
        """Convert bbox to [x, y, w, h] format (top-left)."""
        x1, y1, x2, y2 = self.bbox
        return np.array([x1, y1, x2 - x1, y2 - y1])

    def to_xyxy(self) -> np.ndarray:
        """Convert bbox to [x1, y1, x2, y2] format."""
        return np.array(self.bbox)


@dataclass
class GlobalTrack:
    """Global track that spans multiple cameras.

    This is the cross-camera identity that links local tracks together. Who the
    track is (person, method, confidence) is owned by IdentityLinker's
    TrackIdentityState only; read it via GlobalTrackManager._person_id().
    """

    track_id: str  # Global track ID (e.g., "global_42")

    # Tracking state
    state: TrackState = TrackState.NEW
    cameras_seen: list[str] = field(default_factory=list)
    current_camera_id: Optional[str] = None
    current_local_track_id: Optional[int] = None

    # Timestamps
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)

    # Metadata
    metadata: dict = field(default_factory=dict)

    def update_location(self, camera_id: str, local_track_id: int):
        """Update current location of the track.

        Args:
            camera_id: Current camera
            local_track_id: Local track ID on that camera
        """
        self.current_camera_id = camera_id
        self.current_local_track_id = local_track_id
        self.last_seen = datetime.now()

        if camera_id not in self.cameras_seen:
            self.cameras_seen.append(camera_id)
