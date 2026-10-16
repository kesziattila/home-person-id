"""ByteTrack implementation for single-camera tracking.

ByteTrack is a simple and effective multi-object tracker that associates
detections using only IoU and motion prediction, without requiring Re-ID features.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from filterpy.kalman import KalmanFilter

from src.config import TrackingConfig
from src.detection.person_detector import Detection, DetectionResult
from src.tracking.track import LocalTrack, TrackState

logger = logging.getLogger(__name__)


def iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of boxes.

    Args:
        boxes_a: (N, 4) array of boxes in xyxy format
        boxes_b: (M, 4) array of boxes in xyxy format

    Returns:
        (N, M) array of IoU values
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))

    # Compute intersection
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0].T)
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1].T)
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2].T)
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3].T)

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Compute union
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, np.newaxis] + area_b - intersection

    return intersection / (union + 1e-6)


def linear_assignment(cost_matrix: np.ndarray) -> tuple[list, list, list]:
    """Solve linear assignment problem using scipy or lap.

    Args:
        cost_matrix: (N, M) cost matrix

    Returns:
        Tuple of (matched_indices, unmatched_a, unmatched_b)
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    try:
        import lap

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        matched_indices = [[ix, x[ix]] for ix in range(len(x)) if x[ix] >= 0]
    except ImportError:
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_indices = list(zip(row_ind, col_ind))

    # Use sets for O(1) lookup instead of O(N) list search
    matched_set_a = {m[0] for m in matched_indices}
    matched_set_b = {m[1] for m in matched_indices}
    unmatched_a = [i for i in range(cost_matrix.shape[0]) if i not in matched_set_a]
    unmatched_b = [i for i in range(cost_matrix.shape[1]) if i not in matched_set_b]

    return matched_indices, unmatched_a, unmatched_b


class KalmanBoxTracker:
    """Kalman filter for tracking a single bounding box.

    State: [x, y, s, r, vx, vy, vs]
    where (x, y) is center, s is scale (area), r is aspect ratio
    """

    count = 0

    def __init__(self, bbox: np.ndarray):
        """Initialize tracker with bounding box.

        Args:
            bbox: [x1, y1, x2, y2] bounding box
        """
        # Initialize Kalman filter
        # State: [x, y, s, r, vx, vy, vs]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])

        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])

        # Measurement noise
        self.kf.R[2:, 2:] *= 10.0

        # Process noise
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0

        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Initialize state
        self.kf.x[:4] = self._bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def _bbox_to_z(self, bbox: np.ndarray) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to [x, y, s, r]."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2
        y = bbox[1] + h / 2
        s = w * h
        r = w / (h + 1e-6)
        return np.array([x, y, s, r]).reshape((4, 1))

    def _z_to_bbox(self, z: np.ndarray) -> np.ndarray:
        """Convert [x, y, s, r] to [x1, y1, x2, y2]."""
        w = np.sqrt(z[2] * z[3])
        h = z[2] / (w + 1e-6)
        x1 = z[0] - w / 2
        y1 = z[1] - h / 2
        x2 = z[0] + w / 2
        y2 = z[1] + h / 2
        return np.array([x1, y1, x2, y2]).flatten()

    def update(self, bbox: np.ndarray):
        """Update state with observed bbox."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._bbox_to_z(bbox))

    def predict(self) -> np.ndarray:
        """Advance state and return predicted bbox."""
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0

        self.time_since_update += 1
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """Get current state as bbox."""
        return self._z_to_bbox(self.kf.x[:4].flatten())


@dataclass
class TrackingResult:
    """Result of tracking update."""

    tracks: list[LocalTrack]
    new_track_ids: list[int]
    lost_track_ids: list[int]
    removed_track_ids: list[int]


class ByteTracker:
    """ByteTrack multi-object tracker for a single camera."""

    def __init__(self, camera_id: str, config: TrackingConfig):
        """Initialize ByteTracker.

        Args:
            camera_id: Camera identifier
            config: Tracking configuration
        """
        self.camera_id = camera_id
        self.config = config

        self._trackers: list[KalmanBoxTracker] = []
        self._tracks: dict[int, LocalTrack] = {}
        self._frame_count = 0

        # Track ID counter
        self._next_id = 1

    def update(
        self,
        detections: DetectionResult,
        frame: Optional[np.ndarray] = None,
        reid_embeddings: Optional[dict[int, tuple[np.ndarray, float]]] = None,
    ) -> TrackingResult:
        """Update tracker with new detections.

        Args:
            detections: Person detections from current frame
            frame: Optional frame for extracting crops

        Returns:
            TrackingResult with current tracks
        """
        self._frame_count += 1

        # Get detections as numpy array
        dets = detections.to_numpy()  # (N, 5) with [x1, y1, x2, y2, conf]

        new_track_ids = []
        lost_track_ids = []
        removed_track_ids = []

        # Predict new locations of existing trackers (pre-allocate array)
        if self._trackers:
            predicted_boxes = np.empty((len(self._trackers), 4), dtype=np.float64)
            for i, tracker in enumerate(self._trackers):
                predicted_boxes[i] = tracker.predict()
        else:
            predicted_boxes = np.empty((0, 4), dtype=np.float64)

        # Split detections into high and low confidence (indices into dets)
        high_idx = np.flatnonzero(dets[:, 4] >= self.config.track_thresh)
        low_idx = np.flatnonzero(dets[:, 4] < self.config.track_thresh)

        # First association: high confidence detections with all trackers
        matched, unmatched_dets, unmatched_trks = self._associate(
            dets[high_idx, :4], predicted_boxes
        )
        for det_idx, trk_idx in matched:
            tracker = self._trackers[trk_idx]
            was_confirmed = self._tracks[tracker.id].is_confirmed
            track = self._apply_detection(tracker, dets, high_idx[det_idx], frame, reid_embeddings)
            # Report as new when track becomes confirmed (not on first detection)
            if not was_confirmed and track.is_confirmed:
                new_track_ids.append(track.track_id)

        # Second association: low confidence detections keep existing confirmed
        # tracks alive (occlusion, sitting, motion blur). Unconfirmed tracks are
        # excluded so low-confidence noise can never confirm a spurious track.
        remaining = [
            i for i in unmatched_trks
            if (t := self._tracks.get(self._trackers[i].id)) is not None and t.is_confirmed
        ]
        matched_low, _, _ = self._associate(
            dets[low_idx, :4], predicted_boxes[remaining]
        )
        recovered = set()
        for det_idx, trk_idx in matched_low:
            self._apply_detection(
                self._trackers[remaining[trk_idx]], dets, low_idx[det_idx], frame, reid_embeddings
            )
            recovered.add(remaining[trk_idx])
        unmatched_trks = [i for i in unmatched_trks if i not in recovered]

        # Create new trackers for unmatched high confidence detections.
        # new_track_ids is NOT populated here - tracks are reported as "new"
        # only when they become confirmed to avoid spurious detections.
        for det_idx in unmatched_dets:
            det = dets[high_idx[det_idx]]
            tracker = KalmanBoxTracker(det[:4])
            self._trackers.append(tracker)
            self._tracks[tracker.id] = LocalTrack(
                track_id=self._next_id,
                camera_id=self.camera_id,
                bbox=tuple(det[:4]),
                confidence=det[4],
                state=TrackState.NEW,
            )
            self._next_id += 1
            self._store_observation(
                self._tracks[tracker.id], high_idx[det_idx], frame, reid_embeddings
            )

        # Mark unmatched trackers as lost
        for i in unmatched_trks:
            tracker = self._trackers[i]
            track = self._tracks.get(tracker.id)
            if track:
                track.time_since_update += 1
                if track.time_since_update > self.config.track_buffer:
                    track.state = TrackState.REMOVED
                    removed_track_ids.append(track.track_id)
                elif track.state == TrackState.TRACKED:
                    track.state = TrackState.LOST
                    lost_track_ids.append(track.track_id)

        # Remove dead trackers
        self._trackers = [t for t in self._trackers if self._tracks.get(t.id) and self._tracks[t.id].state != TrackState.REMOVED]

        # Clean up removed tracks
        for track_id in removed_track_ids:
            for tid, track in list(self._tracks.items()):
                if track.track_id == track_id:
                    del self._tracks[tid]
                    break

        # Get active tracks
        active_tracks = [
            track for track in self._tracks.values()
            if track.state in (TrackState.TRACKED, TrackState.NEW) and track.is_confirmed
        ]

        return TrackingResult(
            tracks=active_tracks,
            new_track_ids=new_track_ids,
            lost_track_ids=lost_track_ids,
            removed_track_ids=removed_track_ids,
        )

    def _associate(
        self, det_boxes: np.ndarray, track_boxes: np.ndarray
    ) -> tuple[list, list, list]:
        """Match detections to track boxes by IoU, rejecting pairs below match_thresh.

        Returns:
            (matched [(det_idx, trk_idx)], unmatched det indices, unmatched track indices)
        """
        iou_matrix = iou_batch(det_boxes, track_boxes)
        matched, unmatched_dets, unmatched_trks = linear_assignment(1 - iou_matrix)

        accepted = []
        for det_idx, trk_idx in matched:
            if iou_matrix[det_idx, trk_idx] >= self.config.match_thresh:
                accepted.append((det_idx, trk_idx))
            else:
                unmatched_dets.append(det_idx)
                unmatched_trks.append(trk_idx)
        return accepted, unmatched_dets, unmatched_trks

    def _apply_detection(
        self,
        tracker: "KalmanBoxTracker",
        dets: np.ndarray,
        det_idx: int,
        frame: Optional[np.ndarray],
        reid_embeddings: Optional[dict[int, tuple[np.ndarray, float]]],
    ) -> LocalTrack:
        """Update a matched tracker and its track with detection `dets[det_idx]`."""
        det = dets[det_idx]
        tracker.update(det[:4])
        track = self._tracks[tracker.id]
        track.bbox = tuple(det[:4])
        track.confidence = det[4]
        track.hits += 1
        track.time_since_update = 0
        track.state = TrackState.TRACKED
        self._store_observation(track, det_idx, frame, reid_embeddings)
        return track

    def _store_observation(
        self,
        track: LocalTrack,
        det_idx: int,
        frame: Optional[np.ndarray],
        reid_embeddings: Optional[dict[int, tuple[np.ndarray, float]]],
    ) -> None:
        """Attach the pre-computed Re-ID embedding and crop for a detection."""
        if reid_embeddings and det_idx in reid_embeddings:
            track.last_reid_embedding, track.last_reid_quality = reid_embeddings[det_idx]
        if frame is not None:
            track.last_crop = self._extract_crop(frame, track.bbox)

    def _extract_crop(
        self, frame: np.ndarray, bbox: tuple[float, float, float, float]
    ) -> np.ndarray:
        """Extract person crop from frame.

        Args:
            frame: Full frame
            bbox: Bounding box in xyxy format

        Returns:
            Cropped person image
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]

        # Clip to frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        return frame[y1:y2, x1:x2].copy()

    def get_track(self, track_id: int) -> Optional[LocalTrack]:
        """Get track by ID."""
        for track in self._tracks.values():
            if track.track_id == track_id:
                return track
        return None

    def get_all_tracks(self) -> list[LocalTrack]:
        """Get all active tracks."""
        return [t for t in self._tracks.values() if t.state != TrackState.REMOVED]

    def reset(self):
        """Reset tracker state."""
        self._trackers.clear()
        self._tracks.clear()
        self._frame_count = 0
        KalmanBoxTracker.count = 0
        logger.info(f"ByteTracker reset for camera {self.camera_id}")


class ByteTrackerManager:
    """Manages ByteTracker instances for multiple cameras."""

    def __init__(self, config: TrackingConfig):
        """Initialize tracker manager.

        Args:
            config: Tracking configuration
        """
        self.config = config
        self._trackers: dict[str, ByteTracker] = {}

    def get_tracker(self, camera_id: str) -> ByteTracker:
        """Get or create tracker for a camera.

        Args:
            camera_id: Camera identifier

        Returns:
            ByteTracker for the camera
        """
        if camera_id not in self._trackers:
            self._trackers[camera_id] = ByteTracker(camera_id, self.config)
            logger.debug(f"Created ByteTracker for camera {camera_id}")

        return self._trackers[camera_id]

    def update(
        self,
        camera_id: str,
        detections: DetectionResult,
        frame: Optional[np.ndarray] = None,
        reid_embeddings: Optional[dict[int, tuple[np.ndarray, float]]] = None,
    ) -> TrackingResult:
        """Update tracker for a camera.

        Args:
            camera_id: Camera identifier
            detections: Person detections
            frame: Optional frame for crops
            reid_embeddings: Optional pre-computed Re-ID embeddings

        Returns:
            TrackingResult with current tracks
        """
        tracker = self.get_tracker(camera_id)
        return tracker.update(detections, frame, reid_embeddings)

    def reset(self, camera_id: Optional[str] = None):
        """Reset tracker(s).

        Args:
            camera_id: Specific camera to reset, or None for all
        """
        if camera_id is not None:
            if camera_id in self._trackers:
                self._trackers[camera_id].reset()
        else:
            for tracker in self._trackers.values():
                tracker.reset()
