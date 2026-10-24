"""ByteTrack implementation for single-camera tracking.

ByteTrack is a simple and effective multi-object tracker that associates
detections using only IoU and motion prediction, without requiring Re-ID features.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from filterpy.kalman import KalmanFilter

from src.config import TrackingConfig
from src.detection.person_detector import Detection, DetectionResult
from src.tracking.attribution import APPEARANCE_EMA_ALPHA, overlap_of_smaller, update_appearance
from src.tracking.track import LocalTrack, TrackState
from src.utils.embedding_utils import l2_normalize
from src.utils.image_utils import crop_bbox

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
    intersection = _intersection_batch(boxes_a, boxes_b)
    union = _area(boxes_a)[:, np.newaxis] + _area(boxes_b) - intersection
    return intersection / (union + 1e-6)


def ioa_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Fraction of each box in `boxes_a` covered by each box in `boxes_b` (N, M).

    1.0 means box a lies entirely inside box b (e.g. a partial-body duplicate
    detection inside an already tracked person's box).
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    return _intersection_batch(boxes_a, boxes_b) / (_area(boxes_a)[:, np.newaxis] + 1e-6)


def _area(boxes: np.ndarray) -> np.ndarray:
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def _intersection_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0].T)
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1].T)
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2].T)
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3].T)
    return np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)


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


def _center(box: np.ndarray) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _size(box: np.ndarray) -> tuple[float, float]:
    return box[2] - box[0], box[3] - box[1]


def _direction(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unit (dy, dx) of the centre motion from box a to box b."""
    (ax, ay), (bx, by) = _center(a), _center(b)
    d = np.array([by - ay, bx - ax])
    return d / (np.linalg.norm(d) + 1e-6)


def direction_cost(det_boxes: np.ndarray, det_scores: np.ndarray, trackers: list) -> np.ndarray:
    """OC-SORT OCM (N dets, M tracks): agreement between each track's motion direction and the
    direction from its observation delta_t frames back to each detection, in [-0.5, 0.5] * score
    (0 for a track without a direction yet)."""
    if len(det_boxes) == 0 or not trackers:
        return np.zeros((len(det_boxes), len(trackers)))
    prev = np.array([t.previous_observation() for t in trackers])                  # (M, 4)
    dcx = ((det_boxes[:, 0] + det_boxes[:, 2]) / 2)[np.newaxis, :] - ((prev[:, 0] + prev[:, 2]) / 2)[:, np.newaxis]
    dcy = ((det_boxes[:, 1] + det_boxes[:, 3]) / 2)[np.newaxis, :] - ((prev[:, 1] + prev[:, 3]) / 2)[:, np.newaxis]
    norm = np.sqrt(dcx ** 2 + dcy ** 2) + 1e-6
    vel = np.array([t.velocity if t.velocity is not None else (0.0, 0.0) for t in trackers])   # (M, 2) dy, dx
    cos = np.clip(vel[:, 1:2] * dcx / norm + vel[:, 0:1] * dcy / norm, -1, 1)
    cost = (np.pi / 2.0 - np.abs(np.arccos(cos))) / np.pi                            # (M, N)
    cost[[t.velocity is None for t in trackers]] = 0.0
    return (cost * det_scores[np.newaxis, :]).T


class KalmanBoxTracker:
    """Kalman filter for tracking a single bounding box.

    State: [x, y, s, r, vx, vy, vs]
    where (x, y) is center, s is scale (area), r is aspect ratio
    """

    count = 0

    def __init__(self, bbox: np.ndarray, delta_t: int = 3, reupdate: bool = False):
        """Initialize tracker with bounding box.

        Args:
            bbox: [x1, y1, x2, y2] bounding box
            delta_t: frames back for the motion direction (OC-SORT OCM)
            reupdate: re-run the filter along a virtual straight path after missed frames (OC-SORT ORU)
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
        # OC-SORT (Cao et al. 2023, MIT reference noahcao/OC_SORT): observed boxes by age, the
        # direction of motion between observations delta_t frames apart, the filter state frozen
        # at the first missed frame
        self.delta_t = delta_t
        self.reupdate = reupdate
        self.last_observation = np.asarray(bbox, dtype=np.float64)[:4]
        self.observations: dict[int, np.ndarray] = {0: self.last_observation}
        self.velocity: Optional[np.ndarray] = None    # unit (dy, dx)
        self._frozen: Optional[tuple[np.ndarray, np.ndarray]] = None

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
        bbox = np.asarray(bbox, dtype=np.float64)[:4]
        previous = next((self.observations[self.age - dt] for dt in range(self.delta_t, 0, -1)
                         if self.age - dt in self.observations), self.last_observation)
        self.velocity = _direction(previous, bbox)
        if self.reupdate and self._frozen is not None and self.time_since_update > 1:
            self._reupdate(bbox)
        else:
            self.kf.update(self._bbox_to_z(bbox))
        self._frozen = None
        self.last_observation = bbox
        self.observations[self.age] = bbox
        for age in [a for a in self.observations if a < self.age - self.delta_t]:
            del self.observations[age]
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def _reupdate(self, bbox: np.ndarray) -> None:
        """OC-SORT ORU: the filter coasted on its own prediction while the box was missing (its
        velocity drifts); restart from the state at the first missed frame and update it along a
        straight path from the last observed box to this one, one step per frame."""
        self.kf.x, self.kf.P = self._frozen
        gap = self.time_since_update
        (x1, y1), (w1, h1) = _center(self.last_observation), _size(self.last_observation)
        (x2, y2), (w2, h2) = _center(bbox), _size(bbox)
        for i in range(1, gap + 1):
            f = i / gap
            x, y, w, h = x1 + f * (x2 - x1), y1 + f * (y2 - y1), w1 + f * (w2 - w1), h1 + f * (h2 - h1)
            self.kf.update(np.array([x, y, w * h, w / (h + 1e-6)]).reshape((4, 1)))
            if i < gap:
                self.kf.predict()

    def previous_observation(self) -> np.ndarray:
        """The observed box delta_t frames back (or the closest later one) for the direction cost."""
        for dt in range(self.delta_t, 0, -1):
            if self.age - dt in self.observations:
                return self.observations[self.age - dt]
        return self.last_observation

    def predict(self) -> np.ndarray:
        """Advance state and return predicted bbox."""
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        if self.reupdate and self.time_since_update == 1:
            # the previous frame brought no box: keep the state one step past the last observation
            self._frozen = (self.kf.x.copy(), self.kf.P.copy())

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
class AssociationEvent:
    """Diagnostic record of a track birth, revival or suppressed duplicate (metrics only).

    `neighbors` describes the closest other tracks at that moment so fragmentation
    can be explained offline: a birth next to a LOST track with IoU just below
    match_thresh means association failed; a revival inside a live track's box
    means a duplicate detection.
    """

    kind: str  # "birth" | "revive" | "dup_skip" | "unswap"
    camera_id: str
    track_id: Optional[int]  # local track id (dup_skip: the track that would have revived)
    stage: Optional[int]  # association stage that produced it (1 = high conf, 2 = low conf)
    bbox: tuple[float, float, float, float]  # normalized 0-1
    confidence: float
    neighbors: list[dict] = field(default_factory=list)


@dataclass
class _OverlapEpisode:
    """Two tracks that overlapped: their clean appearance before, and what each looked like after."""

    pre: dict[int, np.ndarray]                                  # tracker id -> clean appearance at overlap start
    post: dict[int, np.ndarray] = field(default_factory=dict)   # tracker id -> EMA of isolated samples since
    samples: dict[int, int] = field(default_factory=dict)


# A partner counts as merged into the other box only right after it lost its own: replay
# 10-23..10-24, a stale lost box (up to track_buffer frames) left 9.9 % of isolated track-frames
# uncertain; 3 frames: 2.9 %, the 08:29 merge still flagged
MERGE_FRAMES = 3


def _ema(prev: Optional[np.ndarray], sample: np.ndarray) -> np.ndarray:
    """Unit-length Re-ID EMA (same rate as the appearance EMA)."""
    sample = l2_normalize(sample)
    if prev is None:
        return sample
    return l2_normalize((1 - APPEARANCE_EMA_ALPHA) * prev + APPEARANCE_EMA_ALPHA * sample)


@dataclass
class TrackingResult:
    """Result of tracking update."""

    tracks: list[LocalTrack]
    new_track_ids: list[int]
    lost_track_ids: list[int]
    removed_track_ids: list[int]
    association_events: list[AssociationEvent] = field(default_factory=list)


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
        # (tracker id, tracker id) -> overlap episode awaiting the un-swap check
        self._episodes: dict[tuple[int, int], _OverlapEpisode] = {}
        # tracker id -> isolated Re-ID samples an uncertain track gave since its last merge
        self._settle: dict[int, int] = {}

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
        keypoints = [d.keypoints for d in detections.detections]

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

        # First association: high confidence detections with all trackers (IoU, plus the
        # motion-direction agreement when velocity_weight > 0, OC-SORT OCM)
        extra = None
        if self.config.velocity_weight > 0:
            extra = self.config.velocity_weight * direction_cost(dets[high_idx, :4], dets[high_idx, 4], self._trackers)
        matched, unmatched_dets, unmatched_trks = self._associate(
            dets[high_idx, :4], predicted_boxes, extra
        )
        frame_shape = detections.frame_shape
        events: list[AssociationEvent] = []

        def take_high(det_idx: int, trk_idx: int, stage: int) -> None:
            tracker = self._trackers[trk_idx]
            prev = self._tracks[tracker.id]
            was_confirmed, was_lost = prev.is_confirmed, prev.state == TrackState.LOST
            track = self._apply_detection(tracker, dets, high_idx[det_idx], frame, reid_embeddings, keypoints)
            # Report as new when track becomes confirmed (not on first detection)
            if not was_confirmed and track.is_confirmed:
                new_track_ids.append(track.track_id)
            if was_lost:
                events.append(self._event("revive", track.track_id, stage, dets[high_idx[det_idx]],
                                          frame_shape, predicted_boxes, exclude=trk_idx))

        for det_idx, trk_idx in matched:
            take_high(det_idx, trk_idx, 1)

        # Second association: low confidence detections keep existing confirmed
        # tracks alive (occlusion, sitting, motion blur). Unconfirmed tracks are
        # excluded so low-confidence noise can never confirm a spurious track.
        remaining = [
            i for i in unmatched_trks
            if (t := self._tracks.get(self._trackers[i].id)) is not None and t.is_confirmed
        ]
        # A low-confidence box lying mostly inside a box matched in stage 1 is a
        # partial/duplicate detection of that live person: letting it revive a
        # stale track would give one person two tracks.
        live_boxes = dets[[high_idx[d] for d, _ in matched], :4]
        is_dup = (ioa_batch(dets[low_idx, :4], live_boxes) >= self.config.duplicate_ioa).any(axis=1)
        if is_dup.any() and remaining:
            dup_ious = iou_batch(dets[low_idx[is_dup], :4], predicted_boxes[remaining])
            for row, det_i in zip(dup_ious, low_idx[is_dup]):
                if row.max() >= self.config.match_thresh:
                    blocked = self._tracks[self._trackers[remaining[int(row.argmax())]].id]
                    events.append(self._event("dup_skip", blocked.track_id, 2, dets[det_i],
                                              frame_shape, predicted_boxes))
        low_idx = low_idx[~is_dup]

        matched_low, _, _ = self._associate(
            dets[low_idx, :4], predicted_boxes[remaining]
        )
        recovered = set()
        for det_idx, trk_idx in matched_low:
            tracker_idx = remaining[trk_idx]
            was_lost = self._tracks[self._trackers[tracker_idx].id].state == TrackState.LOST
            track = self._apply_detection(
                self._trackers[tracker_idx], dets, low_idx[det_idx], frame, reid_embeddings, keypoints
            )
            if was_lost:
                events.append(self._event("revive", track.track_id, 2, dets[low_idx[det_idx]],
                                          frame_shape, predicted_boxes, exclude=tracker_idx))
            recovered.add(tracker_idx)
        unmatched_trks = [i for i in unmatched_trks if i not in recovered]

        # Third association (OC-SORT OCR): a high-confidence box still unmatched may belong to a
        # track whose Kalman prediction drifted while its person was hidden: match it against
        # the track's LAST OBSERVED box instead of the prediction
        if self.config.recover_last_observation and unmatched_dets and unmatched_trks:
            last_boxes = np.array([self._trackers[i].last_observation for i in unmatched_trks])
            rematched, _, _ = self._associate(dets[high_idx[unmatched_dets], :4], last_boxes)
            for d, t in rematched:
                take_high(unmatched_dets[d], unmatched_trks[t], 3)
            taken_dets = {unmatched_dets[d] for d, _ in rematched}
            taken_trks = {unmatched_trks[t] for _, t in rematched}
            unmatched_dets = [d for d in unmatched_dets if d not in taken_dets]
            unmatched_trks = [t for t in unmatched_trks if t not in taken_trks]

        # Create new trackers for unmatched high confidence detections.
        # new_track_ids is NOT populated here - tracks are reported as "new"
        # only when they become confirmed to avoid spurious detections.
        for det_idx in unmatched_dets:
            det = dets[high_idx[det_idx]]
            events.append(self._event("birth", self._next_id, 1, det, frame_shape, predicted_boxes))
            tracker = KalmanBoxTracker(det[:4], self.config.velocity_delta_t, self.config.reupdate_after_occlusion)
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
                self._tracks[tracker.id], high_idx[det_idx], frame, reid_embeddings, keypoints
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

        self._check_swaps(events, frame_shape)

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
            association_events=events,
        )

    def _event(
        self,
        kind: str,
        track_id: Optional[int],
        stage: int,
        det: np.ndarray,
        frame_shape: tuple,
        predicted_boxes: np.ndarray,
        exclude: Optional[int] = None,
        max_neighbors: int = 3,
    ) -> AssociationEvent:
        """Build an AssociationEvent with the nearest existing tracks (by predicted box)."""
        h, w = frame_shape[:2]
        box = det[np.newaxis, :4]
        neighbors = []
        if len(predicted_boxes):
            ious = iou_batch(box, predicted_boxes)[0]
            ioas = ioa_batch(box, predicted_boxes)[0]
            diag = float(np.hypot(det[2] - det[0], det[3] - det[1])) + 1e-6
            centers = (predicted_boxes[:, :2] + predicted_boxes[:, 2:4]) / 2
            dists = np.hypot(*(centers - (det[:2] + det[2:4]) / 2).T) / diag
            for i in np.lexsort((dists, -ious))[:max_neighbors + 1]:
                track = self._tracks.get(self._trackers[i].id)
                if i == exclude or track is None or (ious[i] == 0 and dists[i] > 2):
                    continue
                neighbors.append({
                    "track_id": track.track_id,
                    "state": track.state.name,
                    "confirmed": track.is_confirmed,
                    "time_since_update": track.time_since_update,
                    "iou": round(float(ious[i]), 3),
                    "ioa": round(float(ioas[i]), 3),
                    "center_dist": round(float(dists[i]), 3),
                })
        return AssociationEvent(
            kind=kind,
            camera_id=self.camera_id,
            track_id=track_id,
            stage=stage,
            bbox=(round(det[0] / w, 4), round(det[1] / h, 4), round(det[2] / w, 4), round(det[3] / h, 4)),
            confidence=round(float(det[4]), 3),
            neighbors=neighbors[:max_neighbors],
        )

    def _check_swaps(self, events: list[AssociationEvent], frame_shape: tuple) -> None:
        """Swap two tracks' ids back when, after overlapping, each looks like the other did before;
        flag tracks whose identity is open (`uncertain`) (~0.1 ms).

        IoU carries no identity while one person walks in front of another: live 2026-10-24
        08:29:00 (ground_floor) Virag passed in front of Ati lying on the sofa, one box covered
        both for a frame, and ByteTrack re-associated crosswise when they split (replayed
        exactly); Ati's face-named track left with Virag. In-frame gates (Re-ID cosine, pose
        OKS, 2026-10-23) failed: during the overlap both dets' Re-ID is a mix of the two people.
        Separated again, it is clear: each track's clean appearance (EMA of samples while its
        box overlaps no other live box > unswap_overlap) is frozen when a pair overlaps; after
        it, each collects unswap_samples isolated samples; crosswise mean cosine beating
        straight by > unswap_margin swaps the ids (and with them the global tracks and names).
        An episode is dropped when either track misses a frame (a revived track's "before" is
        not trustworthy: keeping them produced a false swap). Replay 10-23..10-24 (all
        cameras): exactly the two known swaps exceed 0.15 (0.27, 0.20); the next pair 0.08.

        `uncertain`: the track overlaps (> unswap_overlap) a confirmed track that lost its box
        in the last MERGE_FRAMES frames, i.e. the two may have been merged into one box and ByteTrack may hand them
        over crosswise; it stays uncertain until it gives unswap_samples isolated Re-ID samples.
        A survivor whose partner never comes back is uncertain the same way: deciding whose it
        is from Re-ID alone was right 3 / wrong 2 times on 10-23 (TUNING_LOG), so it only waits.
        """
        cfg = self.config
        if cfg.unswap_margin <= 0:
            return
        tracks = {k: t for k, t in self._tracks.items() if t.is_confirmed}
        live = {k: t for k, t in tracks.items() if t.time_since_update == 0 and t.state == TrackState.TRACKED}
        self._episodes = {key: ep for key, ep in self._episodes.items() if key[0] in live and key[1] in live}
        overlap = dict.fromkeys(live, 0.0)
        keys = list(tracks)
        for i, ka in enumerate(keys):
            for kb in keys[i + 1:]:
                a, b = tracks[ka], tracks[kb]
                o = overlap_of_smaller(a.bbox, b.bbox)
                if o <= cfg.unswap_overlap:
                    continue
                if ka not in live or kb not in live:
                    # just merged into one box (or hidden): who is who is open until they separate
                    if max(a.time_since_update, b.time_since_update) <= MERGE_FRAMES:
                        for k in (ka, kb):
                            tracks[k].uncertain, self._settle[k] = True, 0
                    continue
                overlap[ka], overlap[kb] = max(overlap[ka], o), max(overlap[kb], o)
                pair = (ka, kb) if ka < kb else (kb, ka)
                episode = self._episodes.get(pair)
                if episode is not None:
                    episode.post.clear()          # still overlapping: "after" starts later
                    episode.samples.clear()
                elif a.clean_appearance is not None and b.clean_appearance is not None:
                    self._episodes[pair] = _OverlapEpisode(pre={ka: a.clean_appearance, kb: b.clean_appearance})
        for k, track in live.items():
            if (not track.last_reid_fresh or overlap[k] > cfg.unswap_overlap
                    or not np.any(track.last_reid_embedding)):      # an all-zero Re-ID carries nothing
                continue
            track.clean_appearance = _ema(track.clean_appearance, track.last_reid_embedding)
            if track.uncertain:
                self._settle[k] = self._settle.get(k, 0) + 1
                if self._settle[k] >= cfg.unswap_samples:
                    track.uncertain = False
                    del self._settle[k]
            for pair, episode in list(self._episodes.items()):
                if k not in pair:
                    continue
                episode.post[k] = _ema(episode.post.get(k), track.last_reid_embedding)
                episode.samples[k] = episode.samples.get(k, 0) + 1
                a, b = pair
                if min(episode.samples.get(a, 0), episode.samples.get(b, 0)) < cfg.unswap_samples:
                    continue
                del self._episodes[pair]
                pre, post = episode.pre, episode.post
                straight = float(post[a] @ pre[a] + post[b] @ pre[b]) / 2
                cross = float(post[a] @ pre[b] + post[b] @ pre[a]) / 2
                if cross - straight > cfg.unswap_margin:
                    self._swap_ids(self._tracks[a], self._tracks[b], straight, cross, events, frame_shape)
        self._settle = {k: n for k, n in self._settle.items() if k in self._tracks}

    def _swap_ids(self, a: LocalTrack, b: LocalTrack, straight: float, cross: float,
                  events: list[AssociationEvent], frame_shape: tuple) -> None:
        """Give each box the other's id (identity follows the person; Kalman state stays with the box)."""
        a.track_id, b.track_id = b.track_id, a.track_id
        a.created_at, b.created_at = b.created_at, a.created_at
        now = time.time()
        for track in (a, b):
            # the appearance EMA absorbed the overlap: restart it from the clean one, unsettled
            track.appearance, track.appearance_changed_at = track.clean_appearance, now
        logger.info(f"[{self.camera_id}] Un-swapped tracks {a.track_id} <-> {b.track_id} after overlap "
                    f"(straight {straight:.2f}, crosswise {cross:.2f})")
        for track, other in ((a, b), (b, a)):
            events.append(self._track_event("unswap", track, frame_shape,
                                            {"track_id": other.track_id, "straight": round(straight, 3),
                                             "cross": round(cross, 3)}))

    def _track_event(self, kind: str, track: LocalTrack, frame_shape: tuple, neighbor: dict) -> AssociationEvent:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = track.bbox
        return AssociationEvent(
            kind=kind, camera_id=self.camera_id, track_id=track.track_id, stage=None,
            bbox=(round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)),
            confidence=round(float(track.confidence), 3), neighbors=[neighbor])

    def _associate(
        self, det_boxes: np.ndarray, track_boxes: np.ndarray, extra: Optional[np.ndarray] = None
    ) -> tuple[list, list, list]:
        """Match detections to track boxes by IoU (+ `extra` score, e.g. the OC-SORT direction
        agreement), rejecting pairs whose IoU is below match_thresh.

        Returns:
            (matched [(det_idx, trk_idx)], unmatched det indices, unmatched track indices)
        """
        iou_matrix = iou_batch(det_boxes, track_boxes)
        above = iou_matrix >= self.config.match_thresh
        if extra is not None and above.size and above.sum(1).max() <= 1 and above.sum(0).max() <= 1:
            # every det/track has at most one candidate: nothing to weigh (as OC-SORT)
            pairs = [tuple(p) for p in np.argwhere(above)]
            matched = [list(p) for p in pairs]
            unmatched_dets = [i for i in range(len(det_boxes)) if i not in {p[0] for p in pairs}]
            unmatched_trks = [i for i in range(len(track_boxes)) if i not in {p[1] for p in pairs}]
        else:
            score = iou_matrix if extra is None else iou_matrix + extra
            matched, unmatched_dets, unmatched_trks = linear_assignment(1 - score)

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
        keypoints: Optional[list] = None,
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
        self._store_observation(track, det_idx, frame, reid_embeddings, keypoints)
        return track

    def _store_observation(
        self,
        track: LocalTrack,
        det_idx: int,
        frame: Optional[np.ndarray],
        reid_embeddings: Optional[dict[int, tuple[np.ndarray, float]]],
        keypoints: Optional[list] = None,
    ) -> None:
        """Attach the pre-computed Re-ID embedding, the pose keypoints and crop for a detection."""
        track.last_keypoints = keypoints[det_idx] if keypoints else None
        track.last_reid_fresh = bool(reid_embeddings) and det_idx in reid_embeddings
        if track.last_reid_fresh:
            track.last_reid_embedding, track.last_reid_quality = reid_embeddings[det_idx]
            update_appearance(track, track.last_reid_embedding, time.time())
        if frame is not None:
            track.last_crop = self._extract_crop(frame, track.bbox)

    def _extract_crop(
        self, frame: np.ndarray, bbox: tuple[float, float, float, float]
    ) -> np.ndarray:
        """Extract a person crop (copied, so it outlives the frame buffer)."""
        return crop_bbox(frame, bbox).copy()

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
        self._episodes.clear()
        self._settle.clear()
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
