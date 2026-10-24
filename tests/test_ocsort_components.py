"""OC-SORT components ported into ByteTracker (Cao et al. 2023; MIT reference noahcao/OC_SORT).

OCM (velocity_weight): the first association also weighs how well the direction from a track's
observation delta_t frames back to a detection agrees with the track's own motion direction.
OCR (recover_last_observation): a box left unmatched is tried against each unmatched track's
last OBSERVED box (the Kalman prediction drifts while the person is hidden).
ORU (reupdate_after_occlusion): after missed frames the filter is re-run along a straight path
from the last observed box to the new one, so its velocity is not the drifted one.
"""

from dataclasses import replace

import numpy as np

from src.config import TrackingConfig
from src.detection.person_detector import Detection, DetectionResult
from src.tracking.byte_tracker import ByteTracker, KalmanBoxTracker, direction_cost

SHAPE = (1080, 1920, 3)
BASE = TrackingConfig(track_thresh=0.5, track_buffer=60, match_thresh=0.3, stationary_timeout=0,
                      velocity_weight=0.0, recover_last_observation=False, reupdate_after_occlusion=False)


def _frame(*boxes, conf=0.9):
    return DetectionResult(detections=[Detection(bbox=b, confidence=conf) for b in boxes], frame_shape=SHAPE)


def _box(x, w=100, y=300, h=300):
    return (x, y, x + w, y + h)


def test_direction_cost_favours_the_detection_ahead_of_a_moving_track():
    trk = KalmanBoxTracker(np.array(_box(100)))
    for i in range(1, 5):                       # walking right, 20 px per frame
        trk.predict()
        trk.update(np.array(_box(100 + 20 * i)))
    ahead, behind = np.array([_box(200)]), np.array([_box(140)])
    dets = np.vstack([ahead, behind]).astype(float)
    cost = direction_cost(dets, np.array([1.0, 1.0]), [trk])
    assert cost[0, 0] > 0.4 and cost[1, 0] < cost[0, 0]


def test_track_without_motion_has_no_direction_cost():
    trk = KalmanBoxTracker(np.array(_box(100)))
    assert direction_cost(np.array([_box(300)], float), np.array([0.9]), [trk])[0, 0] == 0.0


def _walk_hidden_then_back(config):
    """A person walks right (20 px/frame), is missed for 6 frames, reappears 5 frames' walk
    further than straight-line motion predicts (slowed down while hidden)."""
    tracker = ByteTracker("cam", config)
    for i in range(10):
        tracker.update(_frame(_box(100 + 20 * i)))
    for _ in range(6):
        tracker.update(_frame())
    # straight motion would put it at 100 + 20 * 16 = 420; it stopped at 290 behind a sofa
    return tracker, tracker.update(_frame(_box(290)))


def test_without_recovery_a_drifted_prediction_gives_a_new_track():
    _, result = _walk_hidden_then_back(BASE)
    assert [t.track_id for t in result.tracks if t.time_since_update == 0] != [1]


def test_recovery_matches_the_last_observed_box():
    _, result = _walk_hidden_then_back(replace(BASE, recover_last_observation=True))
    assert [t.track_id for t in result.tracks if t.time_since_update == 0] == [1]
    assert any(e.kind == "revive" and e.stage == 3 for e in result.association_events)


def test_reupdate_reruns_the_filter_once_per_missed_frame_along_the_straight_path():
    """ORU: after 6 missed frames the filter is restarted from the first missed frame's state and
    updated 7 times (one per frame) on boxes interpolated from the last observed one (x 280) to the
    new one (x 290); without ORU one update absorbs the whole jump."""
    for reupdate, expected in ((True, [281.4, 282.9, 284.3, 285.7, 287.1, 288.6, 290.0]), (False, [290.0])):
        tracker = ByteTracker("cam", replace(BASE, recover_last_observation=True, reupdate_after_occlusion=reupdate))
        for i in range(10):
            tracker.update(_frame(_box(100 + 20 * i)))
        for _ in range(6):
            tracker.update(_frame())
        kalman, seen = tracker._trackers[0], []
        original = kalman.kf.update
        kalman.kf.update = lambda z: (seen.append(round(float(z[0, 0]) - 50, 1)), original(z))
        tracker.update(_frame(_box(290)))
        assert seen == expected


def test_all_three_components_on_by_default():
    """Measured together (replay 10-23..10-24): OCR alone turns a takeover into a swap."""
    cfg = TrackingConfig()
    assert cfg.velocity_weight == 0.2 and cfg.recover_last_observation and cfg.reupdate_after_occlusion
