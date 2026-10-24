"""Body signature (src/tracking/body_signature.py) and the shadow body check in ByteTracker.

Live 2026-10-24: after two boxes merged, Re-ID could not tell Ati and Virag apart (light tops,
crosswise - straight 0.10 for a true swap), their build could: over 10-23..10-24 the signature caught
8/8 merge-loss swaps with 0/31 false (TUNING_LOG). The tracker scenes below are synthetic: an adult and
a child stand apart, the child walks into the adult's box, is hidden for a few frames, and the two
separate again; in the swap scene the box that stays has the child's build afterwards.
"""

from dataclasses import replace

import numpy as np

from src.config import TrackingConfig
from src.detection.person_detector import Detection, DetectionResult
from src.tracking.body_signature import HeightModel, distance, features, fit_height_model, scales
from src.tracking.byte_tracker import ByteTracker

W = H = 1000
ADULT = dict(head=0.33, shoulders=0.60, hips=0.38, legs=1.55)
CHILD = dict(head=0.55, shoulders=0.45, hips=0.30, legs=1.25)
CONFIG = TrackingConfig(track_thresh=0.5, track_buffer=60, match_thresh=0.3, stationary_timeout=0)


def skeleton(box, head, shoulders, hips, legs, conf=0.9):
    """(17, 3) COCO keypoints of a person standing upright in `box` with these proportions (over the torso)."""
    x1, y1, x2, y2 = box
    cx, h = (x1 + x2) / 2, y2 - y1
    torso = 0.94 * h / (head + 1 + legs)
    nose_y = y1 + 0.06 * h
    sh_y, hip_y = nose_y + head * torso, nose_y + head * torso + torso
    k = np.zeros((17, 3))
    k[:, 2] = conf
    k[0, :2] = (cx, nose_y)
    k[1, :2], k[2, :2] = (cx - 5, nose_y - 5), (cx + 5, nose_y - 5)
    k[3, :2], k[4, :2] = (cx - 12, nose_y), (cx + 12, nose_y)
    k[5, :2], k[6, :2] = (cx - shoulders * torso / 2, sh_y), (cx + shoulders * torso / 2, sh_y)
    k[7, :2], k[8, :2] = (k[5, 0], sh_y + 0.5 * torso), (k[6, 0], sh_y + 0.5 * torso)
    k[9, :2], k[10, :2] = (k[5, 0], hip_y), (k[6, 0], hip_y)
    for side, hx in ((0, cx - hips * torso / 2), (1, cx + hips * torso / 2)):
        k[11 + side, :2] = (hx, hip_y)
        k[13 + side, :2] = (hx, hip_y + legs * torso / 2)
        k[15 + side, :2] = (hx, hip_y + legs * torso)
    return k


def test_features_are_the_proportions_of_a_standing_body():
    f = features(skeleton((100, 100, 200, 700), **ADULT), (100, 100, 200, 700), (H, W, 3), None)
    assert np.allclose(f[:4], [ADULT["shoulders"], ADULT["hips"], ADULT["head"], ADULT["legs"]], atol=0.01)
    assert np.isnan(f[4])                              # no height model: no relative height


def test_a_sitting_body_gives_no_sample():
    k = skeleton((100, 100, 200, 700), **ADULT)
    k[13, :2], k[14, :2] = k[11, :2] + (120, 0), k[12, :2] + (120, 0)     # thighs horizontal
    assert features(k, (100, 100, 200, 700), (H, W, 3), None) is None


def test_relative_height_is_the_box_height_over_the_expected_one():
    model = HeightModel(coef=(0.5, 0, 0, 0, 0, 0), sd=0.1)          # 0.5 of the frame everywhere
    box = (100, 200, 200, 800)                                       # ankles at 800 -> 0.6 of the frame
    f = features(skeleton(box, **ADULT), box, (H, W, 3), model)
    assert abs(f[4] - 0.6 / 0.5) < 0.02
    cut = (100, 5, 200, 800)                                         # box cut at the top: no height
    assert np.isnan(features(skeleton(cut, **ADULT), cut, (H, W, 3), model)[4])


def test_distance_uses_only_the_features_both_have():
    sd = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    a = np.array([1.0, 1.0, np.nan, 1.0, np.nan])
    b = np.array([2.0, 1.0, 5.0, 1.0, 3.0])
    assert abs(distance(a, b, sd) - np.sqrt(1 / 3)) < 1e-9
    assert np.isnan(distance(np.full(5, np.nan), b, sd))


def test_fit_height_model_recovers_the_household_height_despite_outliers():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0.1, 0.9, 3000), rng.uniform(0.3, 0.95, 3000)
    expected = 0.1 + 0.05 * x + 0.5 * y
    person = rng.choice([1.15, 1.0, 0.85], 3000)                    # three people of different heights
    h = expected * person * rng.normal(1, 0.03, 3000)
    h[:150] *= 0.4                                                    # bent / cut boxes
    model = fit_height_model(x, y, h)
    for px, py in ((0.2, 0.4), (0.5, 0.7), (0.8, 0.9)):
        assert abs(model.expected(px, py) / (0.1 + 0.05 * px + 0.5 * py) - 1) < 0.03
    assert 0.08 < model.sd < 0.2


def _scene(swapped: bool, config=CONFIG):
    """Adult box stays at x 300-400; the child's box walks in from x 600, is hidden for 4 frames,
    reappears and walks back out. `swapped`: afterwards the staying box has the child's build and the
    leaving one the adult's (the people crossed, the boxes did not)."""
    tracker, events = ByteTracker("cam", config), []
    adult_box = (300, 200, 400, 800)

    def step(child_x, child_visible=True, swap=False):
        child_box = (child_x, 400, child_x + 80, 800)
        stay, leave = (CHILD, ADULT) if swap else (ADULT, CHILD)
        dets = [Detection(bbox=adult_box, confidence=0.9, keypoints=skeleton(adult_box, **stay))]
        if child_visible:
            dets.append(Detection(bbox=child_box, confidence=0.9, keypoints=skeleton(child_box, **leave)))
        result = tracker.update(DetectionResult(detections=dets, frame_shape=(H, W, 3)))
        events.extend(result.association_events)
        return result

    for _ in range(20):
        step(600)
    for x in range(600, 339, -10):
        step(x)
    for _ in range(4):
        step(340, child_visible=False)
    for x in list(range(340, 601, 10)) + [600] * 10:
        result = step(x, swap=swapped)
    return result, events


def test_body_check_flags_a_swap_after_a_merge_in_shadow():
    result, events = _scene(swapped=True)
    checks = [e for e in events if e.kind == "body_check"]
    assert len(checks) == 1
    n = checks[0].neighbors[0]
    assert n["would_swap"] and n["apart"] >= 0.5 and n["straight"] - n["cross"] >= 0.2
    # shadow: nothing swapped, the staying box keeps the adult's first track id
    staying = next(t for t in result.tracks if t.bbox[0] == 300)
    assert staying.track_id == 1


def test_body_check_passes_when_both_kept_their_build():
    _, events = _scene(swapped=False)
    checks = [e for e in events if e.kind == "body_check"]
    assert len(checks) == 1
    n = checks[0].neighbors[0]
    assert not n["would_swap"] and n["straight"] < n["cross"]


def test_body_check_off_logs_nothing():
    _, events = _scene(swapped=True, config=replace(CONFIG, body_check=False))
    assert not [e for e in events if e.kind in ("body_check", "body")]


def test_a_removed_track_logs_its_signature():
    tracker = ByteTracker("cam", replace(CONFIG, track_buffer=3))
    box = (300, 200, 400, 800)
    for _ in range(10):
        tracker.update(DetectionResult(detections=[Detection(bbox=box, confidence=0.9,
                                                             keypoints=skeleton(box, **ADULT))],
                                       frame_shape=(H, W, 3)))
    events = []
    for _ in range(5):
        events += tracker.update(DetectionResult(detections=[], frame_shape=(H, W, 3))).association_events
    body = [e for e in events if e.kind == "body"]
    assert len(body) == 1 and body[0].neighbors[0]["n"] >= 5
    assert abs(body[0].neighbors[0]["signature"][2] - ADULT["head"]) < 0.01
    assert body[0].neighbors[0]["signature"][4] is None


def test_height_model_comes_from_the_config_per_camera():
    config = replace(CONFIG, body_height_model={"cam": [0.5, 0, 0, 0, 0, 0, 0.12]})
    assert ByteTracker("cam", config)._height_model.sd == 0.12
    assert ByteTracker("other", config)._height_model is None
    assert np.isnan(scales(None)[4])
