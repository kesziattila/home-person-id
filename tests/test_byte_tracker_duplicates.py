"""Duplicate guard + association diagnostics in ByteTrack.

Live 2026-10-16 ~20:20 (living_room): global_55804 lost its detection for one
frame, a new track global_55806 was born on the same person, and one second
later 55804 was "recovered" — two live tracks for one person until he left.
With the low-confidence stage enabled, a partial-body/duplicate box of the
person (inside the new track's box) overlapped the stale prediction of the old
track and revived it. Two tracks per person break `num_persons == 1` gating
(Re-ID, cross-camera propagation).

Expected: a low-confidence box lying mostly inside a box matched in stage 1
(IoA >= duplicate_ioa) must not revive another track; births/revivals/blocked
duplicates are reported as AssociationEvents for offline analysis.
"""

import pytest

from src.config import TrackingConfig
from src.detection.person_detector import Detection, DetectionResult
from src.tracking.byte_tracker import ByteTracker, ioa_batch
from src.tracking.track import TrackState
from tools.analyze_track_events import classify_birth, summarize

import numpy as np

FRAME_SHAPE = (1000, 1000, 3)
A_BOX = (100.0, 100.0, 220.0, 400.0)       # standing person
B_BOX = (100.0, 280.0, 220.0, 580.0)       # same person after a jump (IoU 0.25 with A)
PARTIAL = (100.0, 280.0, 220.0, 420.0)     # partial box inside B, IoU 0.375 with A's prediction


def _dets(*items):
    return DetectionResult(
        detections=[Detection(bbox=b, confidence=c) for b, c in items], frame_shape=FRAME_SHAPE
    )


@pytest.fixture
def tracker():
    return ByteTracker("living_room", TrackingConfig(track_thresh=0.5, match_thresh=0.3, track_buffer=30))


def _confirmed_a(tracker):
    for _ in range(4):
        result = tracker.update(_dets((A_BOX, 0.9)))
    (track,) = result.tracks
    return track.track_id


def _state(tracker, track_id):
    return tracker.get_track(track_id).state


def test_partial_box_inside_live_track_does_not_revive_stale_track(tracker):
    a_id = _confirmed_a(tracker)

    born = tracker.update(_dets((B_BOX, 0.9)))           # association fails -> new track B
    assert born.lost_track_ids == [a_id]
    (birth,) = [e for e in born.association_events if e.kind == "birth"]
    b_id = birth.track_id

    result = tracker.update(_dets((B_BOX, 0.9), (PARTIAL, 0.3)))

    assert _state(tracker, a_id) == TrackState.LOST       # NOT revived by the duplicate
    assert _state(tracker, b_id) == TrackState.TRACKED
    assert a_id not in {t.track_id for t in result.tracks}
    (skip,) = [e for e in result.association_events if e.kind == "dup_skip"]
    assert skip.track_id == a_id and skip.stage == 2


def test_low_confidence_box_without_live_overlap_still_revives(tracker):
    a_id = _confirmed_a(tracker)
    tracker.update(_dets())                                # lost

    result = tracker.update(_dets((A_BOX, 0.3)))

    assert [t.track_id for t in result.tracks] == [a_id]
    (revive,) = result.association_events
    assert (revive.kind, revive.track_id, revive.stage) == ("revive", a_id, 2)


def test_birth_event_reports_nearest_track_and_normalized_bbox(tracker):
    a_id = _confirmed_a(tracker)

    result = tracker.update(_dets((B_BOX, 0.9)))

    (birth,) = result.association_events
    assert birth.kind == "birth" and birth.stage == 1 and birth.camera_id == "living_room"
    assert birth.bbox == pytest.approx((0.1, 0.28, 0.22, 0.58))
    (neighbor,) = birth.neighbors
    assert neighbor["track_id"] == a_id
    assert neighbor["confirmed"] is True
    assert neighbor["iou"] == pytest.approx(0.25, abs=0.01)
    assert neighbor["center_dist"] < 1.0
    assert classify_birth(vars(birth))[0] == "assoc_fail"


def test_stage1_revive_event(tracker):
    a_id = _confirmed_a(tracker)
    tracker.update(_dets())
    result = tracker.update(_dets((A_BOX, 0.9)))
    (revive,) = result.association_events
    assert (revive.kind, revive.track_id, revive.stage) == ("revive", a_id, 1)


def test_ioa_batch_is_containment_fraction():
    ioa = ioa_batch(np.array([PARTIAL, A_BOX]), np.array([B_BOX]))
    assert ioa[0, 0] == pytest.approx(1.0)
    assert ioa[1, 0] == pytest.approx(120 * 120 / (120 * 300), abs=1e-6)


def _neighbor(**kw):
    base = {"track_id": 1, "state": "TRACKED", "confirmed": True, "time_since_update": 0,
            "iou": 0.0, "ioa": 0.0, "center_dist": 5.0}
    return {**base, **kw}


@pytest.mark.parametrize("neighbors,expected", [
    ([_neighbor(ioa=0.8)], "duplicate"),
    ([_neighbor(state="LOST", time_since_update=1, iou=0.22, center_dist=0.4)], "assoc_fail"),
    ([_neighbor(confirmed=False, center_dist=0.4)], "new"),
    ([_neighbor(center_dist=3.0)], "new"),
    ([], "new"),
])
def test_classify_birth(neighbors, expected):
    assert classify_birth({"neighbors": neighbors})[0] == expected


def test_summarize_counts_and_match_thresh_what_if():
    events = [
        {"kind": "birth", "camera_id": "c", "neighbors": [_neighbor(iou=0.22, center_dist=0.4)]},
        {"kind": "birth", "camera_id": "c", "neighbors": [_neighbor(iou=0.0, center_dist=0.6)]},
        {"kind": "birth", "camera_id": "c", "neighbors": []},
        {"kind": "revive", "camera_id": "c", "stage": 2},
        {"kind": "dup_skip", "camera_id": "c"},
    ]
    r = summarize(events)["c"]
    assert r["births"] == {"assoc_fail": 2, "new": 1}
    assert r["revives"] == {"stage2": 1}
    assert r["dup_skips"] == 1
    assert r["assoc_fail_recovered_by_match_thresh"]["0.2"] == 1
    assert r["assoc_fail_zero_iou"] == 1
