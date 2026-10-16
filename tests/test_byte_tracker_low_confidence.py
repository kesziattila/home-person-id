"""ByteTrack second (low-confidence) association stage.

Live data (2026-06-06..10): 7026 tracks in 4 days for a 4-person household;
~1000 living_room tracks started within 3 s of the previous one. Root causes:

1. detection.confidence_threshold == tracking.track_thresh (0.5), so the
   detector dropped every detection the second stage could have used — people
   seated on the sofa or partly occluded hovered around 0.5 and their track was
   lost and re-created under a new ID.
2. Dormant bugs in the second stage that surface once it receives input:
   - `unmatched_trks` was mutated while indexed with assignment indices from
     the original ordering → wrong track left "unmatched" or IndexError when
     more than one low-confidence detection matched in a frame.
   - A LOST track recovered by a low-confidence detection stayed LOST, so it
     was missing from the active tracks reported downstream.

Expected behavior: low-confidence detections keep confirmed tracks alive (and
recover LOST ones), but never create or confirm tracks on their own.
"""

import pytest

from src.config import TrackingConfig
from src.detection.person_detector import Detection, DetectionResult
from src.tracking.byte_tracker import ByteTracker
from src.tracking.track import TrackState

FRAME_SHAPE = (720, 1280, 3)
HIGH = 0.8
LOW = 0.3


def _dets(*items: tuple[tuple[float, float, float, float], float]) -> DetectionResult:
    return DetectionResult(
        detections=[Detection(bbox=bbox, confidence=conf) for bbox, conf in items],
        frame_shape=FRAME_SHAPE,
    )


def _box(x: float) -> tuple[float, float, float, float]:
    return (x, 100.0, x + 120.0, 400.0)


@pytest.fixture
def tracker():
    return ByteTracker(
        "cam", TrackingConfig(track_thresh=0.5, match_thresh=0.3, track_buffer=30)
    )


def _confirm(tracker, xs):
    """Feed high-confidence frames until every box has a confirmed track."""
    for _ in range(4):  # creation frame + 3 hits
        result = tracker.update(_dets(*[(_box(x), HIGH) for x in xs]))
    assert len(result.tracks) == len(xs)
    return {t.track_id: t.bbox[0] for t in result.tracks}


def test_low_confidence_detection_keeps_confirmed_track_alive(tracker):
    ids = _confirm(tracker, [100])
    (track_id,) = ids

    for _ in range(10):
        result = tracker.update(_dets((_box(100), LOW)))
        assert [t.track_id for t in result.tracks] == [track_id]
        assert result.lost_track_ids == []
        assert result.new_track_ids == []


def test_low_confidence_detection_recovers_lost_track(tracker):
    (track_id,) = _confirm(tracker, [100])

    result = tracker.update(_dets())
    assert result.lost_track_ids == [track_id]
    assert result.tracks == []

    result = tracker.update(_dets((_box(100), LOW)))
    assert [t.track_id for t in result.tracks] == [track_id]
    assert result.tracks[0].state == TrackState.TRACKED
    assert result.new_track_ids == []


def test_low_confidence_detections_never_create_tracks(tracker):
    for _ in range(10):
        result = tracker.update(_dets((_box(100), LOW), (_box(600), LOW)))
        assert result.tracks == []
        assert result.new_track_ids == []


def test_low_confidence_detections_do_not_confirm_new_track(tracker):
    tracker.update(_dets((_box(100), HIGH)))  # NEW, unconfirmed (1 hit)

    for _ in range(5):
        result = tracker.update(_dets((_box(100), LOW)))
        assert result.tracks == []
        assert result.new_track_ids == []


def test_multiple_low_confidence_matches_update_their_own_tracks(tracker):
    ids = _confirm(tracker, [50, 400, 750])
    track_by_x = {x: tid for tid, x in ids.items()}

    # One high-confidence match, two low-confidence ones in the same frame:
    # exercises the unmatched-track bookkeeping of the second stage.
    for step in range(1, 6):
        shift = 5.0 * step
        result = tracker.update(_dets(
            (_box(50 + shift), HIGH),
            (_box(400 + shift), LOW),
            (_box(750 + shift), LOW),
        ))
        assert result.lost_track_ids == []
        assert result.new_track_ids == []
        by_id = {t.track_id: t for t in result.tracks}
        assert set(by_id) == set(ids)
        for x, tid in track_by_x.items():
            assert by_id[tid].bbox[0] == pytest.approx(x + shift)
            assert by_id[tid].state == TrackState.TRACKED


def test_all_low_confidence_matches_in_one_frame(tracker):
    ids = _confirm(tracker, [50, 400, 750])

    result = tracker.update(_dets(
        (_box(50), LOW), (_box(400), LOW), (_box(750), LOW),
    ))
    assert {t.track_id for t in result.tracks} == set(ids)
    assert result.lost_track_ids == []
