"""Event capture: pre-roll + post window of downscaled frames around a merge / swap / crowd."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np

from src.tracking.event_capture import OFF_FILE, EventCapture, frame_rel_path, trigger_reasons

T0 = datetime(2026, 10, 24, 10, 0, 0, tzinfo=timezone.utc).timestamp()


def _capture(tmp_path, **kw):
    kw = {"pre_sec": 2.0, "post_sec": 1.0, "max_event_sec": 5.0, "cooldown_sec": 10.0, **kw}
    return EventCapture(tmp_path / "detections", tmp_path, retention_days=7, frames_retention_days=3, **kw)


def _frame(value=100):
    return np.full((40, 80, 3), value, np.uint8)


def _feed(cap, start, end, step=0.5, camera="living_room"):
    t = start
    while t <= end + 1e-9:
        cap.push(camera, T0 + t, _frame())
        t += step


def _saved(tmp_path):
    return sorted(p.name for p in (tmp_path / "detections" / "event_frames").rglob("*.jpg"))


def _events(tmp_path):
    return [json.loads(l) for f in (tmp_path / "event_capture").glob("*.jsonl") for l in open(f)]


def test_trigger_saves_preroll_and_post_window_at_half_size(tmp_path):
    cap = _capture(tmp_path)
    _feed(cap, 0.0, 5.0)                                   # ring keeps the last 2 s
    assert cap.trigger("living_room", T0 + 5.0, "merge")
    _feed(cap, 5.5, 7.0)                                   # post window until 6.0, closed at 6.5
    cap.flush()
    names = _saved(tmp_path)
    assert names[0] == "100003_000.jpg" and names[-1] == "100006_000.jpg" and len(names) == 7
    import cv2
    img = cv2.imread(str(tmp_path / "detections" / frame_rel_path("living_room", T0 + 5.0)))
    assert img.shape == (20, 40, 3)
    (event,) = _events(tmp_path)
    assert event["reasons"] == ["merge"] and event["frames"] == 7 and event["dropped"] == 0
    assert event["bytes"] > 0


def test_cooldown_and_event_length_budget(tmp_path):
    cap = _capture(tmp_path)
    _feed(cap, 0.0, 1.0)
    cap.trigger("living_room", T0 + 1.0, "merge")
    for t in (2.0, 3.0, 4.0, 5.0):                         # keeps extending, capped at start + 5 s
        _feed(cap, t - 0.5, t)
        cap.trigger("living_room", T0 + t, "crowd")
    _feed(cap, 5.5, 8.0)
    cap.flush()
    (event,) = _events(tmp_path)
    assert event["reasons"] == ["merge", "crowd"] and event["end"].endswith("10:00:04.000+00:00")
    assert not cap.trigger("living_room", T0 + 9.0, "merge")      # cooldown 10 s after the end
    assert cap.trigger("hallway", T0 + 9.0, "merge")              # per camera


def test_off_switch_file_stops_buffering_and_triggers(tmp_path):
    cap = _capture(tmp_path)
    (tmp_path / "detections").mkdir(parents=True, exist_ok=True)
    (tmp_path / "detections" / OFF_FILE).touch()
    _feed(cap, 0.0, 2.0)
    assert not cap.trigger("living_room", T0 + 2.0, "merge")
    cap.flush()
    assert _saved(tmp_path) == []


def _track(tid, uncertain=False, lost=False):
    return SimpleNamespace(track_id=tid, uncertain=uncertain, time_since_update=1 if lost else 0)


def test_trigger_reasons_fire_on_onsets_only():
    unswap = [SimpleNamespace(kind="unswap")]
    assert trigger_reasons([_track(1), _track(2)], unswap, set(), 2) == ["unswap"]
    assert trigger_reasons([_track(1, uncertain=True), _track(2)], [], set(), 2) == ["merge"]
    assert trigger_reasons([_track(1, uncertain=True), _track(2)], [], {1}, 2) == []      # still merged
    three = [_track(1), _track(2), _track(3)]
    assert trigger_reasons(three, [], set(), 2) == ["crowd"]
    assert trigger_reasons(three, [], set(), 3) == []
    assert trigger_reasons([_track(1), _track(2), _track(3, lost=True)], [], set(), 2) == []


def test_redetect_finds_event_frames_by_timestamp(tmp_path):
    from tools.redetect import frame_path
    cap = _capture(tmp_path)
    _feed(cap, 0.0, 1.0)
    cap.trigger("living_room", T0 + 1.0, "merge")
    cap.flush()
    record = {"camera_id": "living_room", "ts": datetime.fromtimestamp(T0 + 0.5, tz=timezone.utc).isoformat()}
    assert frame_path(record, tmp_path / "detections").name == "100000_500.jpg"
    record["ts"] = datetime.fromtimestamp(T0 - 30, tz=timezone.utc).isoformat()
    assert frame_path(record, tmp_path / "detections") is None


def test_manual_capture_frames_do_not_count_against_the_event_budget(tmp_path):
    """Live 2026-10-24 14:21: the first deploy counted the 28 GB of manual captures in frames/
    against the 5 GB event budget and stayed silently off."""
    manual = tmp_path / "detections" / "frames" / "2026-10-22" / "living_room"
    manual.mkdir(parents=True)
    (manual / "100000_000.jpg").write_bytes(b"x" * 5000)
    cap = _capture(tmp_path, max_bytes=1000)
    _feed(cap, 0.0, 1.0)
    assert cap.trigger("living_room", T0 + 1.0, "merge")
