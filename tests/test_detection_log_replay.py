"""Detection log + offline ByteTrack replay (tools/replay_tracking.py).

The replay is only useful if it reproduces the live tracker exactly: feeding the
logged inputs into a fresh ByteTracker must give the same births and events as
the live run that produced the log.
"""

import dataclasses
import time

import numpy as np

from src.config import TrackingConfig
from src.detection.person_detector import Detection, DetectionResult
from src.tracking import detection_log
from src.tracking.byte_tracker import ByteTracker
from tools.replay_tracking import replay, variants

SHAPE = (720, 1280, 3)


def _walk(n=40, step=60.0, gap=range(18, 21)):
    """A person walking right fast, with a few missed detections (gap) and a low-conf frame."""
    frames = []
    for i in range(n):
        x = 50 + i * step
        dets = [] if i in gap else [Detection(bbox=(x, 100.0, x + 120.0, 460.0), confidence=0.9 if i % 7 else 0.3)]
        frames.append(DetectionResult(detections=dets, frame_shape=SHAPE))
    return frames


def _write_and_load(tmp_path, frames, reid=None):
    log = detection_log.DetectionLog(tmp_path, retention_days=7)
    t0 = time.time()
    for i, frame in enumerate(frames):
        log.write("cam_a", t0 + i * 0.2, frame, reid if i == 0 else None)
    return list(detection_log.load(tmp_path))


def test_every_frame_is_logged_including_empty_ones(tmp_path):
    records = _write_and_load(tmp_path, _walk())
    assert len(records) == 40
    assert records[18]["dets"] == []
    assert len({r["session"] for r in records}) == 1


def test_roundtrip_rebuilds_the_tracker_input(tmp_path):
    frames = _walk()
    records = _write_and_load(tmp_path, frames)
    rebuilt = detection_log.to_detection_result(records[3])
    assert rebuilt.frame_shape == SHAPE
    np.testing.assert_allclose(rebuilt.to_numpy(), frames[3].to_numpy(), atol=0.01)


def test_reid_embeddings_survive_as_float16(tmp_path):
    emb = np.random.default_rng(0).standard_normal(512).astype(np.float32)
    emb /= np.linalg.norm(emb)
    records = _write_and_load(tmp_path, _walk(), reid={0: (emb, 0.8)})
    decoded, quality = detection_log.reid_embeddings(records[0])[0]
    assert quality == 0.8
    assert float(decoded @ emb) > 0.999


def test_replay_reproduces_the_live_tracker(tmp_path):
    config = TrackingConfig(match_thresh=0.3, track_buffer=30)
    frames = _walk()
    live = ByteTracker("cam_a", config)
    live_events = []
    for frame in frames:
        live_events += [dataclasses.asdict(e) for e in live.update(frame).association_events]

    result = replay(_write_and_load(tmp_path, frames), config)["cam_a"]

    assert result["births_total"] == sum(e["kind"] == "birth" for e in live_events)
    assert result["frames"] == 40


def test_session_boundary_resets_the_trackers(tmp_path):
    records = _write_and_load(tmp_path, _walk(n=10, gap=()))
    restarted = [{**r, "session": "later"} for r in records]
    one = replay(records, TrackingConfig())["cam_a"]
    two = replay(records + restarted, TrackingConfig())["cam_a"]
    assert two["births_total"] == 2 * one["births_total"]


def test_sweep_is_a_cartesian_product_over_set_overrides():
    assert variants(["track_buffer=90"], ["match_thresh=0.3,0.2", "track_thresh=0.5,0.6"]) == [
        {"track_buffer": 90, "match_thresh": 0.3, "track_thresh": 0.5},
        {"track_buffer": 90, "match_thresh": 0.3, "track_thresh": 0.6},
        {"track_buffer": 90, "match_thresh": 0.2, "track_thresh": 0.5},
        {"track_buffer": 90, "match_thresh": 0.2, "track_thresh": 0.6},
    ]
    assert variants([], []) == [{}]
