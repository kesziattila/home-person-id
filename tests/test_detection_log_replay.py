"""Detection log + offline ByteTrack replay (tools/replay_tracking.py) + frame capture.

The replay is only useful if it reproduces the live tracker exactly: feeding the
logged inputs into a fresh ByteTracker must give the same births and events as
the live run that produced the log.

Frame capture (2026-10-17): at 12:26 UTC Ati and Judit (bending next to him) got
ONE box and Virag at the top of the stairs none. Tuning the detector needs the
frames themselves: a time-limited capture saves the frame of every logged record,
tools/redetect.py re-runs the detector on them and writes a log the replay reads
unchanged.
"""

import dataclasses
import json
import time
from unittest.mock import MagicMock

import cv2
import numpy as np

from src.config import TrackingConfig
from src.detection.person_detector import Detection, DetectionResult
from src.tracking import detection_log
from src.tracking.byte_tracker import ByteTracker
from src.config import Config, DetectionConfig
from src.detection.person_detector import detect_for_tracking
from tools.redetect import apply_overrides, redetect
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


def _capture_log(tmp_path, until_offset, max_bytes=10**9):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / detection_log.TRIGGER_FILE).write_text(str(time.time() + until_offset))
    capture = detection_log.FrameCapture(tmp_path, max_bytes, retention_days=3)
    return detection_log.DetectionLog(tmp_path, retention_days=7, capture=capture), capture


def _frame(value=90):
    return np.full((72, 128, 3), value, dtype=np.uint8)


def test_capture_saves_the_frame_of_each_record_while_triggered(tmp_path):
    log, capture = _capture_log(tmp_path, until_offset=600)
    for i, frame in enumerate(_walk(n=3, gap=())):
        log.write("cam_a", time.time() + i * 0.2, frame, frame=_frame(40 * i))
    capture.flush()

    records = list(detection_log.load(tmp_path))
    assert all(r["frame"].startswith("frames/") for r in records)
    saved = cv2.imread(str(tmp_path / records[2]["frame"]))
    assert abs(int(saved.mean()) - 80) <= 2


def test_no_trigger_or_expired_trigger_captures_nothing(tmp_path):
    log, capture = _capture_log(tmp_path, until_offset=-1)
    log.write("cam_a", time.time(), _walk(n=1, gap=())[0], frame=_frame())
    capture.flush()
    assert "frame" not in next(detection_log.load(tmp_path))
    assert not (tmp_path / "frames").exists()


def test_capture_stops_at_the_size_cap(tmp_path):
    log, capture = _capture_log(tmp_path, until_offset=600, max_bytes=1)
    t0 = time.time()
    log.write("cam_a", t0, _walk(n=1, gap=())[0], frame=_frame())
    capture.flush()
    log.write("cam_a", t0 + 0.2, _walk(n=1, gap=())[0], frame=_frame())
    capture.flush()
    assert ["frame" in r for r in detection_log.load(tmp_path)] == [True, False]


def test_reid_only_for_confident_detections():
    frame = np.random.default_rng(0).integers(0, 255, (360, 640, 3), dtype=np.uint8)
    detector = MagicMock()
    detector.detect.return_value = DetectionResult(detections=[
        Detection(bbox=(10, 10, 110, 300), confidence=0.8),
        Detection(bbox=(300, 10, 400, 300), confidence=0.2)], frame_shape=frame.shape)
    extractor = MagicMock()
    extractor.extract_batch.side_effect = lambda crops: [(np.ones(4), 0.9)] * len(crops)

    dets, reid = detect_for_tracking(frame, detector, None, extractor, reid_min_confidence=0.5)

    assert len(dets.detections) == 2 and list(reid) == [0]


def test_redetect_writes_a_replayable_log_and_counts_the_difference(tmp_path):
    log, capture = _capture_log(tmp_path / "live", until_offset=600)
    t0 = time.time()
    frames = _walk(n=10, gap=())
    for i, frame in enumerate(frames):
        log.write("cam_a", t0 + i * 0.2, frame, frame=_frame())
    capture.flush()
    records = list(detection_log.load(tmp_path / "live"))
    (tmp_path / "live" / records[9]["frame"]).unlink()      # a frame the writer dropped
    detector = MagicMock()                                  # the variant also finds a second person
    detector.detect.side_effect = lambda img: DetectionResult(detections=[
        Detection(bbox=(50, 100, 170, 460), confidence=0.9),
        Detection(bbox=(400, 100, 520, 460), confidence=0.7)], frame_shape=SHAPE)
    config = dataclasses.replace(Config(), detection=DetectionConfig())

    stats = redetect(records, tmp_path / "live", config, tmp_path / "variant", with_reid=False,
                     draw_dir=tmp_path / "draw", detector=detector)["cam_a"]

    out = list(detection_log.load(tmp_path / "variant" / "detections"))
    assert [r["ts"] for r in out] == [r["ts"] for r in records]
    assert len(out[0]["dets"]) == 2 and out[9]["dets"] == records[9]["dets"]
    assert stats["redetected"] == 9 and stats["missing_image"] == 1
    assert stats["more"] == 9 and stats["fewer"] == 0
    assert len(list((tmp_path / "draw").glob("*.jpg"))) == 9
    assert replay(out, TrackingConfig())["cam_a"]["frames"] == 10


def test_overrides_reach_detection_and_reid_sections():
    config = apply_overrides(Config(), ["detection.model=yolo26m.pt", "detection.use_tensorrt_native=false",
                                        "reid.use_tensorrt_native=False", "detection.confidence_threshold=0.1"])
    assert config.detection.model == "yolo26m.pt" and config.detection.confidence_threshold == 0.1
    assert config.detection.use_tensorrt_native is False and config.reid.use_tensorrt_native is False
