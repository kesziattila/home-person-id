"""Identity audit sampler: evidence for Claude's offline spot-check review.

Mislabels are found by looking at what the system showed (frame + labels) next to
a sharp crop of each track — automatic metrics only catch failure modes known in
advance. Expected: a sample per camera at most every audit_interval_sec while
tracks are visible, plus an immediate one when a track's label changes (rate
limited), never one for an empty scene; each sample writes a JSONL record, an
annotated frame (downscaled) and full-resolution per-track crops.
"""

import json
from types import SimpleNamespace

import numpy as np

from src.config import MetricsConfig
from src.visualization.identity_audit import IdentityAuditSampler


def _sampler(tmp_path, **overrides):
    config = MetricsConfig(path=str(tmp_path), audit_interval_sec=60.0,
                           audit_change_min_interval_sec=5.0, audit_max_width=640, **overrides)
    return IdentityAuditSampler(config)


def _frame():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (1440, 2560, 3), dtype=np.uint8)


TRACK = SimpleNamespace(track_id=7, bbox=(1000.0, 400.0, 1400.0, 1300.0))


def _identity(name, method="face"):
    return SimpleNamespace(track_id="global_1", person_name=name, method=method,
                           confidence=0.67, face_info=(name, 0.67), is_reid_identified=False)


def test_periodic_and_label_change_triggers(tmp_path):
    s = _sampler(tmp_path)
    frame = _frame()
    assert s.observe("cam", frame, [], {}, now=100.0) is None                       # empty scene
    assert s.observe("cam", frame, [TRACK], {}, now=100.0) == "periodic"
    assert s.observe("cam", frame, [TRACK], {}, now=130.0) is None
    assert s.observe("cam", frame, [TRACK], {7: _identity("Ati")}, now=131.0) == "label_change"
    assert s.observe("cam", frame, [TRACK], {7: _identity("Virag")}, now=133.0) is None  # rate limited
    assert s.observe("cam", frame, [TRACK], {7: _identity("Virag")}, now=191.5) == "periodic"
    assert s.observe("other", frame, [TRACK], {}, now=133.0) == "periodic"          # per camera


def test_brief_track_loss_is_not_a_label_change(tmp_path):
    # Live: ByteTrack dropped a seated child's box for a few frames every ~10 s;
    # re-appearing with the same name must not count as a label change
    s = _sampler(tmp_path)
    frame = _frame()
    assert s.observe("cam", frame, [TRACK], {7: _identity("Virag")}, now=100.0) == "label_change"
    assert s.observe("cam", frame, [], {}, now=110.0) is None
    assert s.observe("cam", frame, [TRACK], {7: _identity("Virag")}, now=110.4) is None


def test_sample_writes_record_frame_and_full_res_crop(tmp_path):
    s = _sampler(tmp_path)
    s.observe("living_room", _frame(), [TRACK], {7: _identity("Ati", "reid_gallery")}, {7: "global_1"},
              now=1_800_000_000.0)
    s.flush()

    (jsonl,) = (tmp_path / "audit").glob("*.jsonl")
    record = json.loads(jsonl.read_text().splitlines()[0])
    assert record["camera_id"] == "living_room"
    (track,) = record["tracks"]
    assert track["person"] == "Ati" and track["method"] == "reid_gallery"
    assert track["global_id"] == "global_1"
    assert track["bbox"] == [0.3906, 0.2778, 0.5469, 0.9028]

    import cv2
    frame = cv2.imread(str(tmp_path / record["image"]))
    crop = cv2.imread(str(tmp_path / track["crop"]))
    assert frame.shape[1] == 640                       # annotated frame downscaled
    assert crop.shape[:2] == (640, 284)                # 900x400 crop capped at 640 high
