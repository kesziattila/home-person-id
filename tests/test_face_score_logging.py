"""Top1/top2 face score logging and margin-based match policy.

Live data (2026-06-06..10) showed face_match scores bunched at 0.60-0.65,
right above the 0.6 threshold, while enrolled-gallery impostor pairs never
exceeded 0.31 — the threshold rejects most genuine matches. Before lowering
it, every attempt's per-person similarities are logged so a looser policy
(lower threshold + min top1-top2 margin) can be validated on live data with
tools/analyze_face_scores.py. These tests cover the scoring, the policy,
the logger and the offline analysis.
"""

import json
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.config import Config, FaceRecognitionConfig, MetricsConfig, ReIDConfig
from src.recognition.face_recognizer import Face, FaceDetectionResult
from src.recognition.face_score_logger import FaceScoreLogger
from src.recognition.identification_manager import FaceGalleryEntry, FaceMatchScores, IdentificationManager
from src.recognition.identity_linker import IdentityLinker
from tools.analyze_face_scores import Attempt, analyze, emulate_track, load_attempts

EMB = np.ones(512, dtype=np.float32)


def _gallery():
    return [
        FaceGalleryEntry(person_id=1, person_name="Ati", embedding=EMB, embedding_id=10),
        FaceGalleryEntry(person_id=1, person_name="Ati", embedding=EMB, embedding_id=11),
        FaceGalleryEntry(person_id=2, person_name="Judit", embedding=EMB, embedding_id=20),
        FaceGalleryEntry(person_id=3, person_name="Virag", embedding=EMB, embedding_id=30),
    ]


def _manager(similarities):
    config = Config(face_recognition=FaceRecognitionConfig(enabled=True), reid=ReIDConfig(enabled=False))
    recognizer = MagicMock()
    recognizer.compare_embeddings_batch.return_value = np.array(similarities)
    return IdentificationManager(config, face_gallery=_gallery(), face_recognizer=recognizer)


# ---------------------------------------------------------------- scoring

def test_score_face_aggregates_per_person_and_takes_runner_up_from_other_person():
    scores = _manager([0.40, 0.55, 0.30, 0.10]).score_face(EMB)

    assert scores.best_entry.embedding_id == 11
    assert scores.per_person == pytest.approx({1: 0.55, 2: 0.30, 3: 0.10})
    assert scores.top1 == pytest.approx(0.55)
    # Ati's second embedding (0.40) is NOT the runner-up: top2 is the best other person
    assert scores.top2 == pytest.approx(0.30)
    assert scores.margin == pytest.approx(0.25)


def test_score_face_without_gallery_is_empty():
    manager = _manager([])
    manager._face_gallery = []
    scores = manager.score_face(EMB)
    assert scores.best_entry is None
    assert not scores.accepts(0.0)


@pytest.mark.parametrize("top1,top2,threshold,margin,expected", [
    (0.62, 0.10, 0.60, 0.00, True),
    (0.58, 0.10, 0.60, 0.00, False),   # below threshold
    (0.45, 0.10, 0.42, 0.10, True),    # lower threshold, clear lead
    (0.45, 0.40, 0.42, 0.10, False),   # ambiguous: two people score alike
])
def test_accepts_requires_threshold_and_margin(top1, top2, threshold, margin, expected):
    scores = FaceMatchScores(best_entry=_gallery()[0], top1=top1, top2=top2)
    assert scores.accepts(threshold, margin) is expected


# ---------------------------------------------------------------- linker

def _linker(tmp_path, similarities, threshold=0.42, min_margin=0.10):
    recognizer = MagicMock()
    recognizer.detect_faces.return_value = FaceDetectionResult(
        faces=[Face(bbox=(100.0, 10.0, 200.0, 130.0), confidence=0.9, embedding=EMB)],
        frame_shape=(400, 400, 3),
    )
    recognizer.compare_embeddings_batch.return_value = np.array(similarities)
    score_logger = FaceScoreLogger(MetricsConfig(path=str(tmp_path)))
    linker = IdentityLinker(
        face_config=FaceRecognitionConfig(
            enabled=True, similarity_threshold=threshold, min_margin=min_margin, min_face_size=40,
        ),
        reid_config=ReIDConfig(enabled=False),
        repository=MagicMock(),
        face_recognizer=recognizer,
        reid_extractor=MagicMock(),
        face_score_logger=score_logger,
    )
    linker._id_manager._face_gallery = _gallery()
    linker._id_manager._gallery_loaded_at = time.time()  # keep the injected gallery
    return linker, score_logger


def _attempt(linker, track_id="global_1"):
    state = linker.register_track(track_id)
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    linker._try_face_identification(state, frame, (0.0, 0.0, 300.0, 400.0), np.zeros((10, 10, 3)), 1, "cam")
    return state


def _records(score_logger):
    return [json.loads(l) for f in sorted(score_logger.dir.glob("*.jsonl")) for l in f.read_text().splitlines()]


def test_linker_rejects_ambiguous_match_and_logs_scores(tmp_path):
    linker, score_logger = _linker(tmp_path, [0.50, 0.48, 0.45, 0.0])
    state = _attempt(linker)
    _attempt(linker)

    assert state.person_id is None
    (record,) = _records(score_logger)[:1]
    assert record["track_id"] == "global_1"
    assert record["camera_id"] == "cam"
    assert record["top1_person_id"] == 1
    assert record["top1"] == pytest.approx(0.50)
    assert record["top2"] == pytest.approx(0.45)
    assert record["scores"] == {"1": 0.5, "2": 0.45, "3": 0.0}
    assert record["accepted"] is False
    assert record["face_h"] == pytest.approx(120.0)


def test_linker_confirms_clear_match_below_old_threshold(tmp_path):
    linker, score_logger = _linker(tmp_path, [0.50, 0.30, 0.20, 0.0])
    state = _attempt(linker)
    _attempt(linker)  # 2 consecutive accepted attempts confirm

    assert state.person_id == 1
    assert [r["accepted"] for r in _records(score_logger)] == [True, True]


def test_default_policy_is_unchanged(tmp_path):
    # Defaults: threshold 0.6, margin 0 -> same decision as before the change
    linker, _ = _linker(tmp_path, [0.59, 0.10, 0.10, 0.10], threshold=0.6, min_margin=0.0)
    state = _attempt(linker)
    _attempt(linker)
    assert state.person_id is None


# ---------------------------------------------------------------- logger

def _log(score_logger, track_id, accepted, now):
    return score_logger.log(
        camera_id="cam", track_id=track_id, per_person={1: 0.5}, top1_person_id=1,
        top1=0.5, top2=0.1, accepted=accepted, face_height=80, face_width=60,
        det_confidence=0.8, threshold=0.6, min_margin=0.0, now=now,
    )


def test_logger_rate_limits_rejected_attempts_per_track_but_keeps_accepted(tmp_path):
    score_logger = FaceScoreLogger(MetricsConfig(path=str(tmp_path), face_scores_min_interval_sec=1.0))
    t0 = 1_800_000_000.0

    assert _log(score_logger, "a", False, t0)
    assert not _log(score_logger, "a", False, t0 + 0.5)   # rate-limited
    assert _log(score_logger, "b", False, t0 + 0.5)       # other track unaffected
    assert _log(score_logger, "a", True, t0 + 0.6)        # accepted always logged
    assert _log(score_logger, "a", False, t0 + 1.7)

    assert len(_records(score_logger)) == 4


def test_logger_prunes_files_older_than_retention(tmp_path):
    score_logger = FaceScoreLogger(MetricsConfig(path=str(tmp_path), retention_days=7))
    now = datetime(2026, 10, 16, 12, tzinfo=timezone.utc)
    old = score_logger.dir / f"{(now - timedelta(days=8)):%Y-%m-%d}.jsonl"
    recent = score_logger.dir / f"{(now - timedelta(days=6)):%Y-%m-%d}.jsonl"
    old.write_text("")
    recent.write_text("")

    _log(score_logger, "a", False, now.timestamp())

    assert not old.exists()
    assert recent.exists()
    assert (score_logger.dir / "2026-10-16.jsonl").exists()


# ---------------------------------------------------------------- analysis

def _a(track, i, person, top1, top2):
    return Attempt(ts=f"2026-10-16T10:00:{i:02d}", track_id=track, camera_id="cam",
                   top1_person_id=person, top1=top1, top2=top2)


def test_emulate_track_needs_two_accepted_attempts_for_same_person():
    attempts = [_a("t", 0, 1, 0.45, 0.1), _a("t", 1, 2, 0.30, 0.1), _a("t", 2, 1, 0.47, 0.1)]
    assert emulate_track(attempts, 0.42, 0.1) == 1          # rejection in between does not reset
    assert emulate_track(attempts, 0.60, 0.0) is None
    assert emulate_track([_a("t", 0, 1, 0.5, 0.1), _a("t", 1, 2, 0.5, 0.1)], 0.42, 0.1) is None


def test_emulate_track_one_strong_attempt_confirms():
    attempts = [_a("t", 0, 2, 0.45, 0.1), _a("t", 1, 1, 0.62, 0.40)]
    assert emulate_track(attempts, 0.42, 0.1) is None
    assert emulate_track(attempts, 0.42, 0.1, strong=(0.58, 0.15)) == 1
    assert emulate_track([_a("t", 0, 1, 0.62, 0.50)], 0.42, 0.1, strong=(0.58, 0.15)) is None


def test_analyze_recommends_nothing_without_enough_evidence():
    attempts = [_a("t1", 0, 1, 0.65, 0.1), _a("t1", 1, 1, 0.45, 0.1)]
    report = analyze(attempts)
    assert report.labeled_tracks == 1
    assert report.recommended is None
    assert "insufficient" in report.reason


def test_analyze_recommends_lower_threshold_on_clean_live_data():
    attempts = []
    # 120 tracks x 4 weak genuine attempts: ~400 clean accepts are needed before the
    # Wilson lower bound of precision reaches 0.99 (deliberately conservative)
    for t in range(120):
        person = 1 + t % 3
        attempts.append(_a(f"t{t}", 0, person, 0.66, 0.05))
        for i, s in enumerate([0.40, 0.44, 0.47, 0.52], start=1):
            attempts.append(_a(f"t{t}", i, person, s, 0.12))
    for t in range(30):  # never-anchored tracks with weak but clear scores
        attempts += [_a(f"u{t}", 0, 1, 0.46, 0.1), _a(f"u{t}", 1, 1, 0.48, 0.1)]

    report = analyze(attempts)

    assert report.current_policy.tracks_identified == 0  # anchors alone never confirm twice
    assert report.recommended is not None
    assert report.recommended.threshold < 0.6
    assert report.recommended.labeled_tracks_wrong == 0
    assert report.recommended.track_coverage > report.current_policy.track_coverage


def test_analyze_rejects_policy_that_confuses_people():
    attempts = []
    for t in range(60):
        attempts.append(_a(f"t{t}", 0, 1, 0.66, 0.05))
        # weak attempts where the WRONG person wins narrowly
        attempts += [_a(f"t{t}", i, 2, 0.45, 0.42) for i in range(1, 5)]
    report = analyze(attempts)
    if report.recommended:
        # only policies whose margin filters out the confusions may be recommended
        assert report.recommended.labeled_accepted == 0 or report.recommended.precision >= 0.99


def test_load_attempts_skips_partial_lines_and_old_days(tmp_path):
    (tmp_path / "2026-10-15.jsonl").write_text(json.dumps(
        {"ts": "2026-10-15T00:00:00", "track_id": "x", "top1_person_id": 1, "top1": 0.5, "top2": 0.1}) + "\n")
    (tmp_path / "2026-10-16.jsonl").write_text(json.dumps(
        {"ts": "2026-10-16T00:00:00", "track_id": "y", "top1_person_id": 2, "top1": 0.5, "top2": 0.1})
        + "\n{\"ts\": \"2026-10-16T00:0")
    attempts = load_attempts(tmp_path, since="2026-10-16")
    assert [a.track_id for a in attempts] == ["y"]


def test_attempt_record_carries_the_face_embedding_for_offline_rescoring(tmp_path):
    """The embedding (float16) lets a gallery change be re-scored on past attempts offline."""
    from src.recognition.face_score_logger import decode_embedding
    linker, score_logger = _linker(tmp_path, [0.50, 0.48, 0.45, 0.0])
    _attempt(linker)
    (record,) = _records(score_logger)[:1]
    emb = decode_embedding(record)
    assert emb is not None and emb.shape == np.asarray(EMB).shape
    assert np.allclose(emb, np.asarray(EMB, dtype=np.float16).astype(np.float32))
    assert decode_embedding({"emb": None}) is None
