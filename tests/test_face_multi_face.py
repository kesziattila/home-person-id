"""Ambiguous face attempts: several recognizable faces in one track's head region.

Live 2026-10-17 07:12 UTC (living_room, global_56136): the track was a child
(Virag, track_created snapshot) and her face scored Virag 0.46-0.55. Then an adult
bent over her; the linker took the largest face in the head region and the track
got a face_match "Ati" (0.606). Of the 'wrong' high scores in the face score log
(top1 >= 0.45 against the track's anchor label) all 19 were this Ati/Virag mix-up.

Expected: with >1 face of at least min_face_size, the attempt is scored and logged
(num_faces, for analysis) but never accepted, never counted toward confirmation and
never saved as an unidentified (training) face; the analyzer ignores such records.
Small background faces below min_face_size do not make an attempt ambiguous.

Refinement (audit review 2026-10-17 10:25 UTC): sitting side by side, Virag's face
poked into the right edge of Ati's box and kept Ati unidentified for 90 s. The face
whose center lies in the middle half of the head region owns the box; only when no
face or several faces are central is the attempt ambiguous.
"""

import json
import time
from unittest.mock import MagicMock

import numpy as np

from src.config import FaceRecognitionConfig, MetricsConfig, ReIDConfig
from src.recognition.face_recognizer import Face, FaceDetectionResult
from src.recognition.face_score_logger import FaceScoreLogger
from src.recognition.identification_manager import FaceGalleryEntry
from src.recognition.identity_linker import IdentityLinker
from tools.analyze_face_scores import load_attempts

EMB = np.ones(512, dtype=np.float32)
# Head region of the (0,0,300,400) box is 300 px wide: middle half = x 75..225
ADULT = Face(bbox=(80.0, 10.0, 200.0, 150.0), confidence=0.9, embedding=EMB)   # 140 px, center 140
CHILD = Face(bbox=(100.0, 60.0, 200.0, 180.0), confidence=0.9, embedding=EMB)  # 120 px, center 150
EDGE = Face(bbox=(240.0, 40.0, 300.0, 120.0), confidence=0.9, embedding=EMB)   # 80 px, center 270
TINY = Face(bbox=(300.0, 10.0, 325.0, 40.0), confidence=0.9, embedding=EMB)    # 30 px


def _linker(tmp_path, faces):
    recognizer = MagicMock()
    # Like the real detectors: faces smaller than min_face_size are dropped
    recognizer.detect_faces.return_value = FaceDetectionResult(
        faces=[f for f in faces if min(f.width, f.height) >= 40], frame_shape=(400, 400, 3))
    recognizer.compare_embeddings_batch.return_value = np.array([0.7, 0.1])
    unidentified = MagicMock()
    linker = IdentityLinker(
        face_config=FaceRecognitionConfig(enabled=True, similarity_threshold=0.6, min_face_size=40),
        reid_config=ReIDConfig(enabled=False),
        repository=MagicMock(),
        face_recognizer=recognizer,
        reid_extractor=MagicMock(),
        face_score_logger=FaceScoreLogger(MetricsConfig(path=str(tmp_path), face_scores_min_interval_sec=0)),
        unidentified_face_manager=unidentified,
    )
    linker._id_manager._face_gallery = [
        FaceGalleryEntry(person_id=1, person_name="Ati", embedding=EMB, embedding_id=10),
        FaceGalleryEntry(person_id=3, person_name="Virag", embedding=EMB, embedding_id=30),
    ]
    linker._id_manager._gallery_loaded_at = time.time()
    return linker, unidentified


def _attempts(linker, n=3):
    state = linker.register_track("global_1")
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    for _ in range(n):
        linker._try_face_identification(state, frame, (0.0, 0.0, 300.0, 400.0), np.zeros((10, 10, 3)), 1, "cam")
    return state


def _records(tmp_path):
    return [json.loads(l) for f in sorted((tmp_path / "face_scores").glob("*.jsonl")) for l in f.read_text().splitlines()]


def test_two_recognizable_faces_are_never_accepted_but_logged(tmp_path):
    linker, unidentified = _linker(tmp_path, [ADULT, CHILD])
    state = _attempts(linker)

    assert state.person_id is None
    assert linker._id_manager.get_identity("global_1").consecutive_matches == 0
    records = _records(tmp_path)
    assert [r["num_faces"] for r in records] == [2, 2, 2]
    assert not any(r["accepted"] for r in records)
    unidentified.submit.assert_not_called()


def test_small_background_face_does_not_block_identification(tmp_path):
    linker, _ = _linker(tmp_path, [ADULT, TINY])
    state = _attempts(linker)

    assert state.person_id == 1
    assert {r["num_faces"] for r in _records(tmp_path)} == {1}


def test_analyzer_ignores_ambiguous_attempts(tmp_path):
    linker, _ = _linker(tmp_path, [ADULT, CHILD])
    _attempts(linker)
    assert load_attempts(tmp_path / "face_scores") == []


def test_neighbour_face_at_the_box_edge_does_not_block_the_owner(tmp_path):
    linker, unidentified = _linker(tmp_path, [ADULT, EDGE])
    state = _attempts(linker)

    assert state.person_id == 1
    assert {r["num_faces"] for r in _records(tmp_path)} == {2}
