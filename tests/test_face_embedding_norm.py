"""Raw ArcFace embedding norm as an identity-independent face quality signal.

The pixel-based sharpness score cannot separate moderately grainy night crops
from low-contrast but usable faces (2026-10-17). The L2 norm of the raw ArcFace
output (before normalization) is known to track face quality (small for blurred,
noisy or occluded faces) and is free — the embedding is computed anyway. It is
only logged for now (face score log `emb_norm`, unidentified_face_saved event
`embedding_norm`) so its separating power can be checked on live data before
gating on it.
"""

import json
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config import FaceRecognitionConfig, MetricsConfig, ReIDConfig, UnidentifiedFacesConfig
from src.recognition.face_recognizer import Face, FaceDetectionResult
from src.recognition.face_score_logger import FaceScoreLogger
from src.recognition.identification_manager import FaceGalleryEntry
from src.recognition.identity_linker import IdentityLinker
from src.recognition.tensorrt_face import TensorRTFaceEmbedder
from src.recognition.unidentified_face_manager import UnidentifiedFaceManager

EMB = np.ones(512, dtype=np.float32) / np.sqrt(512)


def test_embedder_returns_normalized_embedding_and_raw_norm():
    raw = np.full((1, 512), 0.9, dtype=np.float32)  # norm = 0.9 * sqrt(512) ≈ 20.4
    embedder = TensorRTFaceEmbedder.__new__(TensorRTFaceEmbedder)
    embedder.input_size = 112
    with patch.object(TensorRTFaceEmbedder, "_load_engine"), \
         patch.object(TensorRTFaceEmbedder, "_preprocess", return_value=np.zeros((1, 3, 112, 112))), \
         patch.object(TensorRTFaceEmbedder, "_run_inference", return_value=[raw]):
        embedding, norm = embedder.extract_with_norm(np.zeros((112, 112, 3), dtype=np.uint8))
        plain = embedder.extract(np.zeros((112, 112, 3), dtype=np.uint8))

    assert norm == pytest.approx(0.9 * np.sqrt(512), rel=1e-4)
    assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_allclose(plain, embedding)


def test_linker_logs_embedding_norm_and_passes_it_to_unidentified_faces(tmp_path):
    face = Face(bbox=(90.0, 10.0, 210.0, 150.0), confidence=0.9, embedding=EMB, embedding_norm=17.25)
    recognizer = MagicMock()
    recognizer.detect_faces.return_value = FaceDetectionResult(faces=[face], frame_shape=(400, 400, 3))
    recognizer.compare_embeddings_batch.return_value = np.array([0.40])
    unidentified = MagicMock()
    linker = IdentityLinker(
        face_config=FaceRecognitionConfig(enabled=True, similarity_threshold=0.6, min_face_size=40),
        reid_config=ReIDConfig(enabled=False),
        repository=MagicMock(),
        face_recognizer=recognizer,
        reid_extractor=MagicMock(),
        face_score_logger=FaceScoreLogger(MetricsConfig(path=str(tmp_path))),
        unidentified_face_manager=unidentified,
    )
    linker._id_manager._face_gallery = [FaceGalleryEntry(1, "Ati", EMB, 10)]
    linker._id_manager._gallery_loaded_at = time.time()

    state = linker.register_track("global_1")
    frame = np.full((400, 400, 3), 128, dtype=np.uint8)
    linker._try_face_identification(state, frame, (0.0, 0.0, 300.0, 400.0), np.zeros((10, 10, 3)), 1, "cam")

    (line,) = (tmp_path / "face_scores").glob("*.jsonl")
    assert json.loads(line.read_text().splitlines()[0])["emb_norm"] == pytest.approx(17.25)
    assert unidentified.submit.call_args.kwargs["embedding_norm"] == pytest.approx(17.25)


def test_saved_unidentified_face_event_carries_embedding_norm(tmp_path):
    repo = MagicMock()
    repo.get_unidentified_embeddings_for_camera.return_value = []
    repo.get_track_suggestion.return_value = None
    repo.add_unidentified_face.return_value = MagicMock(id=5)
    manager = UnidentifiedFaceManager(
        UnidentifiedFacesConfig(
            enabled=True, images_path=str(tmp_path), min_quality_score=0.0, min_face_size=10,
        ),
        repo,
    )
    manager._quality_assessor = MagicMock(score=MagicMock(return_value=0.6))
    task = MagicMock(
        camera_id="cam", track_id="global_1", face_crop=np.full((120, 100, 3), 128, dtype=np.uint8),
        embedding=EMB, best_match_person_id=None, best_match_score=None, best_match_face_id=None,
        face_bbox=(0, 0, 100, 120), det_confidence=0.8, aligned_face=np.zeros((112, 112, 3), np.uint8),
        embedding_norm=21.5,
    )
    manager._process_task(task)

    extra = repo.create_event.call_args.kwargs["extra_data"]
    assert extra["embedding_norm"] == pytest.approx(21.5)
    assert extra["quality_score"] == pytest.approx(0.6)
