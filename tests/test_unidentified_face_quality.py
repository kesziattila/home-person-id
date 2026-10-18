"""Which unidentified faces are saved and suggested for enrollment.

History of the quality score:
- 2026-06 .. 2026-10-17: a hand-made formula (face size, sharpness = min of
  focus / motion / detail / noise terms, brightness, detector confidence,
  frontalness from landmarks) with a gate per term. Every live failure added a
  term or a threshold, and it still failed: on 2026-10-18 a faded, grainy dawn
  crop and a clean gallery face measured the same on every pixel statistic,
  and a face half hidden by a phone scored sharpness 1.0 and frontalness 1.0.
- Since 2026-10-18: one learned score, eDifFIQA-tiny (`FaceQualityAssessor`).
  On 51 hand-labelled live crops it separated good from bad 84% at a single
  threshold and correlated 0.75 with each face's similarity to its own
  person's gallery (hand-made formula: 0.31).

Also since 2026-10-18 a track keeps ONE suggestion, its best crop (one 4-min
phone track had produced 13 suggestions and evicted another person's crops
from the per-camera cap); a better crop archives the previous one.
"""

from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from src.config import UnidentifiedFacesConfig
from src.recognition.face_quality import FaceQualityAssessor
from src.recognition.unidentified_face_manager import UnidentifiedFaceManager, UnidentifiedFaceTask

MODEL = UnidentifiedFacesConfig().quality_model
LABELLED = Path("tests/test-images/sensitive/face_quality")
ALIGNED = np.zeros((112, 112, 3), dtype=np.uint8)


def _manager(tmp_path, **config):
    """Manager on a real SQLite repository; the quality model is stubbed per task."""
    from src.database.repository import Repository
    repo = Repository(str(tmp_path / "test.db"))
    config.setdefault("min_face_size", 10)
    manager = UnidentifiedFaceManager(
        UnidentifiedFacesConfig(enabled=True, images_path=str(tmp_path / "faces"), **config), repo)
    manager._quality_assessor = MagicMock()
    return manager, repo


def _task(manager, quality, track_id="global_1", seed=2, aligned=ALIGNED, size=160):
    manager._quality_assessor.score.return_value = quality
    crop = np.random.default_rng(seed).integers(0, 255, (size, size, 3), dtype=np.uint8)
    return UnidentifiedFaceTask(
        camera_id="cam_a", track_id=track_id, face_crop=crop,
        embedding=np.random.default_rng(seed).standard_normal(512).astype(np.float32),
        best_match_person_id=None, best_match_score=None, face_bbox=(0, 0, size, size),
        det_confidence=0.8, aligned_face=aligned,
    )


class TestGates:
    def test_face_below_min_quality_is_not_saved(self, tmp_path):
        manager, repo = _manager(tmp_path, min_quality_score=0.3)
        manager._process_task(_task(manager, 0.29))
        assert repo.get_unidentified_faces() == []

    def test_face_without_landmarks_is_not_saved(self, tmp_path):
        manager, repo = _manager(tmp_path)
        manager._process_task(_task(manager, 0.9, aligned=None))
        assert repo.get_unidentified_faces() == []
        manager._quality_assessor.score.assert_not_called()

    def test_small_face_is_not_scored_or_saved(self, tmp_path):
        """Live #836: a 44 px face (71 px margin crop) scored 0.42 — the model
        upscales to 112 and cannot see resolution; the size gate must."""
        manager, repo = _manager(tmp_path, min_face_size=64)
        manager._process_task(_task(manager, 0.42, size=44))
        assert repo.get_unidentified_faces() == []
        manager._quality_assessor.score.assert_not_called()

    def test_saved_face_stores_the_model_score(self, tmp_path):
        manager, repo = _manager(tmp_path)
        manager._process_task(_task(manager, 0.62))
        (face,) = repo.get_unidentified_faces()
        assert face.quality_score == pytest.approx(0.62)
        assert face.blur_score is None and face.frontalness is None


class TestOneSuggestionPerTrack:
    def test_worse_or_equal_crop_of_the_track_is_not_saved(self, tmp_path):
        manager, repo = _manager(tmp_path)
        manager._process_task(_task(manager, 0.6, seed=2))
        manager._process_task(_task(manager, 0.5, seed=3))
        manager._process_task(_task(manager, 0.6, seed=4))
        assert len(repo.get_unidentified_faces()) == 1

    def test_better_crop_archives_the_previous_suggestion(self, tmp_path):
        manager, repo = _manager(tmp_path)
        manager._process_task(_task(manager, 0.4, seed=2))
        manager._process_task(_task(manager, 0.6, seed=3))

        shown = repo.get_unidentified_faces(archived=False)
        archived = repo.get_unidentified_faces(archived=True)
        assert len(shown) == 1 and len(archived) == 1
        assert archived[0].superseded_by_id == shown[0].id
        assert Path(archived[0].image_path).exists()   # history kept, image too
        assert repo.get_unidentified_faces_summary() == {"cam_a": 1}

    def test_other_tracks_each_get_a_suggestion(self, tmp_path):
        manager, repo = _manager(tmp_path)
        manager._process_task(_task(manager, 0.5, "global_1", seed=2))
        manager._process_task(_task(manager, 0.5, "global_2", seed=3))
        assert len(repo.get_unidentified_faces(archived=False)) == 2

    def test_reviewed_suggestion_is_not_archived(self, tmp_path):
        manager, repo = _manager(tmp_path)
        manager._process_task(_task(manager, 0.4, seed=2))
        first = repo.get_unidentified_faces()[0]
        repo.update_unidentified_face(first.id, dismissed=True)
        manager._process_task(_task(manager, 0.6, seed=3))
        assert repo.get_unidentified_face(first.id).superseded_by_id is None
        assert len(repo.get_unidentified_faces(dismissed=False, archived=False)) == 1

    def test_archived_faces_do_not_count_in_the_camera_cap(self, tmp_path):
        manager, repo = _manager(tmp_path, max_per_camera=2)
        for i, quality in enumerate((0.4, 0.5, 0.6, 0.7)):     # one track improving
            manager._process_task(_task(manager, quality, "global_1", seed=10 + i))
        manager._process_task(_task(manager, 0.5, "global_2", seed=20))
        assert len(repo.get_unidentified_faces(archived=False)) == 2
        assert len(repo.get_unidentified_faces(archived=True)) == 3


def _aligned_live_faces(label):
    """Labelled live crops (gitignored), aligned by insightface landmarks like live."""
    paths = sorted((LABELLED / label).glob("*.jpg"))
    if not paths or not Path(MODEL).exists():
        pytest.skip("labelled live crops or face quality model not available")
    insightface = pytest.importorskip("insightface.app")
    from src.recognition.tensorrt_face import align_face
    app = insightface.FaceAnalysis(name="buffalo_l", allowed_modules=["detection"],
                                   providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(320, 320), det_thresh=0.3)
    faces = []
    for p in paths:
        img = cv2.copyMakeBorder(cv2.imread(str(p)), 40, 40, 40, 40, cv2.BORDER_CONSTANT)
        found = app.get(img)
        if found:   # the live detector found these faces in the full frame
            faces.append((p.name, align_face(img, max(found, key=lambda f: f.det_score).kps)))
    return faces


def test_labelled_live_crops_rank_by_usability():
    """2026-10-18 measurement: usable 0.29-0.70 (median 0.45), phone 0.22-0.37,
    faded 0.11-0.28, grainy 0.34-0.45. Bad crops are never "Learning" except one
    grainy borderline (0.454); usable crops below 0.45 are looking-down heads."""
    assessor = FaceQualityAssessor(MODEL)
    learn = UnidentifiedFacesConfig().learning_min_quality
    scores = {label: [assessor.score(face) for _, face in _aligned_live_faces(label)]
              for label in ("usable", "phone", "faded", "grainy")}
    bad = scores["phone"] + scores["faded"] + scores["grainy"]
    assert max(scores["phone"] + scores["faded"]) < learn
    assert sum(s >= learn for s in bad) <= 1
    assert np.median(scores["usable"]) > np.median(bad) + 0.1
