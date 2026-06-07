"""Tests for the revamped unidentified-face quality scoring (2026-06-06).

Live-data findings that drove the changes:
- The FFT high-frequency-ratio sharpness metric measured sensor/JPEG noise as
  much as detail: at native size it scored all visually usable live crops
  0.15-0.29 (below the 0.4 default gate) while ranking a heavily
  motion-blurred face above all of them. Replaced by min(Laplacian-variance
  focus score, gradient-anisotropy motion score) on a denoised fixed 112x112
  input — calibrated so usable live crops score 0.87+, the motion-blurred
  sample 0.36, gaussian-blurred copies <=0.03.
- InsightFace detection confidence tracked embedding utility better than any
  other component but was unused — now weighted into quality_score.
- One sitting episode produced 13 near-duplicate crops in 17s, churning the
  per-camera-capped review queue — now burst-suppressed per track.
"""

import time
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from src.config import UnidentifiedFacesConfig
from src.recognition.unidentified_face_manager import (
    UnidentifiedFaceManager,
    UnidentifiedFaceTask,
)


def _make_manager(**config_overrides) -> tuple[UnidentifiedFaceManager, MagicMock]:
    config_overrides.setdefault("images_path", "data/debug/test_unidentified")
    config = UnidentifiedFacesConfig(enabled=True, **config_overrides)
    repo = MagicMock()
    repo.get_unidentified_embeddings_for_camera.return_value = []
    manager = UnidentifiedFaceManager(config=config, repository=repo)
    return manager, repo


def _textured_crop(size: int = 160, seed: int = 1) -> np.ndarray:
    """A noisy textured crop — sharp by construction."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (size, size, 3), dtype=np.uint8)


class TestSharpnessFixedSize:
    def test_score_is_size_invariant(self):
        manager, _ = _make_manager()
        base = _textured_crop(112)
        upscaled = cv2.resize(base, (400, 400), interpolation=cv2.INTER_NEAREST)
        s_small = manager._compute_sharpness(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY))
        s_large = manager._compute_sharpness(cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY))
        # Same content at different sizes must score comparably (was wildly
        # size-dependent at native-size FFT)
        assert abs(s_small - s_large) < 0.25

    def test_blur_lowers_score(self):
        manager, _ = _make_manager()
        sharp = cv2.cvtColor(_textured_crop(160), cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(sharp, (21, 21), 0)
        assert manager._compute_sharpness(sharp) > manager._compute_sharpness(blurred)

    def test_sharp_texture_passes_default_gate(self):
        manager, _ = _make_manager()
        sharp = cv2.cvtColor(_textured_crop(160), cv2.COLOR_BGR2GRAY)
        assert manager._compute_sharpness(sharp) >= 0.4


class TestQualityWithDetConfidence:
    def test_det_confidence_raises_quality(self):
        manager, _ = _make_manager()
        crop = _textured_crop(160)
        bbox = (0, 0, 160, 160)
        q_high, _ = manager._compute_quality(crop, bbox, det_confidence=0.95)
        q_low, _ = manager._compute_quality(crop, bbox, det_confidence=0.30)
        assert q_high > q_low

    def test_legacy_weighting_without_det_confidence(self):
        manager, _ = _make_manager()
        crop = _textured_crop(160)
        bbox = (0, 0, 160, 160)
        q_legacy, sharp = manager._compute_quality(crop, bbox)
        q_with_det, _ = manager._compute_quality(crop, bbox, det_confidence=1.0)
        assert 0.0 <= q_legacy <= 1.0
        assert 0.0 <= q_with_det <= 1.0
        assert 0.0 <= sharp <= 1.0


class TestBurstSuppression:
    def _task(self, track_id="global_1"):
        return UnidentifiedFaceTask(
            camera_id="cam_a",
            track_id=track_id,
            face_crop=_textured_crop(160),
            embedding=np.random.default_rng(2).random(512).astype(np.float32),
            best_match_person_id=None,
            best_match_score=None,
            face_bbox=(0, 0, 160, 160),
            det_confidence=0.9,
            landmarks=_landmarks(0.0, 1.2),  # frontal — pass the pose gate
        )

    def test_second_save_within_window_is_suppressed(self, tmp_path):
        manager, repo = _make_manager(
            images_path=str(tmp_path), burst_min_interval_sec=10.0,
            min_quality_score=0.0, min_sharpness_score=0.0, min_face_size=10,
        )
        face = MagicMock()
        face.id = 1
        repo.add_unidentified_face.return_value = face

        manager._process_task(self._task())
        manager._process_task(self._task())
        assert repo.add_unidentified_face.call_count == 1

    def test_different_track_not_suppressed(self, tmp_path):
        manager, repo = _make_manager(
            images_path=str(tmp_path), burst_min_interval_sec=10.0,
            min_quality_score=0.0, min_sharpness_score=0.0, min_face_size=10,
        )
        face = MagicMock()
        face.id = 1
        repo.add_unidentified_face.return_value = face

        manager._process_task(self._task("global_1"))
        manager._process_task(self._task("global_2"))
        assert repo.add_unidentified_face.call_count == 2

    def test_save_allowed_after_window_expires(self, tmp_path):
        manager, repo = _make_manager(
            images_path=str(tmp_path), burst_min_interval_sec=10.0,
            min_quality_score=0.0, min_sharpness_score=0.0, min_face_size=10,
        )
        face = MagicMock()
        face.id = 1
        repo.add_unidentified_face.return_value = face

        manager._process_task(self._task())
        manager._last_saved_per_track["global_1"] = time.time() - 11.0
        manager._process_task(self._task())
        assert repo.add_unidentified_face.call_count == 2

    def test_zero_interval_disables_suppression(self, tmp_path):
        manager, repo = _make_manager(
            images_path=str(tmp_path), burst_min_interval_sec=0.0,
            min_quality_score=0.0, min_sharpness_score=0.0, min_face_size=10,
        )
        face = MagicMock()
        face.id = 1
        repo.add_unidentified_face.return_value = face

        manager._process_task(self._task())
        manager._process_task(self._task())
        assert repo.add_unidentified_face.call_count == 2


@pytest.mark.parametrize("path,min_expected", [
    ("local/remote/snapshots/data/unidentified_faces/living_room/20260606_175844_31ec98e4.jpg", 0.4),
    ("local/remote/snapshots/data/faces/person_4/20260606_165829_d6d48726.jpg", 0.4),
])
def test_real_live_crops_pass_default_sharpness_gate(path, min_expected):
    """Calibration regression: visually-usable live crops must pass the 0.4 gate.

    Skipped when the fetched live samples are not present on this machine.
    """
    img = cv2.imread(path)
    if img is None:
        pytest.skip(f"live sample not available: {path}")
    manager, _ = _make_manager()
    score = manager._compute_sharpness(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    assert score >= min_expected


@pytest.mark.parametrize("path", [
    # Evening captures 2026-06-06 19:13 local: camera low-light temporal NR
    # smeared away facial detail; the old metric scored these 0.48-1.0
    "local/remote/snapshots/data/unidentified_faces/ground_floor/20260606_191410_25f69e7e.jpg",
    "local/remote/snapshots/data/unidentified_faces/living_room/20260606_191355_bd921d2c.jpg",
    "local/remote/snapshots/data/unidentified_faces/hallway/20260606_191352_ad2ea10d.jpg",
    "local/remote/snapshots/data/unidentified_faces/entrance/20260606_191345_c70cebe9.jpg",
])
def test_nr_smeared_live_crops_fail_default_sharpness_gate(path):
    """Calibration regression: NR-smeared low-light crops must FAIL the 0.4 gate.

    These are the user-flagged 'noisy/blurry' captures: the fine-grain detail
    component rejects them (they measure 0.64-1.00 vs 1.55+ for usable crops).
    Skipped when the fetched live samples are not present on this machine.
    """
    img = cv2.imread(path)
    if img is None:
        pytest.skip(f"live sample not available: {path}")
    manager, _ = _make_manager()
    score = manager._compute_sharpness(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    assert score < 0.4


def _landmarks(yaw_offset: float, ratio: float) -> np.ndarray:
    """Synthesize 5-point landmarks with the given pose geometry.

    Eyes at (-0.5, 0) and (0.5, 0) so eye_dist == 1; mouth midline at
    y == ratio; nose offset along the eye axis from the face center.
    """
    le, re = np.array([-0.5, 0.0]), np.array([0.5, 0.0])
    ml, mr = np.array([-0.25, ratio]), np.array([0.25, ratio])
    nose = np.array([yaw_offset, ratio / 2.0])
    return np.stack([le, re, nose, ml, mr])


class TestFrontalness:
    """Geometry values below are measurements from labeled live crops."""

    @pytest.mark.parametrize("yaw,ratio", [
        (0.316, 1.057),  # frontal gallery crop
        (0.007, 1.028),  # frontal live crop
        (0.442, 1.360),  # near-frontal live crop
    ])
    def test_frontal_passes_default_gate(self, yaw, ratio):
        score = UnidentifiedFaceManager._compute_frontalness(_landmarks(yaw, ratio))
        assert score >= 0.5

    @pytest.mark.parametrize("yaw,ratio", [
        (1.758, 6.468),  # full profile
        (0.855, 2.088),  # 3/4 profile
        (0.105, 0.656),  # looking straight down
    ])
    def test_profile_and_lookdown_fail_default_gate(self, yaw, ratio):
        score = UnidentifiedFaceManager._compute_frontalness(_landmarks(yaw, ratio))
        assert score < 0.5

    def test_degenerate_landmarks_score_zero(self):
        collapsed = np.zeros((5, 2))
        assert UnidentifiedFaceManager._compute_frontalness(collapsed) == 0.0

    def test_profile_task_is_not_saved(self, tmp_path):
        manager, repo = _make_manager(
            images_path=str(tmp_path), min_quality_score=0.0,
            min_sharpness_score=0.0, min_face_size=10, min_frontalness=0.5,
        )
        task = UnidentifiedFaceTask(
            camera_id="cam_a",
            track_id="global_1",
            face_crop=_textured_crop(160),
            embedding=np.random.default_rng(3).random(512).astype(np.float32),
            best_match_person_id=None,
            best_match_score=None,
            face_bbox=(0, 0, 160, 160),
            landmarks=_landmarks(1.758, 6.468),  # full profile
        )
        manager._process_task(task)
        repo.add_unidentified_face.assert_not_called()

    def test_task_without_landmarks_is_rejected(self, tmp_path):
        """A crop without landmarks cannot be pose-checked — reject it.

        Changed 2026-06-07: live face #554 (top-of-head, no re-detectable
        face) was stored and flagged learnable; landmark-less submissions now
        score frontalness 0.0 and fail the gate. Live detections always carry
        landmarks, so this only rejects untrustworthy crops.
        """
        manager, repo = _make_manager(
            images_path=str(tmp_path), min_quality_score=0.0,
            min_sharpness_score=0.0, min_face_size=10, min_frontalness=0.5,
        )
        task = UnidentifiedFaceTask(
            camera_id="cam_a",
            track_id="global_1",
            face_crop=_textured_crop(160),
            embedding=np.random.default_rng(3).random(512).astype(np.float32),
            best_match_person_id=None,
            best_match_score=None,
            face_bbox=(0, 0, 160, 160),
        )
        manager._process_task(task)
        repo.add_unidentified_face.assert_not_called()

    def test_frontalness_weighted_into_quality(self):
        """Same crop, frontal vs profile pose: quality must differ."""
        manager, _ = _make_manager()
        crop = _textured_crop(200)
        bbox = (0, 0, 200, 200)
        q_frontal, _ = manager._compute_quality(crop, bbox, det_confidence=0.9, frontalness=1.0)
        q_profile, _ = manager._compute_quality(crop, bbox, det_confidence=0.9, frontalness=0.0)
        assert q_frontal - q_profile >= 0.15

    def test_saved_face_stores_frontalness_and_det_confidence(self, tmp_path):
        manager, repo = _make_manager(
            images_path=str(tmp_path), min_quality_score=0.0,
            min_sharpness_score=0.0, min_face_size=10, min_frontalness=0.5,
            burst_min_interval_sec=0.0,
        )
        face = MagicMock()
        face.id = 1
        repo.add_unidentified_face.return_value = face
        task = UnidentifiedFaceTask(
            camera_id="cam_a",
            track_id="global_1",
            face_crop=_textured_crop(160),
            embedding=np.random.default_rng(3).random(512).astype(np.float32),
            best_match_person_id=None,
            best_match_score=None,
            face_bbox=(0, 0, 160, 160),
            det_confidence=0.77,
            landmarks=_landmarks(0.0, 1.2),  # frontal
        )
        manager._process_task(task)
        kwargs = repo.add_unidentified_face.call_args.kwargs
        assert kwargs["det_confidence"] == pytest.approx(0.77)
        assert kwargs["frontalness"] == pytest.approx(1.0)
