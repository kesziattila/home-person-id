"""Shared helpers that replaced duplicated implementations.

- `DetectionResult.excluding_zones()`: exclusion-zone filtering used by both the
  CLI preview and the production pipeline (production previously ignored
  `cameras[].exclusion_zones` entirely — only the CLI preview applied them).
- `src.utils.embedding_utils`: one cosine-similarity implementation for face
  recognition, Re-ID and unidentified-face dedup.
- `create_person_detector(DetectionConfig)`: one place mapping config → backend.
"""

from unittest.mock import patch

import numpy as np
import pytest

from src.config import DetectionConfig, ExclusionZone
from src.detection.person_detector import Detection, DetectionResult, create_person_detector
from src.utils.embedding_utils import cosine_similarity, cosine_similarity_batch, l2_normalize

FRAME_SHAPE = (1000, 2000, 3)


def _result(*bboxes):
    return DetectionResult(
        detections=[Detection(bbox=b, confidence=0.9) for b in bboxes],
        frame_shape=FRAME_SHAPE,
    )


def test_excluding_zones_drops_detections_centered_in_zone():
    # Zone covers the right 8% of the frame (e.g. a TV)
    zones = [ExclusionZone(x1=0.92, y1=0.0, x2=1.0, y2=0.6)]
    inside = (1880.0, 100.0, 1960.0, 400.0)   # center (0.96, 0.25)
    outside = (200.0, 100.0, 400.0, 900.0)    # center (0.15, 0.5)
    below = (1880.0, 700.0, 1960.0, 900.0)    # center (0.96, 0.8) — below zone

    result = _result(inside, outside, below).excluding_zones(zones)

    assert [d.bbox for d in result.detections] == [outside, below]
    assert result.frame_shape == FRAME_SHAPE


@pytest.mark.parametrize("zones", [None, []])
def test_excluding_zones_without_zones_keeps_everything(zones):
    original = _result((0.0, 0.0, 10.0, 10.0))
    assert original.excluding_zones(zones) is original


def test_cosine_similarity_matches_manual_formula():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=512), rng.normal(size=512)
    expected = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    assert cosine_similarity(a, b) == pytest.approx(expected, abs=1e-6)
    assert cosine_similarity(a, 3.0 * a) == pytest.approx(1.0, abs=1e-6)


def test_cosine_similarity_batch_matches_pairwise():
    rng = np.random.default_rng(1)
    query = rng.normal(size=512)
    gallery = rng.normal(size=(5, 512))
    batch = cosine_similarity_batch(query, gallery)
    assert batch.shape == (5,)
    np.testing.assert_allclose(batch, [cosine_similarity(query, g) for g in gallery], atol=1e-6)


def test_l2_normalize_handles_vectors_rows_and_zero():
    np.testing.assert_allclose(np.linalg.norm(l2_normalize(np.array([3.0, 4.0]))), 1.0)
    rows = l2_normalize(np.array([[3.0, 4.0], [0.0, 2.0]]))
    np.testing.assert_allclose(np.linalg.norm(rows, axis=1), [1.0, 1.0])
    assert np.all(np.isfinite(l2_normalize(np.zeros(4))))


def test_create_person_detector_picks_backend_from_config():
    with patch("src.detection.person_detector.TensorRTDetector") as trt, \
         patch("src.detection.person_detector.PersonDetector") as pt:
        create_person_detector(DetectionConfig(
            model="m.engine", use_tensorrt_native=True, confidence_threshold=0.15,
        ))
        trt.assert_called_once_with(model_path="m.engine", confidence_threshold=0.15, nms_iou_threshold=0.4)
        pt.assert_not_called()

        create_person_detector(DetectionConfig(
            model="m.pt", confidence_threshold=0.2, nms_iou_threshold=0.45,
            device="cpu", num_threads=2,
        ))
        pt.assert_called_once_with(
            model_path="m.pt", confidence_threshold=0.2, nms_iou_threshold=0.45,
            device="cpu", num_threads=2,
        )
