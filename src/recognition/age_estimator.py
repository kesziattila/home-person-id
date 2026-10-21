"""Apparent age of a person from a face crop and/or a body crop (MiVOLO v2).

MiVOLO (Kuprashevich & Tolstykh, 2023/2024) is a VOLO-D1 transformer that fuses a face
crop and a body crop; either may be missing, so it still estimates from the body when
the face is turned away, hidden by a hat or too small. Chosen 2026-10-19 because the
insightface genderage model put a 6-year-old at 28-31 years; on hand-labelled live
crops MiVOLO gave the child a median of 8 years and the adults 33-35 (TUNING_LOG).

Model: one 6-channel input (face RGB + body RGB, each letterboxed to 384x384, ImageNet
normalised), output [male, female, age_norm]; age = age_norm * 122 + 61.
- `.onnx`: ONNX Runtime on CPU (~0.5 s per sample on an x86 CPU) — dev / tests
- `.engine`: TensorRT (FP16 ~25 ms on the Jetson Orin under live load)
Export and conversion: tools/convert_mivolo_to_onnx.py, docs/JETSON.md.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.inference.tensorrt_base import TensorRTEngine
from src.utils.profiler import profiler

logger = logging.getLogger(__name__)

INPUT_SIZE = 384
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MIN_AGE, MAX_AGE, AVG_AGE = 0.0, 122.0, 61.0


def letterbox(image: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """Resize keeping the aspect ratio and pad with black to size x size (MiVOLO's class_letterbox)."""
    h, w = image.shape[:2]
    if (h, w) == (size, size):
        return image
    r = min(size / h, size / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    if (new_w, new_h) != (w, h):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def _channels(crop: Optional[np.ndarray]) -> np.ndarray:
    """One BGR crop -> normalised RGB CHW; a missing crop is a black (all-zero) image."""
    if crop is None or crop.size == 0:
        x = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
    else:
        x = cv2.cvtColor(letterbox(crop), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((x - MEAN) / STD).transpose(2, 0, 1)


def prepare_input(face: Optional[np.ndarray], body: Optional[np.ndarray]) -> np.ndarray:
    """MiVOLO input (1, 6, 384, 384) float32: face channels first, then body."""
    return np.concatenate([_channels(face), _channels(body)])[np.newaxis].astype(np.float32)


class _TensorRTAgeModel(TensorRTEngine):
    """MiVOLO TensorRT engine: input (1, 6, 384, 384), output (1, 3)."""

    def _get_input_size(self) -> int:
        return INPUT_SIZE

    def _validate_engine(self):
        outputs = [self._buffers[name][2] for name in self._output_names]
        if len(outputs) != 1 or tuple(outputs[0])[-1] != 3:
            raise RuntimeError(f"MiVOLO engine must have one (1, 3) output, got {outputs}")

    def run(self, x: np.ndarray) -> Optional[np.ndarray]:
        outputs = self._run_inference(x)
        return outputs[0].reshape(-1, 3) if outputs else None


class AgeEstimator:
    """MiVOLO age estimate; the backend (TensorRT / ONNX Runtime) is lazy-loaded on first use."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"age model not found: {self.model_path}")
            if self.model_path.endswith((".engine", ".trt")):
                self._model = _TensorRTAgeModel(self.model_path)
            else:
                import onnxruntime as ort
                self._model = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
            logger.info(f"Age model loaded: {self.model_path}")
        return self._model

    def estimate(self, face: Optional[np.ndarray], body: Optional[np.ndarray]) -> Optional[float]:
        """Apparent age in years from BGR crops (either may be None); None if inference failed."""
        if (face is None or face.size == 0) and (body is None or body.size == 0):
            return None
        with profiler.measure("AgeEstimator.estimate"):
            x = prepare_input(face, body)
            model = self.model
            if isinstance(model, _TensorRTAgeModel):
                out = model.run(x)
            else:
                out = model.run(None, {model.get_inputs()[0].name: x})[0]
            if out is None:
                return None
            return float(out[0, 2] * (MAX_AGE - MIN_AGE) + AVG_AGE)

    def shutdown(self):
        if isinstance(self._model, _TensorRTAgeModel):
            self._model.shutdown()
