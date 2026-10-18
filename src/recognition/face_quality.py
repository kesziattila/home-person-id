"""Face image quality: how usable a face is for recognition (eDifFIQA-tiny).

One learned score replaces the hand-made photo metrics (sharpness, noise,
frontalness, brightness, detector confidence) used until 2026-10-18. Those kept
needing new conditions and still failed on live crops: a faded, grainy dawn
crop and a clean gallery face measured the same on every pixel statistic
(noise sigma 1.5 vs 1.6, structure-to-noise 12.9 vs 12.2), and a face half
hidden by a phone scored sharpness 1.0 and frontalness 1.0 (landmarks are
guessed under occlusion). Pixel statistics describe the photo, not whether
the face in it yields a reliable embedding.

eDifFIQA(T) (Babnik et al., 2024; OpenCV model zoo
`face_image_quality_assessment_ediffiqa`) is a small network distilled to
predict exactly that from the aligned 112x112 face the recognizer also sees.
On 51 hand-labelled live crops (2026-10-18) it separated good from bad 84% at
a single threshold and correlated 0.75 with each face's similarity to its own
person's gallery (the hand-made score: 0.31). Observed range: phone-occluded
0.22-0.37, faded/grainy 0.11-0.45, usable 0.29-0.70 (low end = strongly
looking-down heads, which do embed worse), enrolled gallery 0.45-0.68.

Model file: models/fiqa/ediffiqa_tiny_jun2024.onnx (7 MB, gitignored), from
https://huggingface.co/opencv/face_image_quality_assessment_ediffiqa
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from src.utils.profiler import profiler

logger = logging.getLogger(__name__)


class FaceQualityAssessor:
    """eDifFIQA-tiny on CPU (ONNX Runtime), lazy-loaded on first use.

    Runs only where a face is stored for review, never per frame, so it stays off
    the GPU the live pipeline is bound by. ~6 ms per face on an x86 CPU, ~28 ms
    on the Jetson Orin Nano CPU under live load (single thread).
    """

    INPUT_SIZE = 112

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._session = None

    @property
    def session(self):
        if self._session is None:
            import onnxruntime as ort
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"face quality model not found: {self.model_path}")
            # One thread: faster on the loaded Jetson (28 vs 40 ms) and leaves
            # the other cores to the live pipeline
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            self._session = ort.InferenceSession(self.model_path, options, providers=["CPUExecutionProvider"])
            logger.info(f"Face quality model loaded: {self.model_path}")
        return self._session

    def score(self, aligned_face: np.ndarray) -> float:
        """Quality of an ArcFace-aligned BGR face (112x112, `align_face`); higher = more usable."""
        with profiler.measure("FaceQualityAssessor.score"):
            face = aligned_face
            if face.shape[:2] != (self.INPUT_SIZE, self.INPUT_SIZE):
                face = cv2.resize(face, (self.INPUT_SIZE, self.INPUT_SIZE))
            x = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            x = ((x - 0.5) / 0.5).transpose(2, 0, 1)[np.newaxis]
            return float(self.session.run(None, {"input": x})[0][0, 0])

