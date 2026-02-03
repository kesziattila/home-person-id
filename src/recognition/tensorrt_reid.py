"""TensorRT-native Re-ID embedding extraction.

Provides lower memory usage than ONNX Runtime by using TensorRT directly.
Requires TensorRT engine converted from OSNet ONNX model.

Convert with:
    trtexec --onnx=osnet.onnx --saveEngine=osnet.engine --fp16
"""

import logging
from typing import Optional

import cv2
import numpy as np

from src.inference.tensorrt_base import TensorRTEmbedderBase

logger = logging.getLogger(__name__)

# ImageNet normalization constants (used by OSNet/torchreid)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class TensorRTReIDEmbedder(TensorRTEmbedderBase):
    """Re-ID embedding extractor using TensorRT (OSNet).

    Takes person crop images and outputs 512-dim embeddings.
    """

    def __init__(
        self,
        model_path: str,
        input_height: int = 256,
        input_width: int = 128,
    ):
        """Initialize TensorRT Re-ID embedder.

        Args:
            model_path: Path to TensorRT engine (.engine or .trt)
            input_height: Input height (default 256)
            input_width: Input width (default 128)
        """
        super().__init__(model_path, input_size=input_height)
        self.input_height = input_height
        self.input_width = input_width

    def _get_input_size(self) -> int:
        """Return height for base class compatibility."""
        return self.input_height

    def _get_input_height(self) -> int:
        """Override for non-square input."""
        return self.input_height

    def _get_input_width(self) -> int:
        """Override for non-square input."""
        return self.input_width

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Preprocess person crop for OSNet.

        Args:
            crop: BGR person crop image

        Returns:
            Preprocessed tensor (1, 3, 256, 128)
        """
        # Resize to input size
        resized = cv2.resize(crop, (self.input_width, self.input_height))

        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Convert to float32 and normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # Apply ImageNet normalization: (x - mean) / std
        normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD

        # HWC to CHW
        chw = normalized.transpose(2, 0, 1)

        # Get target dtype from input buffer
        self._load_engine()
        input_host, _ = self._get_input_buffer()
        target_dtype = input_host.dtype

        return chw.astype(target_dtype)[np.newaxis, ...]

    def _compute_quality(self, crop: np.ndarray) -> float:
        """Compute quality score for a crop.

        Quality is based on:
        - Image sharpness (Laplacian variance)
        - Size relative to optimal input size

        Args:
            crop: BGR image

        Returns:
            Quality score in [0, 1]
        """
        h, w = crop.shape[:2]

        # Size score - optimal is close to input size
        size_ratio = min(h / self.input_height, w / self.input_width)
        size_score = min(1.0, size_ratio)

        # Sharpness score using Laplacian variance
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalize sharpness (100 is a reasonable threshold for sharp images)
        sharpness_score = min(1.0, laplacian_var / 100.0)

        # Combined score
        return 0.5 * size_score + 0.5 * sharpness_score

    def extract(
        self,
        crop: np.ndarray,
        return_quality: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, float]:
        """Extract Re-ID embedding from person crop.

        Args:
            crop: BGR person crop image
            return_quality: Whether to return quality score

        Returns:
            512-dim normalized embedding, or (embedding, quality) tuple
        """
        self._load_engine()

        quality = self._compute_quality(crop) if return_quality else 0.0

        input_tensor = self._preprocess(crop)
        outputs = self._run_inference(input_tensor)

        if not outputs:
            embedding = np.zeros(512, dtype=np.float32)
        else:
            embedding = outputs[0].flatten()
            embedding = self._normalize_embedding(embedding)

        if return_quality:
            return embedding, quality
        return embedding

    def extract_batch(
        self,
        crops: list[np.ndarray],
    ) -> list[tuple[np.ndarray, float]]:
        """Extract Re-ID embeddings from multiple crops.

        Note: TensorRT native backend processes one at a time.
        For batch processing with TensorRT, use ONNX Runtime backend.

        Args:
            crops: List of BGR person crop images

        Returns:
            List of (embedding, quality) tuples
        """
        results = []
        for crop in crops:
            embedding, quality = self.extract(crop, return_quality=True)
            results.append((embedding, quality))
        return results

    def warmup(self):
        """Warm up the engine."""
        self._load_engine()
        dummy_crop = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
        self.extract(dummy_crop)
        logger.info("TensorRT Re-ID embedder warmed up")
