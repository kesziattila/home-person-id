"""Person Re-Identification using OSNet.

OSNet (Omni-Scale Network) extracts appearance features from person crops
for cross-camera matching when faces aren't visible.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from src.config import ReIDConfig

logger = logging.getLogger(__name__)


def is_grayscale_image(image: np.ndarray, saturation_threshold: float = 15.0) -> bool:
    """Check if an image is grayscale/IR (black and white).

    IR cameras typically output images with very low color saturation.
    This function detects such images to avoid using them for Re-ID,
    as appearance features are unreliable without color information.

    Args:
        image: BGR image
        saturation_threshold: Mean saturation below this is considered grayscale

    Returns:
        True if image appears to be grayscale/IR
    """
    if image is None or image.size == 0:
        return True

    # Convert to HSV and check saturation channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    mean_saturation = np.mean(saturation)

    # Low saturation means grayscale/IR image
    return mean_saturation < saturation_threshold


@dataclass
class ReIDEmbedding:
    """Re-ID embedding with metadata."""

    embedding: np.ndarray  # 512-dim feature vector
    quality_score: float  # Quality of the crop (0-1)
    timestamp: float = 0.0


@dataclass
class EmbeddingGallery:
    """Gallery of embeddings for stable matching.

    Stores multiple embeddings per track and uses median similarity
    for more stable matching than single-embedding comparison.
    """

    embeddings: list[ReIDEmbedding] = field(default_factory=list)
    max_size: int = 10

    def add(self, embedding: np.ndarray, quality_score: float, timestamp: float = 0.0):
        """Add embedding to gallery if quality is sufficient.

        Args:
            embedding: 512-dim feature vector
            quality_score: Quality of the crop (0-1)
            timestamp: Time when embedding was extracted
        """
        if quality_score < 0.3:  # Skip low quality
            return

        self.embeddings.append(
            ReIDEmbedding(
                embedding=embedding,
                quality_score=quality_score,
                timestamp=timestamp,
            )
        )

        # Keep only the best quality embeddings if over limit
        if len(self.embeddings) > self.max_size:
            self.embeddings.sort(key=lambda x: x.quality_score, reverse=True)
            self.embeddings = self.embeddings[: self.max_size]

    def get_average_embedding(self) -> Optional[np.ndarray]:
        """Get average embedding from gallery."""
        if not self.embeddings:
            return None
        return np.mean([e.embedding for e in self.embeddings], axis=0)

    def match(self, query_embedding: np.ndarray) -> float:
        """Match query against gallery using median similarity.

        Args:
            query_embedding: 512-dim feature vector to match

        Returns:
            Median cosine similarity (more stable than single comparison)
        """
        if not self.embeddings:
            return 0.0

        similarities = []
        for emb in self.embeddings:
            sim = cosine_similarity(query_embedding, emb.embedding)
            similarities.append(sim)

        return float(np.median(similarities))

    def __len__(self) -> int:
        return len(self.embeddings)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_norm, b_norm))


class ReIDExtractor:
    """Re-ID feature extractor using OSNet.

    Extracts 512-dimensional appearance features from person crops
    for cross-camera person matching.
    """

    def __init__(self, config: ReIDConfig):
        """Initialize Re-ID extractor.

        Args:
            config: Re-ID configuration
        """
        self.config = config
        self._model = None
        self._transform = None
        self._initialized = False
        self._device = None
        self._is_ort = False

    def _initialize(self):
        """Initialize model (lazy loading)."""
        if self._initialized:
            return

        try:
            import torch
            import torchvision.transforms as T
            from pathlib import Path

            # Determine device
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Re-ID using device: {self._device}")

            model_path = self.config.model

            # Check for TensorRT engine or ONNX model
            if model_path.endswith(".engine") or model_path.endswith(".onnx"):
                self._load_tensorrt_model(model_path)
            # Check if model is a file path
            elif model_path.endswith(".pth") or "/" in model_path:
                # Load from local file
                model_file = Path(model_path)
                if not model_file.exists():
                    raise FileNotFoundError(f"Re-ID model not found: {model_path}")

                logger.info(f"Loading Re-ID model from file: {model_path}")

                # Determine model architecture from filename
                from torchreid import models

                if "osnet_ain_x1_0" in model_path:
                    arch = "osnet_ain_x1_0"
                elif "osnet_x1_0" in model_path:
                    arch = "osnet_x1_0"
                elif "osnet_x0_75" in model_path:
                    arch = "osnet_x0_75"
                elif "osnet_x0_5" in model_path:
                    arch = "osnet_x0_5"
                elif "osnet_x0_25" in model_path:
                    arch = "osnet_x0_25"
                else:
                    arch = "osnet_x1_0"  # Default

                logger.info(f"Using architecture: {arch}")

                # Build model without pretrained weights
                self._model = models.build_model(
                    name=arch,
                    num_classes=1,  # Not used for feature extraction
                    loss="softmax",
                    pretrained=False,
                )

                # Load weights from file
                state_dict = torch.load(model_path, map_location=self._device)
                # Handle different checkpoint formats
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                # Remove 'module.' prefix if present (from DataParallel)
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                # Remove classifier weights (we only need feature extractor)
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier")}
                # Load weights
                self._model.load_state_dict(state_dict, strict=False)
                self._model = self._model.to(self._device)
                self._model.eval()

            else:
                # Try to load from torchreid by name
                try:
                    from torchreid import models

                    logger.info(f"Loading Re-ID model: {model_path}")

                    self._model = models.build_model(
                        name=model_path,
                        num_classes=1,  # Not used for feature extraction
                        loss="softmax",
                        pretrained=True,
                    )
                    self._model = self._model.to(self._device)
                    self._model.eval()

                except ImportError:
                    # Fallback to loading OSNet directly with torch hub
                    logger.warning("torchreid not available, using torch hub")
                    self._model = torch.hub.load(
                        "KaiyangZhou/deep-person-reid",
                        "osnet_x1_0",
                        pretrained=True,
                    )
                    self._model = self._model.to(self._device)
                    self._model.eval()

            # Standard Re-ID preprocessing
            self._transform = T.Compose([
                T.ToPILImage(),
                T.Resize((256, 128)),  # Standard Re-ID input size
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            self._initialized = True
            logger.info("Re-ID model loaded successfully")

        except ImportError as e:
            raise ImportError(
                f"PyTorch and torchreid are required for Re-ID. "
                f"Install with: pip install torch torchvision torchreid. Error: {e}"
            )

    def extract(
        self, crop: np.ndarray, return_quality: bool = False
    ) -> np.ndarray | tuple[np.ndarray, float]:
        """Extract Re-ID features from a person crop.

        Args:
            crop: BGR person crop image
            return_quality: Whether to also return quality score

        Returns:
            512-dim feature vector, optionally with quality score
        """
        self._initialize()

        import torch

        # Check minimum crop size
        h, w = crop.shape[:2]
        if h < self.config.min_crop_height or w < 30:
            if return_quality:
                return np.zeros(512, dtype=np.float32), 0.0
            return np.zeros(512, dtype=np.float32)

        # Skip grayscale/IR images - Re-ID relies on color features
        if is_grayscale_image(crop):
            logger.debug("Skipping Re-ID extraction for grayscale/IR image")
            if return_quality:
                return np.zeros(512, dtype=np.float32), 0.0
            return np.zeros(512, dtype=np.float32)

        # Compute quality score based on crop size and aspect ratio
        quality = self._compute_quality(crop)

        # Convert BGR to RGB
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # Transform and add batch dimension
        tensor = self._transform(rgb).unsqueeze(0)

        # Extract features
        if self._is_ort:
            input_name = self._model.get_inputs()[0].name
            ort_inputs = {input_name: tensor.numpy()}
            features_np = self._model.run(None, ort_inputs)[0]
            # Normalize features
            norm = np.linalg.norm(features_np, ord=2, axis=1, keepdims=True)
            embedding = (features_np / norm).flatten()
        else:
            tensor = tensor.to(self._device)
            import torch
            with torch.no_grad():
                features = self._model(tensor)

                # Handle different model output formats
                if isinstance(features, tuple):
                    features = features[0]

                # Normalize features
                features = torch.nn.functional.normalize(features, p=2, dim=1)

                embedding = features.cpu().numpy().flatten()

        if return_quality:
            return embedding, quality

        return embedding

    def _load_tensorrt_model(self, model_path: str):
        """Load TensorRT engine or ONNX model using onnxruntime."""
        import onnxruntime as ort
        from pathlib import Path

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # If it's a raw .engine file, we can't use onnxruntime InferenceSession directly.
        # We need the .onnx file to initialize the session.
        if model_path.endswith(".engine"):
            onnx_path = model_path.replace(".engine", ".onnx")
            if Path(onnx_path).exists():
                logger.info(f"Found matching ONNX model for engine: {onnx_path}")
                model_path = onnx_path
            else:
                logger.warning(f"Raw .engine file provided but no matching .onnx found at {onnx_path}")
                logger.warning("onnruntime cannot load raw .engine files directly. "
                               "Please point to the .onnx file instead.")

        logger.info(f"Loading Re-ID model for TensorRT: {model_path}")

        # Check for TensorrtExecutionProvider
        available_providers = ort.get_available_providers()
        if "TensorrtExecutionProvider" not in available_providers:
            logger.warning("TensorrtExecutionProvider not available in onnxruntime, "
                           "falling back to CUDA or CPU. Performance will be degraded.")

        providers = [
            ("TensorrtExecutionProvider", {
                "device_id": 0,
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "data/cache/trt_cache",
            }),
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]

        # Filter providers to only those available
        providers = [p for p in providers if (p[0] if isinstance(p, tuple) else p) in available_providers]

        try:
            self._model = ort.InferenceSession(model_path, providers=providers)
            self._is_ort = True
            logger.info(f"Re-ID model loaded successfully using {self._model.get_providers()[0]}")
        except Exception as e:
            if "INVALID_PROTOBUF" in str(e) and model_path.endswith(".engine"):
                logger.error("Failed to load .engine file directly. onnxruntime requires the .onnx file "
                             "to use the TensorRT execution provider.")
            raise e

    def extract_batch(
        self, crops: list[np.ndarray]
    ) -> list[tuple[np.ndarray, float]]:
        """Extract Re-ID features from multiple crops (batch processing).

        Args:
            crops: List of BGR person crop images

        Returns:
            List of (embedding, quality_score) tuples
        """
        if not crops:
            return []

        self._initialize()

        import cv2
        import torch

        results = []
        valid_indices = []
        tensors = []

        # Preprocess all crops
        for i, crop in enumerate(crops):
            h, w = crop.shape[:2]
            if h < self.config.min_crop_height or w < 30:
                results.append((np.zeros(512, dtype=np.float32), 0.0))
                continue

            # Skip grayscale/IR images
            if is_grayscale_image(crop):
                results.append((np.zeros(512, dtype=np.float32), 0.0))
                continue

            quality = self._compute_quality(crop)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self._transform(rgb)
            tensors.append(tensor)
            valid_indices.append((i, quality))

        if not tensors:
            return results

        # Batch inference
        batch = torch.stack(tensors)

        if self._is_ort:
            input_name = self._model.get_inputs()[0].name
            ort_inputs = {input_name: batch.numpy()}
            features_np = self._model.run(None, ort_inputs)[0]
            # Normalize features
            norm = np.linalg.norm(features_np, ord=2, axis=1, keepdims=True)
            embeddings = features_np / norm
        else:
            batch = batch.to(self._device)
            with torch.no_grad():
                features = self._model(batch)

                if isinstance(features, tuple):
                    features = features[0]

                features = torch.nn.functional.normalize(features, p=2, dim=1)
                embeddings = features.cpu().numpy()

        # Rebuild results list
        result_dict = {}
        for idx, (orig_idx, quality) in enumerate(valid_indices):
            result_dict[orig_idx] = (embeddings[idx], quality)

        final_results = []
        for i in range(len(crops)):
            if i in result_dict:
                final_results.append(result_dict[i])
            else:
                final_results.append((np.zeros(512, dtype=np.float32), 0.0))

        return final_results

    def _compute_quality(self, crop: np.ndarray) -> float:
        """Compute quality score for a crop.

        Quality is based on:
        - Size (larger is better)
        - Aspect ratio (close to 2:1 height:width is ideal for person)
        - Not too blurry

        Args:
            crop: Person crop image

        Returns:
            Quality score between 0 and 1
        """
        import cv2

        h, w = crop.shape[:2]

        # Size score (normalized by expected good size)
        size_score = min(1.0, (h * w) / (256 * 128))

        # Aspect ratio score (ideal is ~2:1 for standing person)
        aspect = h / (w + 1e-6)
        ideal_aspect = 2.0
        aspect_score = 1.0 - min(1.0, abs(aspect - ideal_aspect) / ideal_aspect)

        # Blur score using Laplacian variance
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(1.0, laplacian_var / 500.0)  # 500 is empirically good

        # Combine scores
        quality = 0.4 * size_score + 0.3 * aspect_score + 0.3 * blur_score

        return float(quality)

    def compare(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compare two Re-ID embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity (0-1, higher is more similar)
        """
        return cosine_similarity(embedding1, embedding2)

    def warmup(self):
        """Warm up the model with a dummy inference."""
        self._initialize()
        dummy = np.zeros((256, 128, 3), dtype=np.uint8)
        self.extract(dummy)
        logger.info("Re-ID extractor warmed up")
