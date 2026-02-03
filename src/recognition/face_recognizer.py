"""Face detection and recognition using InsightFace or TensorRT native backend."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import FaceRecognitionConfig

logger = logging.getLogger(__name__)


# Lazy import for TensorRT backend
_tensorrt_face_module = None


def _get_tensorrt_face_module():
    """Lazy import TensorRT face module."""
    global _tensorrt_face_module
    if _tensorrt_face_module is None:
        from src.recognition import tensorrt_face
        _tensorrt_face_module = tensorrt_face
    return _tensorrt_face_module


@dataclass
class Face:
    """Detected face information."""

    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    landmarks: Optional[np.ndarray] = None  # 5 facial landmarks
    embedding: Optional[np.ndarray] = None  # 512-dim embedding

    @property
    def width(self) -> float:
        """Width of face bounding box."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        """Height of face bounding box."""
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        """Area of face bounding box."""
        return self.width * self.height


@dataclass
class FaceDetectionResult:
    """Result of face detection."""

    faces: list[Face]
    frame_shape: tuple[int, int, int]

    @property
    def count(self) -> int:
        """Number of faces detected."""
        return len(self.faces)


class FaceRecognizer:
    """Face detector and recognizer using InsightFace or TensorRT native backend."""

    def __init__(self, config: FaceRecognitionConfig):
        """Initialize face recognizer.

        Args:
            config: Face recognition configuration
        """
        self.config = config
        self._app = None  # InsightFace backend
        self._trt_recognizer = None  # TensorRT native backend
        self._initialized = False
        self._use_tensorrt_native = config.use_tensorrt_native

    def _initialize(self):
        """Initialize face recognition model (lazy loading)."""
        if self._initialized:
            return

        if self._use_tensorrt_native:
            self._initialize_tensorrt_native()
        else:
            self._initialize_insightface()

    def _initialize_tensorrt_native(self):
        """Initialize TensorRT native backend."""
        det_model = self.config.trt_det_model
        rec_model = self.config.trt_rec_model

        if not Path(det_model).exists():
            raise FileNotFoundError(
                f"TensorRT detection model not found: {det_model}\n"
                f"Convert with: python tools/convert_insightface_to_trt.py"
            )
        if not Path(rec_model).exists():
            raise FileNotFoundError(
                f"TensorRT recognition model not found: {rec_model}\n"
                f"Convert with: python tools/convert_insightface_to_trt.py"
            )

        logger.info(f"Loading TensorRT native face recognition")
        logger.info(f"  Detection model: {det_model}")
        logger.info(f"  Recognition model: {rec_model}")

        trt_module = _get_tensorrt_face_module()
        self._trt_recognizer = trt_module.TensorRTFaceRecognizer(
            det_model_path=det_model,
            rec_model_path=rec_model,
            confidence_threshold=0.5,
            nms_threshold=0.4,
            det_size=self.config.det_size,
            min_face_size=self.config.min_face_size,
        )

        self._initialized = True
        logger.info("TensorRT native face recognition loaded")

    def _initialize_insightface(self):
        """Initialize InsightFace backend."""
        try:
            from insightface.app import FaceAnalysis

            logger.info(f"Loading InsightFace model: {self.config.model}")

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

            # Add TensorRT if enabled and available
            if self.config.use_tensorrt:
                try:
                    import onnxruntime as ort
                    if "TensorrtExecutionProvider" in ort.get_available_providers():
                        logger.info("Using TensorRT acceleration for face recognition")
                        trt_options = {
                            "device_id": 0,
                            "trt_fp16_enable": True,
                            "trt_engine_cache_enable": True,
                            "trt_engine_cache_path": "data/cache/trt_cache",
                        }
                        if self.config.trt_max_workspace_size > 0:
                            trt_options["trt_max_workspace_size"] = self.config.trt_max_workspace_size

                        providers.insert(0, ("TensorrtExecutionProvider", trt_options))
                    else:
                        logger.warning("TensorrtExecutionProvider not available for face recognition")
                except ImportError:
                    logger.warning("onnxruntime not available to check for TensorRT")

            self._app = FaceAnalysis(
                name=self.config.model,
                providers=providers,
                # Only load detection + recognition, skip genderage/landmarks/3d
                allowed_modules=["detection", "recognition"],
            )
            det_size = self.config.det_size
            self._app.prepare(ctx_id=0, det_size=(det_size, det_size))
            logger.info(f"InsightFace det_size: {det_size}x{det_size}")

            self._initialized = True
            logger.info("InsightFace model loaded successfully")

        except ImportError:
            raise ImportError(
                "insightface is required for face recognition. "
                "Install with: pip install insightface"
            )

    def detect_faces(
        self,
        frame: np.ndarray,
        min_face_size: Optional[int] = None,
    ) -> FaceDetectionResult:
        """Detect faces in a frame.

        Args:
            frame: BGR image
            min_face_size: Optional override for minimum face size filter.
                          If None, uses config.min_face_size.

        Returns:
            FaceDetectionResult with detected faces
        """
        self._initialize()

        # Use provided min_face_size or fall back to config
        effective_min_size = min_face_size if min_face_size is not None else self.config.min_face_size

        if self._use_tensorrt_native:
            return self._detect_faces_tensorrt(frame, effective_min_size)
        else:
            return self._detect_faces_insightface(frame, effective_min_size)

    def _detect_faces_tensorrt(
        self, frame: np.ndarray, min_face_size: int
    ) -> FaceDetectionResult:
        """Detect faces using TensorRT native backend."""
        trt_faces = self._trt_recognizer.detect_and_embed(frame)

        faces = []
        for trt_face in trt_faces:
            # Check minimum face size
            width = trt_face.bbox[2] - trt_face.bbox[0]
            height = trt_face.bbox[3] - trt_face.bbox[1]
            if min(width, height) < min_face_size:
                continue

            face = Face(
                bbox=trt_face.bbox,
                confidence=trt_face.confidence,
                landmarks=trt_face.landmarks,
                embedding=trt_face.embedding,
            )
            faces.append(face)

        return FaceDetectionResult(faces=faces, frame_shape=frame.shape)

    def _detect_faces_insightface(
        self, frame: np.ndarray, min_face_size: int
    ) -> FaceDetectionResult:
        """Detect faces using InsightFace backend."""
        faces_data = self._app.get(frame)

        faces = []
        for face_data in faces_data:
            bbox = face_data.bbox.astype(float)

            # Check minimum face size
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if min(width, height) < min_face_size:
                continue

            face = Face(
                bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                confidence=float(face_data.det_score),
                landmarks=face_data.kps if hasattr(face_data, "kps") else None,
                embedding=face_data.embedding if hasattr(face_data, "embedding") else None,
            )
            faces.append(face)

        return FaceDetectionResult(faces=faces, frame_shape=frame.shape)

    def extract_embedding(
        self, frame: np.ndarray, face: Optional[Face] = None
    ) -> Optional[np.ndarray]:
        """Extract face embedding.

        Args:
            frame: BGR image
            face: Optional detected face (if None, will detect)

        Returns:
            512-dim face embedding or None if no face found
        """
        self._initialize()

        # If face already has embedding, return it
        if face is not None and face.embedding is not None:
            return face.embedding

        if self._use_tensorrt_native:
            return self._extract_embedding_tensorrt(frame, face)
        else:
            return self._extract_embedding_insightface(frame, face)

    def _extract_embedding_tensorrt(
        self, frame: np.ndarray, face: Optional[Face] = None
    ) -> Optional[np.ndarray]:
        """Extract embedding using TensorRT native backend."""
        if face is not None and face.landmarks is not None:
            # Align and extract using provided landmarks
            trt_module = _get_tensorrt_face_module()
            aligned = trt_module.align_face(frame, face.landmarks)
            return self._trt_recognizer._embedder.extract(aligned)

        # Detect and get embedding
        trt_faces = self._trt_recognizer.detect_and_embed(frame)
        if not trt_faces:
            return None
        return trt_faces[0].embedding

    def _extract_embedding_insightface(
        self, frame: np.ndarray, face: Optional[Face] = None
    ) -> Optional[np.ndarray]:
        """Extract embedding using InsightFace backend."""
        # Detect faces and get embedding
        faces_data = self._app.get(frame)

        if not faces_data:
            return None

        # Return embedding of first (largest) face
        return faces_data[0].embedding

    def compare_embeddings(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Compare two face embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity score (0-1, higher is more similar)
        """
        # Normalize embeddings
        e1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        e2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)

        # Cosine similarity
        similarity = np.dot(e1, e2)

        return float(similarity)

    def compare_embeddings_batch(
        self, query_embedding: np.ndarray, gallery_embeddings: np.ndarray
    ) -> np.ndarray:
        """Compare query embedding against multiple gallery embeddings (vectorized).

        Args:
            query_embedding: Query embedding (512-dim)
            gallery_embeddings: Gallery embeddings (N, 512)

        Returns:
            Array of cosine similarity scores
        """
        # Normalize query once
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        # Normalize gallery
        gallery_norms = gallery_embeddings / (
            np.linalg.norm(gallery_embeddings, axis=1, keepdims=True) + 1e-8
        )
        # Vectorized dot product
        return gallery_norms @ query_norm

    def find_best_match(
        self,
        query_embedding: np.ndarray,
        gallery: list[tuple[int, np.ndarray]],
    ) -> Optional[tuple[int, float]]:
        """Find best matching person from gallery.

        Args:
            query_embedding: Face embedding to match
            gallery: List of (person_id, embedding) tuples

        Returns:
            Tuple of (person_id, similarity) or None if no match above threshold
        """
        if not gallery:
            return None

        best_match = None
        best_score = 0.0

        for person_id, embedding in gallery:
            similarity = self.compare_embeddings(query_embedding, embedding)

            if similarity > self.config.similarity_threshold and similarity > best_score:
                best_match = person_id
                best_score = similarity

        if best_match is not None:
            return (best_match, best_score)

        return None

    def warmup(self) -> None:
        """Warm up the model with dummy inferences.

        Performs both detection and extraction to ensure all components are ready.
        """
        if not self.config.enabled:
            return

        self._initialize()

        if self._use_tensorrt_native:
            self._trt_recognizer.warmup()
            logger.info("TensorRT native face recognizer warmed up")
        else:
            # Dummy frame (1080p)
            dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

            # Warmup detection
            self.detect_faces(dummy_frame)

            # Warmup extraction (needs a crop that looks like a face-ish)
            dummy_face_crop = np.zeros((200, 200, 3), dtype=np.uint8)
            self.extract_embedding(dummy_face_crop)

            logger.info("InsightFace recognizer warmed up (detection and extraction)")
