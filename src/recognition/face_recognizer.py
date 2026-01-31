"""Face detection and recognition using InsightFace."""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.config import FaceRecognitionConfig

logger = logging.getLogger(__name__)


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
    """Face detector and recognizer using InsightFace."""

    def __init__(self, config: FaceRecognitionConfig):
        """Initialize face recognizer.

        Args:
            config: Face recognition configuration
        """
        self.config = config
        self._app = None
        self._initialized = False

    def _initialize(self):
        """Initialize InsightFace model (lazy loading)."""
        if self._initialized:
            return

        try:
            import insightface
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
            )
            self._app.prepare(ctx_id=0, det_size=(640, 640))

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

        faces_data = self._app.get(frame)

        # Use provided min_face_size or fall back to config
        effective_min_size = min_face_size if min_face_size is not None else self.config.min_face_size

        faces = []
        for face_data in faces_data:
            bbox = face_data.bbox.astype(float)

            # Check minimum face size
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if min(width, height) < effective_min_size:
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

        if face is not None and face.embedding is not None:
            return face.embedding

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
        # Dummy frame (1080p)
        dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Warmup detection
        self.detect_faces(dummy_frame)
        
        # Warmup extraction (needs a crop that looks like a face-ish)
        dummy_face_crop = np.zeros((200, 200, 3), dtype=np.uint8)
        self.extract_embedding(dummy_face_crop)
        
        logger.info("Face recognizer warmed up (detection and extraction)")
