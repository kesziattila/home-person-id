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

            self._app = FaceAnalysis(
                name=self.config.model,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=0, det_size=(640, 640))

            self._initialized = True
            logger.info("InsightFace model loaded successfully")

        except ImportError:
            raise ImportError(
                "insightface is required for face recognition. "
                "Install with: pip install insightface"
            )

    def detect_faces(self, frame: np.ndarray) -> FaceDetectionResult:
        """Detect faces in a frame.

        Args:
            frame: BGR image

        Returns:
            FaceDetectionResult with detected faces
        """
        self._initialize()

        faces_data = self._app.get(frame)

        faces = []
        for face_data in faces_data:
            bbox = face_data.bbox.astype(float)

            # Check minimum face size
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if min(width, height) < self.config.min_face_size:
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
        e1 = embedding1 / np.linalg.norm(embedding1)
        e2 = embedding2 / np.linalg.norm(embedding2)

        # Cosine similarity
        similarity = np.dot(e1, e2)

        return float(similarity)

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
