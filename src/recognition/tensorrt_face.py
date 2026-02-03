"""TensorRT-native face detection and recognition.

Provides lower memory usage than ONNX Runtime by using TensorRT directly.
Requires TensorRT engines converted from InsightFace ONNX models.

Models needed:
- det_10g.engine: SCRFD detection (from det_10g.onnx)
- w600k_r50.engine: ArcFace recognition (from w600k_r50.onnx)

Convert with:
    python tools/convert_insightface_to_trt.py --det-size 640
"""

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.inference.tensorrt_base import TensorRTDetectorBase, TensorRTEmbedderBase

logger = logging.getLogger(__name__)


@dataclass
class TensorRTFace:
    """Detected face with landmarks and embedding."""
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    landmarks: np.ndarray  # (5, 2) facial landmarks
    embedding: Optional[np.ndarray] = None  # 512-dim embedding


# Standard face alignment destination points for 112x112 output
ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_face(frame: np.ndarray, landmarks: np.ndarray, output_size: int = 112) -> np.ndarray:
    """Align face using 5 landmarks.

    Args:
        frame: BGR image
        landmarks: 5 facial landmarks (5, 2)
        output_size: Output size (default 112 for ArcFace)

    Returns:
        Aligned face image (output_size, output_size, 3)
    """
    dst = ARCFACE_DST.copy()
    if output_size != 112:
        dst = dst * (output_size / 112.0)

    # Estimate similarity transform
    M, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), dst)

    # Apply transformation
    aligned = cv2.warpAffine(frame, M, (output_size, output_size), borderValue=0)
    return aligned


class TensorRTFaceEmbedder(TensorRTEmbedderBase):
    """Face embedding extractor using TensorRT (ArcFace).

    Takes aligned 112x112 face images and outputs 512-dim embeddings.
    """

    def __init__(self, model_path: str, input_size: int = 112):
        """Initialize TensorRT face embedder.

        Args:
            model_path: Path to TensorRT engine (.engine or .trt)
            input_size: Input face size (default 112)
        """
        super().__init__(model_path, input_size)

    def _preprocess(self, aligned_face: np.ndarray) -> np.ndarray:
        """Preprocess aligned face for ArcFace.

        Args:
            aligned_face: BGR aligned face (112, 112, 3)

        Returns:
            Preprocessed tensor (1, 3, 112, 112)
        """
        # Resize if needed
        if aligned_face.shape[0] != self.input_size or aligned_face.shape[1] != self.input_size:
            aligned_face = cv2.resize(aligned_face, (self.input_size, self.input_size))

        # BGR to RGB, HWC to CHW
        rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2, 0, 1)

        # Get target dtype from input buffer
        input_host, _ = self._get_input_buffer()
        target_dtype = input_host.dtype

        # Normalize to [-1, 1] (ArcFace normalization)
        normalized = (chw.astype(np.float32) - 127.5) / 127.5
        normalized = normalized.astype(target_dtype)

        return normalized[np.newaxis, ...]

    def extract(self, aligned_face: np.ndarray) -> np.ndarray:
        """Extract face embedding from aligned face.

        Args:
            aligned_face: BGR aligned face (112, 112, 3)

        Returns:
            512-dim normalized embedding
        """
        self._load_engine()

        input_tensor = self._preprocess(aligned_face)
        outputs = self._run_inference(input_tensor)

        if not outputs:
            return np.zeros(512, dtype=np.float32)

        embedding = outputs[0].flatten()
        return self._normalize_embedding(embedding)

    def warmup(self):
        """Warm up the engine."""
        self._load_engine()
        dummy_face = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        self.extract(dummy_face)
        logger.info("TensorRT face embedder warmed up")


class TensorRTFaceDetector(TensorRTDetectorBase):
    """Face detector using TensorRT (SCRFD/RetinaFace).

    Detects faces and returns bounding boxes with 5-point landmarks.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: int = 640,
    ):
        """Initialize TensorRT face detector.

        Args:
            model_path: Path to TensorRT engine (.engine or .trt)
            confidence_threshold: Minimum confidence for detections
            nms_threshold: NMS IoU threshold
            input_size: Input size for detection (640, 480, or 320)
        """
        super().__init__(model_path, confidence_threshold, nms_threshold, input_size)

        # SCRFD uses these strides
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self._anchors_cache = {}

    def _validate_engine(self):
        """Validate SCRFD output format (9 tensors: 3 scores, 3 bboxes, 3 kps)."""
        num_outputs = len(self._output_names)
        if num_outputs != 9:
            raise RuntimeError(
                f"Expected 9 output tensors for SCRFD, got {num_outputs}.\n"
                f"Make sure you're using the correct detection model (det_10g.onnx)."
            )
        logger.debug(f"SCRFD output tensors: {self._output_names}")

    def _generate_anchors(self, height: int, width: int) -> dict:
        """Generate anchors for given input size."""
        cache_key = (height, width)
        if cache_key in self._anchors_cache:
            return self._anchors_cache[cache_key]

        anchors = {}
        for stride in self._feat_stride_fpn:
            fh = height // stride
            fw = width // stride

            # InsightFace convention: anchors at grid corners
            anchor_centers = np.stack(
                np.mgrid[:fh, :fw][::-1], axis=-1
            ).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))

            # Repeat for num_anchors (SCRFD uses 2 anchors per position)
            if self._num_anchors > 1:
                anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1)
                anchor_centers = anchor_centers.reshape((-1, 2))

            anchors[stride] = anchor_centers

        self._anchors_cache[cache_key] = anchors
        return anchors

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """Preprocess frame for SCRFD."""
        def normalize_scrfd(chw: np.ndarray) -> np.ndarray:
            input_host, _ = self._get_input_buffer()
            target_dtype = input_host.dtype
            # SCRFD: (pixel - 127.5) / 128
            normalized = (chw.astype(np.float32) - 127.5) / 128.0
            return normalized.astype(target_dtype)

        return self._letterbox_preprocess(frame, self.input_size, normalize_scrfd)

    def _decode_outputs(
        self,
        outputs: list[np.ndarray],
        scale: float,
        pad_h: int,
        pad_w: int,
        orig_shape: tuple[int, int],
    ) -> list[TensorRTFace]:
        """Decode SCRFD outputs to faces."""
        anchors = self._generate_anchors(self.input_size, self.input_size)

        # Group outputs by their last dimension
        scores_outputs = []
        bbox_outputs = []
        kps_outputs = []

        for out in outputs:
            if out.ndim >= 2 and out.shape[0] == 1:
                out = out.squeeze(0)

            if out.ndim == 1:
                scores_outputs.append((out.size, out))
            elif out.shape[-1] == 1:
                scores_outputs.append((out.shape[0], out.reshape(-1)))
            elif out.shape[-1] == 4:
                bbox_outputs.append((out.shape[0], out.reshape(-1, 4)))
            elif out.shape[-1] == 10:
                kps_outputs.append((out.shape[0], out.reshape(-1, 10)))
            else:
                size = out.size
                if size % 10 == 0:
                    kps_outputs.append((size // 10, out.reshape(-1, 10)))
                elif size % 4 == 0:
                    bbox_outputs.append((size // 4, out.reshape(-1, 4)))
                else:
                    scores_outputs.append((size, out.reshape(-1)))

        # Sort by element count (descending = stride 8 first)
        scores_outputs.sort(key=lambda x: -x[0])
        bbox_outputs.sort(key=lambda x: -x[0])
        kps_outputs.sort(key=lambda x: -x[0])

        all_scores = []
        all_boxes = []
        all_landmarks = []

        for idx, stride in enumerate(self._feat_stride_fpn):
            anchor_centers = anchors[stride]
            num_anchors = anchor_centers.shape[0]

            if idx >= len(scores_outputs) or idx >= len(bbox_outputs) or idx >= len(kps_outputs):
                logger.warning(f"Missing outputs for stride {stride}")
                continue

            scores = scores_outputs[idx][1].reshape(-1)[:num_anchors]
            bbox_preds = bbox_outputs[idx][1].reshape(-1, 4)[:num_anchors] * stride
            kps_preds = kps_outputs[idx][1].reshape(-1, 10)[:num_anchors] * stride

            # distance2bbox
            boxes = np.zeros_like(bbox_preds)
            boxes[:, 0] = anchor_centers[:, 0] - bbox_preds[:, 0]
            boxes[:, 1] = anchor_centers[:, 1] - bbox_preds[:, 1]
            boxes[:, 2] = anchor_centers[:, 0] + bbox_preds[:, 2]
            boxes[:, 3] = anchor_centers[:, 1] + bbox_preds[:, 3]

            # distance2kps
            kps = kps_preds.reshape(-1, 5, 2)
            kps[:, :, 0] = kps[:, :, 0] + anchor_centers[:, 0:1]
            kps[:, :, 1] = kps[:, :, 1] + anchor_centers[:, 1:2]

            all_scores.append(scores)
            all_boxes.append(boxes)
            all_landmarks.append(kps)

        if not all_scores:
            return []

        scores = np.concatenate(all_scores)
        boxes = np.concatenate(all_boxes)
        landmarks = np.concatenate(all_landmarks)

        # Filter by confidence
        mask = scores >= self.confidence_threshold
        scores = scores[mask]
        boxes = boxes[mask]
        landmarks = landmarks[mask]

        if len(scores) == 0:
            return []

        # Scale to original coordinates
        boxes = self._scale_coords_to_original(boxes, scale, pad_h, pad_w, orig_shape, is_bbox=True)
        landmarks = self._scale_coords_to_original(landmarks.reshape(-1, 2), scale, pad_h, pad_w, orig_shape, is_bbox=False)
        landmarks = landmarks.reshape(-1, 5, 2)

        # NMS
        indices = self._apply_nms(boxes, scores)

        faces = []
        for idx in indices:
            faces.append(TensorRTFace(
                bbox=(float(boxes[idx, 0]), float(boxes[idx, 1]),
                      float(boxes[idx, 2]), float(boxes[idx, 3])),
                confidence=float(scores[idx]),
                landmarks=landmarks[idx],
            ))

        return faces

    def detect(self, frame: np.ndarray) -> list[TensorRTFace]:
        """Detect faces in frame.

        Args:
            frame: BGR image

        Returns:
            List of detected faces with landmarks
        """
        self._load_engine()

        input_tensor, scale, pad_h, pad_w = self._preprocess(frame)
        outputs = self._run_inference(input_tensor)

        if not outputs:
            return []

        return self._decode_outputs(outputs, scale, pad_h, pad_w, frame.shape[:2])

    def warmup(self):
        """Warm up the engine."""
        self._load_engine()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.detect(dummy_frame)
        logger.info("TensorRT face detector warmed up")


class TensorRTFaceRecognizer:
    """Combined face detector and embedder using TensorRT.

    Provides a complete face recognition pipeline.
    """

    def __init__(
        self,
        det_model_path: str,
        rec_model_path: str,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        det_size: int = 640,
        min_face_size: int = 80,
    ):
        """Initialize TensorRT face recognizer.

        Args:
            det_model_path: Path to detection engine (SCRFD)
            rec_model_path: Path to recognition engine (ArcFace)
            confidence_threshold: Face detection confidence threshold
            nms_threshold: NMS IoU threshold
            det_size: Detection input size
            min_face_size: Minimum face size to keep
        """
        self.min_face_size = min_face_size

        self._detector = TensorRTFaceDetector(
            model_path=det_model_path,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            input_size=det_size,
        )

        self._embedder = TensorRTFaceEmbedder(model_path=rec_model_path)

    def detect_and_embed(self, frame: np.ndarray) -> list[TensorRTFace]:
        """Detect faces and extract embeddings.

        Args:
            frame: BGR image

        Returns:
            List of faces with embeddings
        """
        faces = self._detector.detect(frame)

        result = []
        for face in faces:
            width = face.bbox[2] - face.bbox[0]
            height = face.bbox[3] - face.bbox[1]

            if min(width, height) < self.min_face_size:
                continue

            aligned = align_face(frame, face.landmarks)
            face.embedding = self._embedder.extract(aligned)
            result.append(face)

        return result

    def warmup(self):
        """Warm up both models."""
        self._detector.warmup()
        self._embedder.warmup()
        logger.info("TensorRT face recognizer warmed up")

    def shutdown(self):
        """Release TensorRT resources for both models."""
        try:
            self._detector.shutdown()
        except Exception:
            pass
        try:
            self._embedder.shutdown()
        except Exception:
            pass
