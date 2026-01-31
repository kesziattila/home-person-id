"""Person detection using YOLOv8."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# COCO class ID for person
PERSON_CLASS_ID = 0


@dataclass
class Detection:
    """A single person detection."""

    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int = PERSON_CLASS_ID

    @property
    def center(self) -> tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def width(self) -> float:
        """Get width of bounding box."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        """Get height of bounding box."""
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        """Get area of bounding box."""
        return self.width * self.height

    def to_xyxy(self) -> list[float]:
        """Convert to [x1, y1, x2, y2] format."""
        return list(self.bbox)

    def to_xywh(self) -> list[float]:
        """Convert to [x, y, w, h] format (center-based)."""
        cx, cy = self.center
        return [cx, cy, self.width, self.height]

    def to_tlwh(self) -> list[float]:
        """Convert to [x, y, w, h] format (top-left based)."""
        x1, y1, x2, y2 = self.bbox
        return [x1, y1, x2 - x1, y2 - y1]


def compute_iou(
    box1: tuple[float, float, float, float],
    box2: tuple[float, float, float, float],
) -> float:
    """Compute Intersection over Union between two bounding boxes.

    Args:
        box1, box2: Bounding boxes as (x1, y1, x2, y2)

    Returns:
        IoU value between 0 and 1
    """
    x1_i = max(box1[0], box2[0])
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[2], box2[2])
    y2_i = min(box1[3], box2[3])

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


@dataclass
class DetectionResult:
    """Result of person detection on a frame."""

    detections: list[Detection]
    frame_shape: tuple[int, int, int]  # (height, width, channels)

    @property
    def count(self) -> int:
        """Number of detections."""
        return len(self.detections)

    def filter_by_confidence(self, min_confidence: float) -> "DetectionResult":
        """Filter detections by minimum confidence."""
        filtered = [d for d in self.detections if d.confidence >= min_confidence]
        return DetectionResult(detections=filtered, frame_shape=self.frame_shape)

    def filter_by_height(self, min_height: float) -> "DetectionResult":
        """Filter detections by minimum height in pixels."""
        filtered = [d for d in self.detections if d.height >= min_height]
        return DetectionResult(detections=filtered, frame_shape=self.frame_shape)

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array of shape (N, 5) with [x1, y1, x2, y2, conf]."""
        if not self.detections:
            return np.empty((0, 5))

        return np.array(
            [list(d.bbox) + [d.confidence] for d in self.detections],
            dtype=np.float32,
        )


class PersonDetector:
    """Person detector using YOLOv8."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        nms_iou_threshold: float = 0.4,
        device: Optional[str] = None,
        num_threads: int = 0,
    ):
        """Initialize person detector.

        Args:
            model_path: Path to YOLO model. If None, uses yolov8n.pt
            confidence_threshold: Minimum confidence for detections
            nms_iou_threshold: IoU threshold for NMS (lower = more aggressive merging)
            device: Device to run on ('cpu', 'cuda', '0', etc.)
            num_threads: Number of CPU threads (0 = auto)
        """
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = device
        self.num_threads = num_threads

        # Lazy load ultralytics to avoid import overhead
        self._model = None
        self._model_path = model_path or "yolov8n.pt"

        # Set thread count if specified
        if num_threads > 0:
            import torch
            torch.set_num_threads(num_threads)
            logger.info(f"Set PyTorch threads to {num_threads}")

    def _load_model(self):
        """Load the YOLO model (lazy loading)."""
        if self._model is not None:
            return

        try:
            from ultralytics import YOLO
            import torch

            # Enable CUDA optimizations if available
            if torch.cuda.is_available():
                # Enable cuDNN autotuner for faster convolutions
                torch.backends.cudnn.benchmark = True
                # Reduce memory fragmentation
                torch.cuda.empty_cache()

            logger.info(f"Loading YOLO model: {self._model_path}")
            self._model = YOLO(self._model_path, task="detect")

            if self.device:
                self._model.to(self.device)

            logger.info("YOLO model loaded successfully")

        except ImportError:
            raise ImportError(
                "ultralytics is required for person detection. "
                "Install with: pip install ultralytics"
            )

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: Optional[float] = None,
    ) -> DetectionResult:
        """Detect persons in a frame.

        Args:
            frame: BGR image from camera
            confidence_threshold: Override default confidence threshold

        Returns:
            DetectionResult containing all person detections
        """
        self._load_model()

        conf = confidence_threshold or self.confidence_threshold

        # Run inference
        results = self._model(
            frame,
            conf=conf,
            iou=self.nms_iou_threshold,  # NMS IoU threshold (lower = fewer overlapping boxes)
            classes=[PERSON_CLASS_ID],  # Only detect persons
            verbose=False,
        )

        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            # Batch transfer from GPU to CPU (single transfer instead of per-detection)
            all_bboxes = boxes.xyxy.cpu().numpy()
            all_confs = boxes.conf.cpu().numpy()
            all_classes = boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                if all_classes[i] == PERSON_CLASS_ID:
                    bbox = all_bboxes[i]
                    detections.append(
                        Detection(
                            bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                            confidence=float(all_confs[i]),
                            class_id=int(all_classes[i]),
                        )
                    )

        return DetectionResult(
            detections=detections,
            frame_shape=frame.shape,
        )

    def warmup(self, frame_shape: tuple[int, int, int] = (1080, 1920, 3)) -> None:
        """Warm up the model with a dummy inference.

        Args:
            frame_shape: Shape of frames that will be processed
        """
        self._load_model()
        # Use a real-size frame for better warmup
        dummy_frame = np.zeros(frame_shape, dtype=np.uint8)
        self.detect(dummy_frame)
        logger.info(f"Person detector warmed up with shape {frame_shape}")
