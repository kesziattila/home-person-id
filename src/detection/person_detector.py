"""Person detection using YOLOv8.

Supports two backends:
1. Ultralytics YOLO (default) - uses PyTorch, supports .pt and .engine files
2. TensorRT native - direct TensorRT inference, lower memory, supports .engine/.trt files

Use TensorRT native backend by setting use_tensorrt_native=True in config when using
.engine or .trt model files. This avoids PyTorch overhead and reduces memory usage.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
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


class TensorRTDetector:
    """Person detector using TensorRT directly (no PyTorch dependency).

    This implementation loads TensorRT engines directly without going through
    Ultralytics/PyTorch, reducing memory usage and startup time.

    Only supports .engine or .trt model files.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        nms_iou_threshold: float = 0.4,
        input_size: int = 640,
    ):
        """Initialize TensorRT detector.

        Args:
            model_path: Path to TensorRT engine file (.engine or .trt)
            confidence_threshold: Minimum confidence for detections
            nms_iou_threshold: IoU threshold for NMS
            input_size: Model input size (assumes square input)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.input_size = input_size

        # Lazy initialization
        self._engine = None
        self._context = None
        self._stream = None
        self._bindings = None
        self._input_buffer = None
        self._output_buffer = None
        self._d_input = None
        self._d_output = None

    def _load_engine(self):
        """Load TensorRT engine."""
        if self._engine is not None:
            return

        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401 - Required for CUDA context
        except ImportError as e:
            raise ImportError(
                f"TensorRT native backend requires tensorrt and pycuda. "
                f"Install with: pip install tensorrt pycuda. Error: {e}"
            )

        logger.info(f"Loading TensorRT engine: {self.model_path}")

        # Load engine
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(self.model_path, "rb") as f:
            engine_data = f.read()

        runtime = trt.Runtime(trt_logger)
        self._engine = runtime.deserialize_cuda_engine(engine_data)

        if self._engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {self.model_path}")

        self._context = self._engine.create_execution_context()
        self._stream = cuda.Stream()

        # Get input/output shapes
        self._setup_buffers(cuda)

        logger.info(
            f"TensorRT engine loaded: input={self._input_shape}, output={self._output_shape}"
        )

    def _setup_buffers(self, cuda):
        """Allocate input/output buffers."""
        import tensorrt as trt

        self._tensor_names = {}  # name -> (host_buffer, device_buffer)

        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = self._engine.get_tensor_shape(name)
            dtype = trt.nptype(self._engine.get_tensor_dtype(name))

            # Allocate host and device memory
            host_mem = np.empty(shape, dtype=dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self._tensor_names[name] = (host_mem, device_mem)

            # Set tensor address for v3 API
            self._context.set_tensor_address(name, int(device_mem))

            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_buffer = host_mem
                self._d_input = device_mem
                self._input_shape = shape
                self._input_name = name
            else:
                self._output_buffer = host_mem
                self._d_output = device_mem
                self._output_shape = shape
                self._output_name = name

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for YOLO inference.

        Args:
            frame: BGR image (H, W, C)

        Returns:
            Preprocessed tensor (1, C, H, W) float32 normalized to [0, 1]
        """
        # Resize with letterboxing to maintain aspect ratio
        h, w = frame.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create letterboxed image (pad with gray)
        letterboxed = np.full(
            (self.input_size, self.input_size, 3), 114, dtype=np.uint8
        )
        pad_h = (self.input_size - new_h) // 2
        pad_w = (self.input_size - new_w) // 2
        letterboxed[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        # Store padding info for postprocessing
        self._pad_h = pad_h
        self._pad_w = pad_w
        self._scale = scale

        # BGR to RGB, HWC to CHW, normalize to [0, 1]
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2, 0, 1)  # HWC -> CHW
        normalized = chw.astype(np.float32) / 255.0

        # Add batch dimension
        return normalized[np.newaxis, ...]

    def _postprocess(
        self, output: np.ndarray, orig_shape: tuple[int, int]
    ) -> list[Detection]:
        """Postprocess YOLO output to detections.

        Args:
            output: Raw model output
            orig_shape: Original image shape (H, W)

        Returns:
            List of Detection objects
        """
        # YOLOv8 output shape: (1, 84, 8400) where 84 = 4 (xywh) + 80 (classes)
        # Transpose to (8400, 84) for easier processing
        if len(output.shape) == 3:
            output = output[0]  # Remove batch dimension
        if output.shape[0] < output.shape[1]:
            output = output.T  # (84, 8400) -> (8400, 84)

        # Split into boxes and class scores
        boxes = output[:, :4]  # xywh format (center x, center y, width, height)
        scores = output[:, 4:]  # class scores

        # Get person class scores (class 0)
        person_scores = scores[:, PERSON_CLASS_ID]

        # Filter by confidence
        mask = person_scores >= self.confidence_threshold
        boxes = boxes[mask]
        person_scores = person_scores[mask]

        if len(boxes) == 0:
            return []

        # Convert xywh to xyxy
        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

        # Remove letterbox padding and scale to original image
        xyxy[:, 0] = (xyxy[:, 0] - self._pad_w) / self._scale
        xyxy[:, 1] = (xyxy[:, 1] - self._pad_h) / self._scale
        xyxy[:, 2] = (xyxy[:, 2] - self._pad_w) / self._scale
        xyxy[:, 3] = (xyxy[:, 3] - self._pad_h) / self._scale

        # Clip to image bounds
        orig_h, orig_w = orig_shape
        xyxy[:, 0] = np.clip(xyxy[:, 0], 0, orig_w)
        xyxy[:, 1] = np.clip(xyxy[:, 1], 0, orig_h)
        xyxy[:, 2] = np.clip(xyxy[:, 2], 0, orig_w)
        xyxy[:, 3] = np.clip(xyxy[:, 3], 0, orig_h)

        # Apply NMS
        indices = self._nms(xyxy, person_scores, self.nms_iou_threshold)

        # Create Detection objects
        detections = []
        for i in indices:
            detections.append(
                Detection(
                    bbox=(xyxy[i, 0], xyxy[i, 1], xyxy[i, 2], xyxy[i, 3]),
                    confidence=float(person_scores[i]),
                    class_id=PERSON_CLASS_ID,
                )
            )

        return detections

    def _nms(
        self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
    ) -> list[int]:
        """Non-maximum suppression.

        Args:
            boxes: Bounding boxes (N, 4) in xyxy format
            scores: Confidence scores (N,)
            iou_threshold: IoU threshold for suppression

        Returns:
            Indices of kept boxes
        """
        # Sort by score descending
        order = scores.argsort()[::-1]

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            # Compute IoU with remaining boxes
            remaining = order[1:]
            ious = np.array(
                [compute_iou(tuple(boxes[i]), tuple(boxes[j])) for j in remaining]
            )

            # Keep boxes with IoU below threshold
            order = remaining[ious <= iou_threshold]

        return keep

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
        import pycuda.driver as cuda

        self._load_engine()

        # Temporarily override confidence threshold if provided
        orig_conf = self.confidence_threshold
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold

        try:
            # Preprocess
            input_tensor = self._preprocess(frame)

            # Copy input to device
            np.copyto(self._input_buffer, input_tensor)
            cuda.memcpy_htod_async(self._d_input, self._input_buffer, self._stream)

            # Run inference (v3 API for TensorRT 10+)
            self._context.execute_async_v3(stream_handle=self._stream.handle)

            # Copy output to host
            cuda.memcpy_dtoh_async(self._output_buffer, self._d_output, self._stream)
            self._stream.synchronize()

            # Postprocess
            detections = self._postprocess(
                self._output_buffer.copy(), frame.shape[:2]
            )

            return DetectionResult(
                detections=detections,
                frame_shape=frame.shape,
            )

        finally:
            self.confidence_threshold = orig_conf

    def warmup(self, frame_shape: tuple[int, int, int] = (1080, 1920, 3)) -> None:
        """Warm up the engine with a dummy inference."""
        self._load_engine()
        dummy_frame = np.zeros(frame_shape, dtype=np.uint8)
        self.detect(dummy_frame)
        logger.info(f"TensorRT detector warmed up with shape {frame_shape}")


def create_person_detector(
    model_path: Optional[str] = None,
    confidence_threshold: float = 0.5,
    nms_iou_threshold: float = 0.4,
    device: Optional[str] = None,
    num_threads: int = 0,
    use_tensorrt_native: bool = False,
):
    """Factory function to create the appropriate person detector.

    Args:
        model_path: Path to model file
        confidence_threshold: Minimum confidence for detections
        nms_iou_threshold: IoU threshold for NMS
        device: Device for inference (PersonDetector only)
        num_threads: CPU threads (PersonDetector only)
        use_tensorrt_native: Force TensorRT native backend for .engine/.trt files

    Returns:
        PersonDetector or TensorRTDetector instance
    """
    model_path = model_path or "yolov8n.pt"

    # Use TensorRT native if explicitly requested and model is an engine file
    if use_tensorrt_native and model_path.endswith((".engine", ".trt")):
        logger.info(f"Using TensorRT native backend for {model_path}")
        return TensorRTDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
        )

    # Default to Ultralytics/PyTorch backend
    return PersonDetector(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        nms_iou_threshold=nms_iou_threshold,
        device=device,
        num_threads=num_threads,
    )
