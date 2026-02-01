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
        self._input_buffer = None
        self._output_buffer = None
        self._d_input = None
        self._d_output = None
        self._cuda = None
        self._cuda_context = None

    def _load_engine(self):
        """Load TensorRT engine."""
        if self._engine is not None:
            return

        try:
            import tensorrt as trt
            import pycuda.driver as cuda
        except ImportError as e:
            raise ImportError(
                f"TensorRT native backend requires tensorrt and pycuda. "
                f"Install with: pip install tensorrt pycuda. Error: {e}"
            )

        logger.info(f"Loading TensorRT engine: {self.model_path}")

        # Initialize CUDA and create context
        cuda.init()
        self._cuda = cuda
        device = cuda.Device(0)
        self._cuda_context = device.make_context()

        # Load engine
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(self.model_path, "rb") as f:
            engine_data = f.read()

        runtime = trt.Runtime(trt_logger)
        self._engine = runtime.deserialize_cuda_engine(engine_data)

        if self._engine is None:
            self._cuda_context.pop()
            raise RuntimeError(f"Failed to load TensorRT engine: {self.model_path}")

        self._context = self._engine.create_execution_context()

        # Get input/output shapes and allocate buffers
        self._setup_buffers(cuda)

        # Pop context - will push when needed
        self._cuda_context.pop()

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
            Preprocessed tensor (1, C, H, W) normalized to [0, 1]
            Dtype matches the model's input requirement (float32 or float16)
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

        # Match dtype to model's input requirement (FP16 or FP32)
        target_dtype = self._input_buffer.dtype if self._input_buffer is not None else np.float32
        normalized = chw.astype(target_dtype) / 255.0

        # Add batch dimension
        return normalized[np.newaxis, ...]

    def _postprocess(
        self, output: np.ndarray, orig_shape: tuple[int, int]
    ) -> list[Detection]:
        """Postprocess YOLO output to detections.

        Supports two formats:
        1. Raw YOLO: (1, 84, 8400) - needs NMS
        2. With NMS: (1, N, 6) - already filtered [x1,y1,x2,y2,conf,class]

        Args:
            output: Raw model output
            orig_shape: Original image shape (H, W)

        Returns:
            List of Detection objects
        """
        # Remove batch dimension if present
        if len(output.shape) == 3:
            output = output[0]

        # Detect output format based on shape
        # NMS format: (N, 6) where 6 = [x1, y1, x2, y2, confidence, class_id]
        # Raw format: (84, 8400) or (8400, 84)
        if output.shape[-1] == 6:
            return self._postprocess_nms_format(output, orig_shape)
        else:
            return self._postprocess_raw_format(output, orig_shape)

    def _postprocess_nms_format(
        self, output: np.ndarray, orig_shape: tuple[int, int]
    ) -> list[Detection]:
        """Postprocess YOLO output with baked-in NMS.

        Format: (N, 6) = [x1, y1, x2, y2, confidence, class_id]
        """
        detections = []
        orig_h, orig_w = orig_shape

        for det in output:
            x1, y1, x2, y2, conf, class_id = det

            # Skip empty detections (padding)
            if conf < self.confidence_threshold:
                continue

            # Filter for person class only
            if int(class_id) != PERSON_CLASS_ID:
                continue

            # Scale coordinates from letterboxed input to original image
            x1 = (x1 - self._pad_w) / self._scale
            y1 = (y1 - self._pad_h) / self._scale
            x2 = (x2 - self._pad_w) / self._scale
            y2 = (y2 - self._pad_h) / self._scale

            # Clip to image bounds
            x1 = np.clip(x1, 0, orig_w)
            y1 = np.clip(y1, 0, orig_h)
            x2 = np.clip(x2, 0, orig_w)
            y2 = np.clip(y2, 0, orig_h)

            detections.append(
                Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(conf),
                    class_id=PERSON_CLASS_ID,
                )
            )

        return detections

    def _postprocess_raw_format(
        self, output: np.ndarray, orig_shape: tuple[int, int]
    ) -> list[Detection]:
        """Postprocess raw YOLO output (no NMS).

        Format: (84, 8400) or (8400, 84) = [xywh + class_scores]
        """
        # Transpose if needed: (84, 8400) -> (8400, 84)
        if output.shape[0] < output.shape[1]:
            output = output.T

        # Split into boxes and class scores
        boxes = output[:, :4]  # xywh format
        scores = output[:, 4:]  # class scores

        # Get person class scores
        person_scores = scores[:, PERSON_CLASS_ID]

        # Filter by confidence
        mask = person_scores >= self.confidence_threshold
        boxes = boxes[mask]
        person_scores = person_scores[mask]

        if len(boxes) == 0:
            return []

        # Convert xywh to xyxy
        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        # Scale to original image
        orig_h, orig_w = orig_shape
        xyxy[:, 0] = (xyxy[:, 0] - self._pad_w) / self._scale
        xyxy[:, 1] = (xyxy[:, 1] - self._pad_h) / self._scale
        xyxy[:, 2] = (xyxy[:, 2] - self._pad_w) / self._scale
        xyxy[:, 3] = (xyxy[:, 3] - self._pad_h) / self._scale

        # Clip to bounds
        xyxy[:, 0] = np.clip(xyxy[:, 0], 0, orig_w)
        xyxy[:, 1] = np.clip(xyxy[:, 1], 0, orig_h)
        xyxy[:, 2] = np.clip(xyxy[:, 2], 0, orig_w)
        xyxy[:, 3] = np.clip(xyxy[:, 3], 0, orig_h)

        # Apply NMS
        indices = self._nms(xyxy, person_scores, self.nms_iou_threshold)

        # Create detections
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
        self._load_engine()

        # Temporarily override confidence threshold if provided
        orig_conf = self.confidence_threshold
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold

        # Push CUDA context for this thread
        self._cuda_context.push()

        try:
            # Preprocess (CPU operation, no CUDA needed)
            input_tensor = self._preprocess(frame)

            # Copy input to device (synchronous)
            np.copyto(self._input_buffer, input_tensor)
            self._cuda.memcpy_htod(self._d_input, self._input_buffer)

            # Run inference - tensor addresses were set in _setup_buffers
            # Use default stream (0) for synchronous execution
            if not self._context.execute_async_v3(stream_handle=0):
                logger.error("TensorRT inference failed")
                return DetectionResult(detections=[], frame_shape=frame.shape)

            # Synchronize to ensure inference is complete
            self._cuda.Context.synchronize()

            # Copy output to host (synchronous)
            self._cuda.memcpy_dtoh(self._output_buffer, self._d_output)

            # Postprocess (CPU operation)
            detections = self._postprocess(
                self._output_buffer.copy(), frame.shape[:2]
            )

            return DetectionResult(
                detections=detections,
                frame_shape=frame.shape,
            )

        finally:
            self.confidence_threshold = orig_conf
            # Pop CUDA context
            self._cuda_context.pop()

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
