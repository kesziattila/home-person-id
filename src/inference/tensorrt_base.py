"""Base classes for TensorRT inference.

Provides common functionality for TensorRT-based models including:
- Engine loading and CUDA context management
- Buffer allocation
- Letterbox preprocessing
- Inference execution

Subclasses implement model-specific preprocessing and postprocessing.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class TensorRTEngine(ABC):
    """Base class for TensorRT inference engines.

    Handles common TensorRT operations:
    - Engine loading with lazy initialization
    - CUDA context management (thread-safe)
    - Input/output buffer allocation
    - Synchronous inference execution

    Subclasses must implement:
    - _validate_engine(): Validate engine format after loading
    - _get_input_size(): Return expected input size for validation
    """

    def __init__(self, model_path: str):
        """Initialize TensorRT engine.

        Args:
            model_path: Path to TensorRT engine file (.engine or .trt)
        """
        self.model_path = model_path

        # Lazy initialization
        self._engine = None
        self._context = None
        self._cuda = None
        self._cuda_context = None
        self._buffers: dict[str, tuple[np.ndarray, object, tuple]] = {}  # name -> (host, device, shape)
        self._input_name: Optional[str] = None
        self._output_names: list[str] = []

    def _load_engine(self):
        """Load TensorRT engine (lazy loading)."""
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

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"TensorRT engine not found: {self.model_path}")

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

        # Setup buffers
        self._setup_buffers(cuda, trt)

        # Validate engine format (subclass-specific)
        self._validate_engine()

        # Pop context - will push when needed
        self._cuda_context.pop()

        logger.info(f"TensorRT engine loaded: {self.model_path}")

    def _setup_buffers(self, cuda, trt):
        """Allocate input/output buffers."""
        self._output_names = []

        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = self._engine.get_tensor_shape(name)
            dtype = trt.nptype(self._engine.get_tensor_dtype(name))
            is_input = self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT

            # Handle dynamic shapes for input
            if is_input and (-1 in shape or 0 in shape):
                input_size = self._get_input_size()
                shape = (1, 3, input_size, input_size)
                self._context.set_input_shape(name, shape)

            # Validate input size matches expected
            if is_input and len(shape) == 4:
                model_h, model_w = shape[2], shape[3]
                expected_size = self._get_input_size()
                if model_h != expected_size or model_w != expected_size:
                    self._cuda_context.pop()
                    raise RuntimeError(
                        f"TensorRT model input size mismatch!\n"
                        f"  Model expects: {model_w}x{model_h}\n"
                        f"  Config expects: {expected_size}x{expected_size}\n\n"
                        f"Re-convert the model with the correct input size."
                    )

            # Allocate buffers
            host_mem = np.empty(shape, dtype=dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self._context.set_tensor_address(name, int(device_mem))
            self._buffers[name] = (host_mem, device_mem, shape)

            if is_input:
                self._input_name = name
            else:
                self._output_names.append(name)

        # Sort output names for consistent ordering
        self._output_names.sort()

    @abstractmethod
    def _validate_engine(self):
        """Validate engine format after loading.

        Subclasses should check output format matches expected structure.
        Raise RuntimeError with helpful message if validation fails.
        """
        pass

    @abstractmethod
    def _get_input_size(self) -> int:
        """Return expected input size for this engine."""
        pass

    def _get_input_buffer(self) -> tuple[np.ndarray, object]:
        """Get input buffer (host, device)."""
        if self._input_name is None:
            raise RuntimeError("Engine not loaded")
        host, device, _ = self._buffers[self._input_name]
        return host, device

    def _get_output_buffers(self) -> list[tuple[np.ndarray, object, tuple]]:
        """Get output buffers in sorted order."""
        return [self._buffers[name] for name in self._output_names]

    def _run_inference(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        """Run inference and return outputs.

        Args:
            input_tensor: Preprocessed input tensor

        Returns:
            List of output arrays (copied from device)
        """
        self._load_engine()
        self._cuda_context.push()

        try:
            # Copy input to device
            input_host, input_device = self._get_input_buffer()
            np.copyto(input_host, input_tensor)
            self._cuda.memcpy_htod(input_device, input_host)

            # Run inference
            if not self._context.execute_async_v3(stream_handle=0):
                logger.error("TensorRT inference failed")
                return []

            self._cuda.Context.synchronize()

            # Copy outputs to host
            outputs = []
            for name in self._output_names:
                host, device, _ = self._buffers[name]
                self._cuda.memcpy_dtoh(host, device)
                outputs.append(host.copy())

            return outputs

        finally:
            self._cuda_context.pop()

    def _letterbox_preprocess(
        self,
        frame: np.ndarray,
        input_size: int,
        normalize_fn: callable,
    ) -> tuple[np.ndarray, float, int, int]:
        """Common letterbox preprocessing.

        Args:
            frame: BGR image (H, W, C)
            input_size: Target size (square)
            normalize_fn: Function to normalize pixel values, receives CHW array

        Returns:
            (preprocessed tensor, scale, pad_h, pad_w)
        """
        h, w = frame.shape[:2]
        scale = min(input_size / h, input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to square
        pad_h = (input_size - new_h) // 2
        pad_w = (input_size - new_w) // 2

        padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # BGR to RGB, HWC to CHW
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2, 0, 1)

        # Normalize using provided function
        normalized = normalize_fn(chw)

        return normalized[np.newaxis, ...], scale, pad_h, pad_w

    def _scale_coords_to_original(
        self,
        coords: np.ndarray,
        scale: float,
        pad_h: int,
        pad_w: int,
        orig_shape: tuple[int, int],
        is_bbox: bool = True,
    ) -> np.ndarray:
        """Scale coordinates from letterboxed input back to original image.

        Args:
            coords: Coordinates array, shape (..., 2) for points or (..., 4) for boxes
            scale: Scale factor used in preprocessing
            pad_h, pad_w: Padding applied during preprocessing
            orig_shape: Original image (H, W)
            is_bbox: If True, coords are (x1, y1, x2, y2) boxes; else (x, y) points

        Returns:
            Scaled coordinates
        """
        coords = coords.copy()
        orig_h, orig_w = orig_shape

        if is_bbox:
            coords[:, 0] = np.clip((coords[:, 0] - pad_w) / scale, 0, orig_w)
            coords[:, 1] = np.clip((coords[:, 1] - pad_h) / scale, 0, orig_h)
            coords[:, 2] = np.clip((coords[:, 2] - pad_w) / scale, 0, orig_w)
            coords[:, 3] = np.clip((coords[:, 3] - pad_h) / scale, 0, orig_h)
        else:
            # Points: shape (..., 2) where last dim is (x, y)
            coords[..., 0] = (coords[..., 0] - pad_w) / scale
            coords[..., 1] = (coords[..., 1] - pad_h) / scale

        return coords


class TensorRTDetectorBase(TensorRTEngine):
    """Base class for TensorRT-based object detectors.

    Provides common detection functionality including confidence thresholding
    and NMS. Subclasses implement model-specific decoding.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: int = 640,
    ):
        """Initialize detector.

        Args:
            model_path: Path to TensorRT engine
            confidence_threshold: Minimum confidence for detections
            nms_threshold: NMS IoU threshold
            input_size: Model input size
        """
        super().__init__(model_path)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

    def _get_input_size(self) -> int:
        return self.input_size

    def _apply_nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
    ) -> list[int]:
        """Apply Non-Maximum Suppression.

        Args:
            boxes: Bounding boxes (N, 4) in x1,y1,x2,y2 format
            scores: Confidence scores (N,)

        Returns:
            Indices of boxes to keep
        """
        if len(boxes) == 0:
            return []

        # Convert to xywh for cv2.dnn.NMSBoxes
        boxes_xywh = np.zeros_like(boxes)
        boxes_xywh[:, 0] = boxes[:, 0]
        boxes_xywh[:, 1] = boxes[:, 1]
        boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(),
            scores.tolist(),
            self.confidence_threshold,
            self.nms_threshold,
        )

        # Handle different return formats from OpenCV versions
        result = []
        for i in indices:
            idx = i[0] if isinstance(i, (list, np.ndarray)) else i
            result.append(idx)
        return result


class TensorRTEmbedderBase(TensorRTEngine):
    """Base class for TensorRT-based embedding extractors.

    Provides common embedding functionality including L2 normalization.
    Subclasses implement model-specific preprocessing.
    """

    def __init__(
        self,
        model_path: str,
        input_size: int = 112,
    ):
        """Initialize embedder.

        Args:
            model_path: Path to TensorRT engine
            input_size: Model input size
        """
        super().__init__(model_path)
        self.input_size = input_size

    def _get_input_size(self) -> int:
        return self.input_size

    def _validate_engine(self):
        """Validate embedder output is a feature vector."""
        # Embedders typically output (1, embedding_dim) or (embedding_dim,)
        # No strict validation needed
        pass

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """L2 normalize embedding."""
        norm = np.linalg.norm(embedding) + 1e-8
        return (embedding / norm).astype(np.float32)
