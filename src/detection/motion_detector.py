"""Motion detection using background subtraction."""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.config import MotionConfig

logger = logging.getLogger(__name__)


@dataclass
class MotionResult:
    """Result of motion detection."""

    has_motion: bool
    motion_ratio: float  # Ratio of pixels with motion (0-1)


class BaseMotionDetector(ABC):
    """Abstract base class for motion detection.

    This serves as a processing gate to skip expensive ML inference
    when no motion is detected in the frame.
    """

    def __init__(self, config: MotionConfig):
        """Initialize motion detector.

        Args:
            config: Motion detection configuration
        """
        self.config = config
        self.enabled = config.enabled

        # Cooldown tracking
        self._last_motion_time: float = 0.0
        self._in_cooldown: bool = False

        # Morphological kernel for noise reduction
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: np.ndarray) -> MotionResult:
        """Detect motion in a frame.

        Args:
            frame: BGR image from camera

        Returns:
            MotionResult with detection info
        """
        if not self.enabled:
            return MotionResult(has_motion=True, motion_ratio=1.0)

        h, w = frame.shape[:2]
        target_width, target_height, scale = self._get_processing_size(h, w)

        # Implementation-specific processing
        motion_ratio = self._process_frame(frame, target_width, target_height, scale)

        # Check if motion exceeds threshold and handle cooldown
        has_motion = self._handle_cooldown(motion_ratio > self.config.min_area_ratio)

        return MotionResult(
            has_motion=has_motion,
            motion_ratio=motion_ratio,
        )

    @abstractmethod
    def _process_frame(
        self,
        frame: np.ndarray,
        target_width: int,
        target_height: int,
        scale: float,
    ) -> float:
        """Implementation-specific motion detection steps.

        Returns:
            motion_ratio: Ratio of pixels with motion (0-1)
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the background model.

        Call this when camera view changes significantly.
        """
        pass

    def _handle_cooldown(self, has_motion: bool) -> bool:
        """Handle motion cooldown logic.

        Args:
            has_motion: Whether motion was detected in current frame

        Returns:
            Effective has_motion after considering cooldown
        """
        current_time = time.time()
        if has_motion:
            self._last_motion_time = current_time
            self._in_cooldown = True
        elif self._in_cooldown:
            # Check if still in cooldown period
            time_since_motion = current_time - self._last_motion_time
            if time_since_motion < self.config.cooldown_sec:
                has_motion = True  # Continue processing during cooldown
            else:
                self._in_cooldown = False
        return has_motion

    def _get_processing_size(self, h: int, w: int) -> tuple[int, int, float]:
        """Calculate processing dimensions based on config.

        Returns:
            (target_width, target_height, scale_factor)
        """
        target_height = self.config.processing_height
        if h > target_height:
            scale = target_height / h
            target_width = int(w * scale)
            return target_width, target_height, scale
        return w, h, 1.0

    @property
    def in_cooldown(self) -> bool:
        """Check if currently in cooldown period after motion stopped."""
        return self._in_cooldown


class CPUMotionDetector(BaseMotionDetector):
    """CPU implementation of motion detection using MOG2."""

    def __init__(self, config: MotionConfig):
        super().__init__(config)
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.history,
            varThreshold=config.var_threshold,
            detectShadows=False,
        )

    def _process_frame(
        self,
        frame: np.ndarray,
        target_width: int,
        target_height: int,
        scale: float,
    ) -> float:
        if scale < 1.0:
            proc_frame = cv2.resize(
                frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR
            )
        else:
            proc_frame = frame

        # Apply background subtraction
        fg_mask = self._bg_subtractor.apply(proc_frame)

        # Apply morphological operations to reduce noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self._kernel)

        # Calculate motion ratio
        total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
        motion_pixels = cv2.countNonZero(fg_mask)
        motion_ratio = motion_pixels / total_pixels

        return motion_ratio

    def reset(self) -> None:
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config.history,
            varThreshold=self.config.var_threshold,
            detectShadows=False,
        )
        self._last_motion_time = 0.0
        self._in_cooldown = False
        logger.info("CPU Motion detector reset")


class CUDAMotionDetector(BaseMotionDetector):
    """CUDA implementation of motion detection using MOG2."""

    def __init__(self, config: MotionConfig):
        super().__init__(config)
        self._bg_subtractor = cv2.cuda.createBackgroundSubtractorMOG2(
            history=config.history,
            varThreshold=config.var_threshold,
            detectShadows=False,
        )
        # CUDA streams for asynchronous processing
        self._stream = cv2.cuda_Stream()
        # GPU mats for processing
        self._gpu_frame = cv2.cuda_GpuMat()
        self._gpu_fg_mask = cv2.cuda_GpuMat()
        # Only allocate resize buffer if doing GPU resize
        self._gpu_resized = cv2.cuda_GpuMat() if not config.resize_on_cpu else None
        self._resize_on_cpu = config.resize_on_cpu

        # CUDA version of morphological filters
        self._morph_filter = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_OPEN, cv2.CV_8UC1, self._kernel
        )
        self._morph_filter_close = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_CLOSE, cv2.CV_8UC1, self._kernel
        )

        if self._resize_on_cpu:
            logger.info("CUDA motion detector using CPU pre-resize (lower GPU memory)")

    def _process_frame(
        self,
        frame: np.ndarray,
        target_width: int,
        target_height: int,
        scale: float,
    ) -> float:
        # Resize on CPU before upload to save GPU memory (~25MB per 4K camera)
        if self._resize_on_cpu and scale < 1.0:
            frame = cv2.resize(
                frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR
            )
            self._gpu_frame.upload(frame, stream=self._stream)
            proc_gpu_frame = self._gpu_frame
        else:
            # GPU Processing path (original behavior)
            self._gpu_frame.upload(frame, stream=self._stream)

            # Downscale frame for faster processing on GPU
            if scale < 1.0:
                self._gpu_resized = cv2.cuda.resize(
                    self._gpu_frame, (target_width, target_height), stream=self._stream
                )
                proc_gpu_frame = self._gpu_resized
            else:
                proc_gpu_frame = self._gpu_frame

        # Apply background subtraction
        self._gpu_fg_mask = self._bg_subtractor.apply(
            proc_gpu_frame, -1, self._stream
        )

        # Apply morphological operations
        self._morph_filter.apply(
            self._gpu_fg_mask, self._gpu_fg_mask, stream=self._stream
        )
        self._morph_filter_close.apply(
            self._gpu_fg_mask, self._gpu_fg_mask, stream=self._stream
        )

        # Calculate motion ratio on GPU
        motion_pixels = cv2.cuda.countNonZero(self._gpu_fg_mask)
        total_pixels = target_width * target_height
        motion_ratio = motion_pixels / total_pixels

        return motion_ratio

    def reset(self) -> None:
        self._bg_subtractor = cv2.cuda.createBackgroundSubtractorMOG2(
            history=self.config.history,
            varThreshold=self.config.var_threshold,
            detectShadows=False,
        )
        self._last_motion_time = 0.0
        self._in_cooldown = False
        logger.info("CUDA Motion detector reset")


def MotionDetector(config: MotionConfig) -> BaseMotionDetector:
    """Factory function to create the appropriate motion detector.

    Args:
        config: Motion detection configuration

    Returns:
        Instance of CPUMotionDetector or CUDAMotionDetector
    """
    if config.use_cuda:
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                logger.info("Using CUDA-accelerated motion detection")
                return CUDAMotionDetector(config)
            else:
                logger.warning("CUDA requested but no CUDA device found. Falling back to CPU.")
        except Exception as e:
            logger.warning(f"Failed to initialize CUDA motion detection: {e}. Falling back to CPU.")
    
    return CPUMotionDetector(config)


class MotionDetectorManager:
    """Manages motion detectors for multiple cameras."""

    def __init__(self, config: MotionConfig):
        """Initialize motion detector manager.

        Args:
            config: Motion detection configuration
        """
        self.config = config
        self._detectors: dict[str, BaseMotionDetector] = {}

    def get_detector(self, camera_id: str) -> BaseMotionDetector:
        """Get or create a motion detector for a camera.

        Args:
            camera_id: Camera identifier

        Returns:
            BaseMotionDetector for the camera
        """
        if camera_id not in self._detectors:
            self._detectors[camera_id] = MotionDetector(self.config)
            logger.debug(f"Created motion detector for camera {camera_id}")

        return self._detectors[camera_id]

    def detect(self, camera_id: str, frame: np.ndarray) -> MotionResult:
        """Detect motion in a frame from a specific camera.

        Args:
            camera_id: Camera identifier
            frame: BGR image from camera

        Returns:
            MotionResult with detection info
        """
        detector = self.get_detector(camera_id)
        return detector.detect(frame)

    def reset(self, camera_id: Optional[str] = None) -> None:
        """Reset motion detector(s).

        Args:
            camera_id: Specific camera to reset, or None for all
        """
        if camera_id is not None:
            if camera_id in self._detectors:
                self._detectors[camera_id].reset()
        else:
            for detector in self._detectors.values():
                detector.reset()
