"""Motion detection using background subtraction."""

import logging
import time
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
    motion_mask: Optional[np.ndarray] = None  # Binary mask of motion areas
    bounding_boxes: Optional[list[tuple[int, int, int, int]]] = None  # Motion regions


class MotionDetector:
    """Motion detector using MOG2 background subtraction.

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

        # Create background subtractor
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.history,
            varThreshold=config.var_threshold,
            detectShadows=False,  # Disable shadow detection for speed
        )

        # Morphological kernel for noise reduction
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Cooldown tracking
        self._last_motion_time: float = 0.0
        self._in_cooldown: bool = False

    def detect(self, frame: np.ndarray, return_mask: bool = False) -> MotionResult:
        """Detect motion in a frame.

        Args:
            frame: BGR image from camera
            return_mask: Whether to include the motion mask in result

        Returns:
            MotionResult with detection info
        """
        if not self.enabled:
            # Motion detection disabled, always return True
            return MotionResult(has_motion=True, motion_ratio=1.0)

        # Apply background subtraction
        fg_mask = self._bg_subtractor.apply(frame)

        # Apply morphological operations to reduce noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self._kernel)

        # Calculate motion ratio
        total_pixels = frame.shape[0] * frame.shape[1]
        motion_pixels = cv2.countNonZero(fg_mask)
        motion_ratio = motion_pixels / total_pixels

        # Check if motion exceeds threshold
        has_motion = motion_ratio > self.config.min_area_ratio

        # Handle cooldown
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

        result = MotionResult(
            has_motion=has_motion,
            motion_ratio=motion_ratio,
        )

        if return_mask:
            result.motion_mask = fg_mask
            result.bounding_boxes = self._find_motion_regions(fg_mask)

        return result

    def _find_motion_regions(
        self, mask: np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        """Find bounding boxes of motion regions.

        Args:
            mask: Binary motion mask

        Returns:
            List of (x, y, w, h) bounding boxes
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        boxes = []
        min_area = 500  # Minimum contour area to consider

        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                boxes.append((x, y, w, h))

        return boxes

    def reset(self) -> None:
        """Reset the background model.

        Call this when camera view changes significantly.
        """
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config.history,
            varThreshold=self.config.var_threshold,
            detectShadows=False,
        )
        self._last_motion_time = 0.0
        self._in_cooldown = False
        logger.info("Motion detector reset")

    @property
    def in_cooldown(self) -> bool:
        """Check if currently in cooldown period after motion stopped."""
        return self._in_cooldown


class MotionDetectorManager:
    """Manages motion detectors for multiple cameras."""

    def __init__(self, config: MotionConfig):
        """Initialize motion detector manager.

        Args:
            config: Motion detection configuration
        """
        self.config = config
        self._detectors: dict[str, MotionDetector] = {}

    def get_detector(self, camera_id: str) -> MotionDetector:
        """Get or create a motion detector for a camera.

        Args:
            camera_id: Camera identifier

        Returns:
            MotionDetector for the camera
        """
        if camera_id not in self._detectors:
            self._detectors[camera_id] = MotionDetector(self.config)
            logger.debug(f"Created motion detector for camera {camera_id}")

        return self._detectors[camera_id]

    def detect(
        self, camera_id: str, frame: np.ndarray, return_mask: bool = False
    ) -> MotionResult:
        """Detect motion in a frame from a specific camera.

        Args:
            camera_id: Camera identifier
            frame: BGR image from camera
            return_mask: Whether to include motion mask

        Returns:
            MotionResult with detection info
        """
        detector = self.get_detector(camera_id)
        return detector.detect(frame, return_mask=return_mask)

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
