"""RTSP stream client for reading camera frames."""

import logging
import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    """Container for a video frame with metadata."""

    image: np.ndarray
    timestamp: float
    camera_id: str
    frame_number: int


class RTSPClient:
    """RTSP stream client that reads frames in a background thread."""

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        target_fps: int = 5,
        buffer_size: int = 2,
        reconnect_delay: float = 5.0,
    ):
        """Initialize RTSP client.

        Args:
            camera_id: Unique identifier for the camera
            rtsp_url: RTSP URL to connect to
            target_fps: Target frames per second to capture
            buffer_size: Maximum frames to buffer
            reconnect_delay: Seconds to wait before reconnecting after failure
        """
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.target_fps = target_fps
        self.buffer_size = buffer_size
        self.reconnect_delay = reconnect_delay

        self._frame_queue: Queue[Frame] = Queue(maxsize=buffer_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0
        self._last_frame_time = 0.0
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if the client is connected to the stream."""
        return self._connected

    @property
    def is_running(self) -> bool:
        """Check if the client is running."""
        return self._running

    def start(self) -> None:
        """Start the stream reader thread."""
        if self._running:
            logger.warning(f"Camera {self.camera_id}: Already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        logger.info(f"Camera {self.camera_id}: Started stream reader")

    def stop(self) -> None:
        """Stop the stream reader thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._release_capture()
        logger.info(f"Camera {self.camera_id}: Stopped stream reader")

    def get_frame(self, timeout: float = 1.0) -> Optional[Frame]:
        """Get the next frame from the buffer.

        Args:
            timeout: Maximum time to wait for a frame

        Returns:
            Frame object or None if no frame available
        """
        try:
            return self._frame_queue.get(timeout=timeout)
        except Empty:
            return None

    def _connect(self) -> bool:
        """Connect to the RTSP stream."""
        self._release_capture()

        logger.info(f"Camera {self.camera_id}: Connecting to {self.rtsp_url}")

        # Configure OpenCV capture with optimal settings for RTSP
        self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        if self._cap is None or not self._cap.isOpened():
            logger.error(f"Camera {self.camera_id}: Failed to open stream")
            self._connected = False
            return False

        # Set buffer size to minimize latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Try to get stream properties
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)

        logger.info(
            f"Camera {self.camera_id}: Connected - {width}x{height} @ {fps:.1f} FPS"
        )

        self._connected = True
        return True

    def _release_capture(self) -> None:
        """Release the video capture object."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._connected = False

    def _read_loop(self) -> None:
        """Main loop for reading frames from the stream."""
        frame_interval = 1.0 / self.target_fps

        while self._running:
            # Connect if not connected
            if not self._connected:
                if not self._connect():
                    logger.warning(
                        f"Camera {self.camera_id}: Reconnecting in {self.reconnect_delay}s"
                    )
                    time.sleep(self.reconnect_delay)
                    continue

            # Always grab frames to prevent buffer accumulation (keeps stream real-time)
            # grab() is fast - it just advances the buffer without decoding
            try:
                if not self._cap.grab():
                    logger.warning(f"Camera {self.camera_id}: Failed to grab frame")
                    self._connected = False
                    continue
            except Exception as e:
                logger.error(f"Camera {self.camera_id}: Error grabbing frame: {e}")
                self._connected = False
                time.sleep(0.1)
                continue

            # Only decode and use frame at target FPS rate
            current_time = time.time()
            elapsed = current_time - self._last_frame_time
            if elapsed < frame_interval:
                # Frame grabbed but not used - prevents buffer buildup
                continue

            # Retrieve (decode) the grabbed frame
            try:
                ret, image = self._cap.retrieve()

                if not ret or image is None:
                    logger.warning(f"Camera {self.camera_id}: Failed to retrieve frame")
                    continue

                self._frame_count += 1
                self._last_frame_time = time.time()

                frame = Frame(
                    image=image,
                    timestamp=self._last_frame_time,
                    camera_id=self.camera_id,
                    frame_number=self._frame_count,
                )

                # Try to add to queue, drop oldest if full
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except Empty:
                        pass

                self._frame_queue.put_nowait(frame)

            except Exception as e:
                logger.error(f"Camera {self.camera_id}: Error retrieving frame: {e}")
                time.sleep(0.1)

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
