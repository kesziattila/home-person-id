"""Stream manager for handling multiple camera streams."""

import logging
from typing import Callable, Iterator, Optional

from src.config import CameraConfig, Config
from src.stream.rtsp_client import Frame, RTSPClient

logger = logging.getLogger(__name__)


class StreamManager:
    """Manages multiple RTSP camera streams."""

    def __init__(self, config: Config):
        """Initialize stream manager.

        Args:
            config: Application configuration
        """
        self.config = config
        self._clients: dict[str, RTSPClient] = {}
        self._running = False

    @property
    def camera_ids(self) -> list[str]:
        """Get list of all camera IDs."""
        return list(self._clients.keys())

    @property
    def is_running(self) -> bool:
        """Check if the stream manager is running."""
        return self._running

    def add_camera(self, camera_config: CameraConfig) -> None:
        """Add a camera to the manager.

        Args:
            camera_config: Camera configuration
        """
        if camera_config.id in self._clients:
            logger.warning(f"Camera {camera_config.id} already exists")
            return

        client = RTSPClient(
            camera_id=camera_config.id,
            rtsp_url=camera_config.rtsp_url,
            target_fps=camera_config.fps,
            use_nvdec=camera_config.use_nvdec,
        )
        self._clients[camera_config.id] = client

        if self._running:
            client.start()

        decoder = "NVDEC" if camera_config.use_nvdec else "FFmpeg"
        logger.info(f"Added camera: {camera_config.id} ({camera_config.name}) [{decoder}]")

    def remove_camera(self, camera_id: str) -> None:
        """Remove a camera from the manager.

        Args:
            camera_id: Camera ID to remove
        """
        if camera_id not in self._clients:
            logger.warning(f"Camera {camera_id} not found")
            return

        client = self._clients.pop(camera_id)
        client.stop()
        logger.info(f"Removed camera: {camera_id}")

    def start(self) -> None:
        """Start all camera streams."""
        if self._running:
            logger.warning("Stream manager already running")
            return

        # Add cameras from config if not already added
        for camera_config in self.config.cameras:
            if camera_config.id not in self._clients:
                self.add_camera(camera_config)

        # Start all clients
        for client in self._clients.values():
            client.start()

        self._running = True
        logger.info(f"Started stream manager with {len(self._clients)} cameras")

    def stop(self) -> None:
        """Stop all camera streams."""
        for client in self._clients.values():
            client.stop()

        self._running = False
        logger.info("Stopped stream manager")

    def get_frame(self, camera_id: str, timeout: float = 1.0) -> Optional[Frame]:
        """Get a frame from a specific camera.

        Args:
            camera_id: Camera ID to get frame from
            timeout: Maximum time to wait for frame

        Returns:
            Frame object or None if no frame available
        """
        if camera_id not in self._clients:
            logger.warning(f"Camera {camera_id} not found")
            return None

        return self._clients[camera_id].get_frame(timeout=timeout)

    def get_frames(self, timeout: float = 0.1) -> Iterator[Frame]:
        """Get frames from all cameras (non-blocking).

        Args:
            timeout: Maximum time to wait per camera

        Yields:
            Frame objects from all cameras that have frames available
        """
        for camera_id, client in self._clients.items():
            frame = client.get_frame(timeout=timeout)
            if frame is not None:
                yield frame

    def get_client(self, camera_id: str) -> Optional[RTSPClient]:
        """Get the RTSP client for a camera.

        Args:
            camera_id: Camera ID

        Returns:
            RTSPClient or None if not found
        """
        return self._clients.get(camera_id)

    def is_camera_connected(self, camera_id: str) -> bool:
        """Check if a camera is connected.

        Args:
            camera_id: Camera ID

        Returns:
            True if camera is connected
        """
        client = self._clients.get(camera_id)
        return client is not None and client.is_connected

    def get_status(self) -> dict[str, dict]:
        """Get status of all cameras.

        Returns:
            Dictionary mapping camera IDs to status info
        """
        status = {}
        for camera_id, client in self._clients.items():
            status[camera_id] = {
                "connected": client.is_connected,
                "running": client.is_running,
            }
        return status

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
