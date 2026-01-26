import asyncio
import cv2
import logging
import os
import threading
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional

from src.visualization.preview import PreviewBuffer, Visualizer

logger = logging.getLogger(__name__)

class APIServer:
    def __init__(self, buffer: PreviewBuffer, host: str = "0.0.0.0", port: int = 8000):
        self.buffer = buffer
        self.host = host
        self.port = port
        self.app = FastAPI(title="Home Person ID API")
        
        # Determine static directory relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.static_dir = os.path.join(os.path.dirname(current_dir), "static")
        
        self.setup_routes()
        self._thread: Optional[threading.Thread] = None

    def setup_routes(self):
        @self.app.get("/")
        async def index():
            index_path = os.path.join(self.static_dir, "index.html")
            if os.path.exists(index_path):
                return FileResponse(index_path)
            return {"status": "running", "cameras": self.buffer.get_all_camera_ids()}

        @self.app.get("/api/v1/cameras")
        async def get_cameras():
            return {"cameras": self.buffer.get_all_camera_ids()}

        @self.app.get("/api/v1/stream/{camera_id}")
        async def stream(camera_id: str):
            return StreamingResponse(
                self.generate_frames(camera_id),
                media_type="multipart/x-mixed-replace; boundary=frame"
            )

    async def generate_frames(self, camera_id: str):
        """Generate MJPEG frames for a camera."""
        while True:
            preview_frame = self.buffer.get_latest_frame(camera_id)
            if preview_frame is not None:
                # Draw annotations
                annotated_image = Visualizer.draw_detections(
                    preview_frame.image, 
                    preview_frame.metadata
                )
                
                # Encode to JPEG
                ret, buffer = cv2.imencode('.jpg', annotated_image)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Control frame rate for the stream
            await asyncio.sleep(0.05)  # ~20 FPS

    def start(self):
        """Start the API server in a background thread."""
        self._thread = threading.Thread(
            target=lambda: uvicorn.run(self.app, host=self.host, port=self.port, log_level="error"),
            daemon=True
        )
        self._thread.start()
        logger.info(f"API server started at http://{self.host}:{self.port}")

    def stop(self):
        """Stop the API server (daemon thread will stop with main)."""
        logger.info("Stopping API server")
