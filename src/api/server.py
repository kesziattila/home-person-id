import asyncio
import cv2
import logging
import os
import threading
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from src.visualization.preview import PreviewBuffer, Visualizer
from src.tracking.zone_manager import ZoneManager
from src.utils.image_utils import encode_jpeg
from src.database.repository import Repository
from src.config import FaceRecognitionConfig, UnidentifiedFacesConfig
from src.api.routes.events import create_events_router
from src.api.routes.persons import create_persons_router
from src.api.routes.unidentified_faces import create_unidentified_faces_router

logger = logging.getLogger(__name__)

class APIServer:
    def __init__(
        self,
        buffer: PreviewBuffer,
        repository: Optional[Repository] = None,
        face_config: Optional[FaceRecognitionConfig] = None,
        unidentified_faces_config: Optional[UnidentifiedFacesConfig] = None,
        host: str = "0.0.0.0",
        port: int = 8000,
        use_nvjpeg: bool = False,
        zone_manager: Optional[ZoneManager] = None
    ):
        self.buffer = buffer
        self.repository = repository
        self.face_config = face_config
        self.unidentified_faces_config = unidentified_faces_config
        self.host = host
        self.port = port
        self.use_nvjpeg = use_nvjpeg
        self.zone_manager = zone_manager
        self.app = FastAPI(title="Home Person ID API")
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

        # Determine static directory relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.static_dir = os.path.join(os.path.dirname(current_dir), "static")

        self.setup_routes()
        self._thread: Optional[threading.Thread] = None

    def setup_routes(self):
        # Mount static files directory
        self.app.mount("/static", StaticFiles(directory=self.static_dir), name="static")

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
        async def stream(camera_id: str, request: Request):
            return StreamingResponse(
                self.generate_frames_threaded(camera_id, request),
                media_type="multipart/x-mixed-replace; boundary=frame"
            )

        # Register events router if repository is available
        if self.repository:
            events_router = create_events_router(self.repository)
            self.app.include_router(events_router)
            logger.info("Events API routes registered")

            # Register persons router if face config is available
            if self.face_config:
                persons_router = create_persons_router(
                    self.repository,
                    self.face_config
                )
                self.app.include_router(persons_router)
                logger.info("Persons API routes registered")

            # Register unidentified faces router if enabled
            if self.unidentified_faces_config and self.unidentified_faces_config.enabled:
                unidentified_faces_router = create_unidentified_faces_router(
                    self.repository,
                    self.face_config,
                    faces_dir=self.face_config.faces_dir if self.face_config else "data/faces"
                )
                self.app.include_router(unidentified_faces_router)
                logger.info("Unidentified faces API routes registered")

    def _process_and_encode_frame(self, preview_frame):
        """Processes a single frame by annotating and encoding it."""
        annotated_image = Visualizer.draw_detections(
            preview_frame.image, 
            preview_frame.metadata,
            zone_manager=self.zone_manager,
            show_zones=False
        )
        try:
            frame_bytes = encode_jpeg(annotated_image, use_nvjpeg=self.use_nvjpeg)
            return (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            logger.error(f"Error encoding frame: {e}")
        return None

    async def generate_frames_threaded(self, camera_id: str, request: Request):
        """Generate MJPEG frames for a camera, skipping duplicates."""
        loop = asyncio.get_running_loop()
        last_sent_frame = None
        while True:
            if await request.is_disconnected():
                logger.info(f"Client disconnected from stream {camera_id}. Stopping.")
                break
            
            try:
                preview_frame = self.buffer.get_latest_frame(camera_id)

                # Only process and send if the frame object itself is new
                if preview_frame is not None and preview_frame is not last_sent_frame:
                    last_sent_frame = preview_frame
                    
                    frame_bytes = await loop.run_in_executor(
                        self.executor, self._process_and_encode_frame, preview_frame
                    )
                    if frame_bytes:
                        yield frame_bytes
                
                # Sleep to control the polling rate / max FPS
                await asyncio.sleep(0.05)  # ~20 FPS polling

            except Exception as e:
                logger.error(f"Error in frame generation loop for {camera_id}: {e}")
                break

    def start(self):
        """Start the API server in a background thread."""
        self._thread = threading.Thread(
            target=lambda: uvicorn.run(self.app, host=self.host, port=self.port, log_level="error"),
            daemon=True
        )
        self._thread.start()
        logger.info(f"API server started at http://{self.host}:{self.port}")

    def stop(self):
        """Stop the API server and shutdown the thread pool."""
        logger.info("Stopping API server")
        self.executor.shutdown(wait=False)
