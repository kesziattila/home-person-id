"""Single-frame endpoint used by scripts/remote/grab_frames.sh for zone review."""

import cv2
import numpy as np
from fastapi.testclient import TestClient

from src.api.server import APIServer
from src.visualization.preview import PreviewBuffer


def _client(buffer):
    server = APIServer(buffer)
    return TestClient(server.app)


def test_frame_returns_jpeg_of_latest_buffered_frame():
    buffer = PreviewBuffer()
    buffer.update("living_room", np.full((90, 160, 3), 128, dtype=np.uint8), {"camera_id": "living_room"})

    response = _client(buffer).get("/api/v1/frame/living_room")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (90, 160, 3)


def test_frame_unknown_camera_is_404():
    assert _client(PreviewBuffer()).get("/api/v1/frame/nope").status_code == 404
