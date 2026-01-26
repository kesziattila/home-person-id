import threading
import cv2
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from src.preview import TrackRenderer

@dataclass
class PreviewFrame:
    image: np.ndarray
    camera_id: str
    timestamp: float
    metadata: Dict[str, Any]

class PreviewBuffer:
    """Thread-safe buffer for storing the latest frames and metadata for web preview."""
    
    def __init__(self):
        self._frames: Dict[str, PreviewFrame] = {}
        self._lock = threading.Lock()
    
    def update(self, camera_id: str, image: np.ndarray, metadata: Dict[str, Any]):
        """Update the latest frame for a camera."""
        import time
        with self._lock:
            # We store a copy to avoid mutation issues if the original frame is modified
            self._frames[camera_id] = PreviewFrame(
                image=image.copy(),
                camera_id=camera_id,
                timestamp=time.time(),
                metadata=metadata
            )
            
    def get_latest_frame(self, camera_id: str) -> Optional[PreviewFrame]:
        """Get the latest frame for a camera."""
        with self._lock:
            return self._frames.get(camera_id)

    def get_all_camera_ids(self) -> List[str]:
        """Get all camera IDs currently in the buffer."""
        with self._lock:
            return list(self._frames.keys())

class Visualizer:
    """Helper class to draw annotations on frames."""
    
    @staticmethod
    def draw_detections(image: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """Draw bounding boxes and labels on the image."""
        from src.preview import TrackRenderer
        track_renderer = TrackRenderer()
        draw_img = image.copy()
        frame_h, frame_w = draw_img.shape[:2]
        
        # Local tracks from TrackerResult
        tracks = metadata.get("tracks", [])
        identities = metadata.get("identities", {})
        camera_id = metadata.get("camera_id", "unknown")
        
        for track in tracks:
            # track is a LocalTrack object with bbox, track_id
            bbox = getattr(track, 'bbox', None)
            local_id = getattr(track, 'track_id', None)
            
            if bbox is not None:
                identity = identities.get(local_id)
                if identity:
                    label, color = track_renderer.get_track_label_and_color(
                        str(local_id), identity, camera_id, bbox, frame_w, frame_h
                    )
                    track_renderer.draw_track(draw_img, bbox, label, color)
                else:
                    # Fallback if no identity object (should not happen with current main.py)
                    x1, y1, x2, y2 = map(int, bbox)
                    label = f"#{local_id}"
                    color = (0, 255, 0)
                    cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(draw_img, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return draw_img
