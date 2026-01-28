import threading
import cv2
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from src.visualization.render import TrackRenderer, ZoneRenderer
from src.recognition.identification_manager import TrackIdentity
from src.tracking.zone_manager import ZoneManager

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
    def draw_detections(
        image: np.ndarray,
        metadata: Dict[str, Any],
        zone_manager: Optional[ZoneManager] = None,
        show_zones: bool = True
    ) -> np.ndarray:
        """Draw bounding boxes, labels and zones on the image."""
        track_renderer = TrackRenderer(zone_manager=zone_manager)
        draw_img = image.copy()
        frame_h, frame_w = draw_img.shape[:2]
        
        # Draw zones if manager provided and requested
        camera_id = metadata.get("camera_id", "unknown")
        if zone_manager and show_zones:
            zone_renderer = ZoneRenderer(zone_manager)
            zone_renderer.draw_zones(draw_img, camera_id)
        
        # Local tracks from TrackerResult
        tracks = metadata.get("tracks", [])
        identities = metadata.get("identities", {})
        stationary_times = metadata.get("stationary_times", {})
        camera_id = metadata.get("camera_id", "unknown")
        
        for track in tracks:
            # track is a LocalTrack object with bbox, track_id
            bbox = getattr(track, 'bbox', None)
            local_id = getattr(track, 'track_id', None)
            
            if bbox is not None:
                identity = identities.get(local_id)
                stationary_time = stationary_times.get(local_id)
                
                if not identity:
                    # Create a dummy TrackIdentity for unidentified tracks
                    # to use the centralized label/color logic
                    identity = TrackIdentity(track_id=str(local_id))
                
                # Ensure stationary_time is reflected in identity if possible,
                # though get_track_label_and_color takes it as an argument anyway.
                label, color = track_renderer.get_track_label_and_color(
                    str(local_id), identity, camera_id, bbox, frame_w, frame_h,
                    stationary_time=stationary_time
                )
                track_renderer.draw_track(draw_img, bbox, label, color)
        
        return draw_img
