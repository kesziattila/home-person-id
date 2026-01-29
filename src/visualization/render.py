import cv2
import numpy as np
from typing import Optional
from src.recognition.identification_manager import TrackIdentity
from src.tracking.zone_manager import ZoneManager

class TrackRenderer:
    """Renders tracks with labels and bounding boxes."""

    # Colors
    COLOR_FACE_IDENTIFIED = (255, 150, 0)    # Blue for face identified
    COLOR_REID_IDENTIFIED = (255, 100, 100)  # Light blue for Re-ID identified
    COLOR_UNIDENTIFIED = (0, 255, 0)         # Green for unidentified
    COLOR_MULTI_CAMERA = (0, 255, 255)       # Yellow for multi-camera
    COLOR_STATIONARY = (0, 200, 255)         # Orange for stationary

    def __init__(self, zone_manager: Optional[ZoneManager] = None):
        """Initialize track renderer.

        Args:
            zone_manager: Optional zone manager for zone labels
        """
        self.zone_manager = zone_manager

    def get_track_label_and_color(
        self,
        track_id: str,
        identity: TrackIdentity,
        camera_id: str,
        bbox: tuple[float, float, float, float],
        frame_w: int,
        frame_h: int,
        stationary_time: Optional[int] = None,
        cameras_seen: Optional[list[str]] = None,
        show_reid_score: bool = True,
    ) -> tuple[str, tuple[int, int, int]]:
        """Get label and color for a track.

        Args:
            track_id: Track identifier (local or global)
            identity: Track identity information
            camera_id: Current camera ID
            bbox: Bounding box
            frame_w: Frame width
            frame_h: Frame height
            stationary_time: Seconds stationary (None if not stationary)
            cameras_seen: List of cameras this track has been seen on
            show_reid_score: Whether to show Re-ID score for unidentified tracks

        Returns:
            Tuple of (label, color)
        """
        is_multi_camera = cameras_seen and len(cameras_seen) > 1

        if identity.person_name:
            # Identified
            if identity.is_reid_identified:
                label = f"{identity.person_name} (R:{identity.confidence:.2f})"
                color = self.COLOR_REID_IDENTIFIED
            else:
                label = f"{identity.person_name} (F:{identity.confidence:.2f})"
                color = self.COLOR_FACE_IDENTIFIED
        else:
            # Unidentified
            # Clean track ID for display
            display_id = track_id.replace('global_', '') if 'global_' in str(track_id) else str(track_id)
            label = f"#{display_id}"

            # Add face info (closest match even if below threshold)
            if identity.face_info:
                closest_name, face_conf = identity.face_info
                label += f" ~{closest_name}({face_conf:.2f})"

            # Add Re-ID info (closest match even if below threshold)
            if identity.reid_info and not identity.face_info:
                closest_name, reid_conf = identity.reid_info
                label += f" (?){closest_name}({reid_conf:.2f})"

            # Always show Re-ID score if available (for debugging/monitoring)
            if identity.reid_score >= 0 and not identity.reid_info:
                label += f" R:{identity.reid_score:.2f}"

            color = self.COLOR_UNIDENTIFIED

        # Add zone info
        if self.zone_manager:
            track_zone = self.zone_manager.get_person_zone(camera_id, bbox, frame_w, frame_h)
            if track_zone:
                label += f" [{track_zone}]"

        # Add multi-camera indicator
        if is_multi_camera:
            label += " *"
            if not identity.person_name:
                color = self.COLOR_MULTI_CAMERA

        # Add stationary indicator
        if stationary_time is not None:
            label += f" S:{stationary_time}s"
            if not identity.person_name:
                color = self.COLOR_STATIONARY

        return label, color

    def draw_track(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
        label: str,
        color: tuple[int, int, int],
    ):
        """Draw a track bounding box and label on the frame.

        Args:
            frame: Frame to draw on (modified in place)
            bbox: Bounding box (x1, y1, x2, y2)
            label: Label text
            color: BGR color tuple
        """
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


class ZoneRenderer:
    """Renders polygon zones on frames."""

    ZONE_COLORS = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]

    def __init__(self, zone_manager: "ZoneManager"):
        """Initialize zone renderer.

        Args:
            zone_manager: Zone manager instance
        """
        self.zone_manager = zone_manager

    def draw_zones(self, frame: np.ndarray, camera_id: str, alpha: float = 0.2):
        """Draw all zones for a camera on the frame.

        Args:
            frame: Frame to draw on (modified in place)
            camera_id: Camera identifier
            alpha: Transparency for zone fill (0-1)
        """
        frame_h, frame_w = frame.shape[:2]
        zone_visualizations = self.zone_manager.visualize_zones(camera_id, frame_w, frame_h)

        for i, (polygon_points, zone_name) in enumerate(zone_visualizations):
            color = self.ZONE_COLORS[i % len(self.ZONE_COLORS)]
            pts = np.array(polygon_points, np.int32).reshape((-1, 1, 2))

            # Draw filled polygon with transparency
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Draw polygon outline
            cv2.polylines(frame, [pts], True, color, 2)

            # Draw zone name at centroid
            if polygon_points:
                centroid_x = int(sum(p[0] for p in polygon_points) / len(polygon_points))
                centroid_y = int(sum(p[1] for p in polygon_points) / len(polygon_points))
                cv2.putText(frame, zone_name, (centroid_x - 30, centroid_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
