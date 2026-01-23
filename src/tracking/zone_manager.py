"""Zone manager for polygon-based room zones.

Manages polygon zones for camera handover detection using
point-in-polygon tests on the bottom-center of bounding boxes.
"""

import logging
from typing import Optional

from src.config import ZoneConfig, ZonesConfig

logger = logging.getLogger(__name__)


class ZoneManager:
    """Manages polygon zones for handover detection."""

    def __init__(self, zones_config: ZonesConfig):
        """Initialize zone manager.

        Args:
            zones_config: Zones configuration from config file
        """
        self.zones = zones_config.zones
        logger.info(f"Loaded {len(self.zones)} polygon zones")

    def point_in_polygon(
        self, point: tuple[float, float], polygon: list[list[float]]
    ) -> bool:
        """Ray casting algorithm for point-in-polygon test.

        Args:
            point: (x, y) coordinates to test
            polygon: List of [x, y] vertex coordinates

        Returns:
            True if point is inside polygon
        """
        x, y = point
        n = len(polygon)
        if n < 3:
            return False

        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def get_person_zone(
        self,
        camera_id: str,
        bbox: tuple[float, float, float, float],
        frame_w: int,
        frame_h: int,
    ) -> Optional[str]:
        """Get zone name if person's bottom-center is inside any zone.

        Args:
            camera_id: Camera identifier
            bbox: Bounding box in pixel coordinates (x1, y1, x2, y2)
            frame_w: Frame width in pixels
            frame_h: Frame height in pixels

        Returns:
            Zone name if person is in a zone, None otherwise
        """
        # Calculate bottom-center (normalized to 0-1)
        x1, y1, x2, y2 = bbox
        bottom_center_x = ((x1 + x2) / 2) / frame_w
        bottom_center_y = y2 / frame_h  # Bottom of bbox

        for zone in self.zones:
            if camera_id not in zone.cameras:
                continue
            polygon = zone.cameras[camera_id].polygon
            if self.point_in_polygon((bottom_center_x, bottom_center_y), polygon):
                return zone.name
        return None

    def get_persons_in_zone(
        self,
        zone_name: str,
        camera_id: str,
        tracks: list,
        frame_w: int,
        frame_h: int,
    ) -> list:
        """Get all tracks that are in a specific zone.

        Args:
            zone_name: Name of the zone to check
            camera_id: Camera identifier
            tracks: List of track objects with bbox attribute
            frame_w: Frame width in pixels
            frame_h: Frame height in pixels

        Returns:
            List of tracks that are in the specified zone
        """
        result = []
        for track in tracks:
            track_zone = self.get_person_zone(camera_id, track.bbox, frame_w, frame_h)
            if track_zone == zone_name:
                result.append(track)
        return result

    def get_zone_config(self, zone_name: str) -> Optional[ZoneConfig]:
        """Get zone configuration by name.

        Args:
            zone_name: Name of the zone

        Returns:
            ZoneConfig if found, None otherwise
        """
        for zone in self.zones:
            if zone.name == zone_name:
                return zone
        return None

    def get_zones_for_camera(self, camera_id: str) -> list[ZoneConfig]:
        """Get all zones visible on a specific camera.

        Args:
            camera_id: Camera identifier

        Returns:
            List of ZoneConfig objects for zones visible on this camera
        """
        return [zone for zone in self.zones if camera_id in zone.cameras]

    def get_other_cameras_for_zone(self, zone_name: str, camera_id: str) -> list[str]:
        """Get other cameras that can see the same zone.

        Args:
            zone_name: Name of the zone
            camera_id: Current camera identifier

        Returns:
            List of camera IDs that can see the zone (excluding current camera)
        """
        zone = self.get_zone_config(zone_name)
        if zone is None:
            return []
        return [c for c in zone.cameras.keys() if c != camera_id]

    def is_alone_in_zone(
        self,
        zone_name: str,
        camera_id: str,
        tracks: list,
        frame_w: int,
        frame_h: int,
    ) -> bool:
        """Check if there is exactly one person in the zone.

        Args:
            zone_name: Name of the zone to check
            camera_id: Camera identifier
            tracks: List of track objects with bbox attribute
            frame_w: Frame width in pixels
            frame_h: Frame height in pixels

        Returns:
            True if exactly one person is in the zone
        """
        persons = self.get_persons_in_zone(zone_name, camera_id, tracks, frame_w, frame_h)
        return len(persons) == 1

    def visualize_zones(
        self,
        camera_id: str,
        frame_w: int,
        frame_h: int,
    ) -> list[tuple[list[tuple[int, int]], str]]:
        """Get zone polygons for visualization.

        Args:
            camera_id: Camera identifier
            frame_w: Frame width in pixels
            frame_h: Frame height in pixels

        Returns:
            List of (polygon_points, zone_name) tuples where polygon_points
            are in pixel coordinates
        """
        result = []
        for zone in self.zones:
            if camera_id not in zone.cameras:
                continue
            polygon = zone.cameras[camera_id].polygon
            # Convert normalized coordinates to pixels
            pixel_points = [
                (int(p[0] * frame_w), int(p[1] * frame_h)) for p in polygon
            ]
            result.append((pixel_points, zone.name))
        return result
