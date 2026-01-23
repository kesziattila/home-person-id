"""Camera handover logic for overlapping views.

Handles the transfer of track identity when a person moves from
one camera's field of view to an overlapping camera.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from src.config import CameraOverlap, CameraTopologyConfig

logger = logging.getLogger(__name__)


@dataclass
class Zone:
    """Rectangular zone in normalized coordinates (0-1)."""

    x1: float
    y1: float
    x2: float
    y2: float

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is within the zone.

        Args:
            x: X coordinate (0-1 normalized)
            y: Y coordinate (0-1 normalized)

        Returns:
            True if point is in zone
        """
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def contains_center(self, bbox: tuple[float, float, float, float]) -> bool:
        """Check if bbox center is within the zone.

        Args:
            bbox: Bounding box in (x1, y1, x2, y2) pixel coordinates

        Returns:
            True if center is in zone
        """
        # This assumes bbox is already normalized or will be normalized by caller
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        return self.contains_point(cx, cy)

    def overlap_ratio(self, bbox: tuple[float, float, float, float]) -> float:
        """Calculate what fraction of bbox is within the zone.

        Args:
            bbox: Bounding box in normalized (x1, y1, x2, y2) coordinates

        Returns:
            Overlap ratio (0-1)
        """
        # Calculate intersection
        ix1 = max(self.x1, bbox[0])
        iy1 = max(self.y1, bbox[1])
        ix2 = min(self.x2, bbox[2])
        iy2 = min(self.y2, bbox[3])

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        intersection = (ix2 - ix1) * (iy2 - iy1)
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        if bbox_area <= 0:
            return 0.0

        return intersection / bbox_area


@dataclass
class HandoverEdge:
    """Directed edge in the camera topology graph."""

    from_camera: str
    to_camera: str
    exit_zone: Zone
    entry_zone: Zone
    max_handover_sec: float


class HandoverManager:
    """Manages camera handover zones and logic.

    Determines when a track is exiting one camera's view and entering another.
    """

    def __init__(self, topology_config: CameraTopologyConfig):
        """Initialize handover manager.

        Args:
            topology_config: Camera topology configuration
        """
        self.topology_config = topology_config
        self._edges: list[HandoverEdge] = []
        self._camera_exits: dict[str, list[HandoverEdge]] = {}
        self._camera_entries: dict[str, list[HandoverEdge]] = {}

        self._build_graph()

    def _build_graph(self):
        """Build the camera topology graph from config."""
        for overlap in self.topology_config.overlaps:
            # Forward direction: cameras[0] -> cameras[1]
            forward_edge = HandoverEdge(
                from_camera=overlap.cameras[0],
                to_camera=overlap.cameras[1],
                exit_zone=Zone(*overlap.cam1_exit_zone),
                entry_zone=Zone(*overlap.cam2_entry_zone),
                max_handover_sec=overlap.max_handover_sec,
            )
            self._edges.append(forward_edge)

            # Backward direction: cameras[1] -> cameras[0]
            backward_edge = HandoverEdge(
                from_camera=overlap.cameras[1],
                to_camera=overlap.cameras[0],
                exit_zone=Zone(*overlap.cam2_entry_zone),
                entry_zone=Zone(*overlap.cam1_exit_zone),
                max_handover_sec=overlap.max_handover_sec,
            )
            self._edges.append(backward_edge)

        # Build lookup tables
        for edge in self._edges:
            if edge.from_camera not in self._camera_exits:
                self._camera_exits[edge.from_camera] = []
            self._camera_exits[edge.from_camera].append(edge)

            if edge.to_camera not in self._camera_entries:
                self._camera_entries[edge.to_camera] = []
            self._camera_entries[edge.to_camera].append(edge)

        logger.info(
            f"Built handover graph with {len(self._edges)} edges "
            f"for {len(self._camera_exits)} cameras"
        )

    def get_exit_zones(self, camera_id: str) -> list[tuple[str, Zone]]:
        """Get exit zones for a camera.

        Args:
            camera_id: Camera identifier

        Returns:
            List of (target_camera_id, exit_zone) tuples
        """
        edges = self._camera_exits.get(camera_id, [])
        return [(e.to_camera, e.exit_zone) for e in edges]

    def get_entry_zones(self, camera_id: str) -> list[tuple[str, Zone]]:
        """Get entry zones for a camera.

        Args:
            camera_id: Camera identifier

        Returns:
            List of (source_camera_id, entry_zone) tuples
        """
        edges = self._camera_entries.get(camera_id, [])
        return [(e.from_camera, e.entry_zone) for e in edges]

    def check_exit(
        self,
        camera_id: str,
        bbox: tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
    ) -> Optional[tuple[str, float]]:
        """Check if a track is exiting through an overlap zone.

        Args:
            camera_id: Current camera
            bbox: Track bounding box in pixel coordinates
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels

        Returns:
            Tuple of (target_camera_id, overlap_ratio) if exiting, else None
        """
        edges = self._camera_exits.get(camera_id, [])
        if not edges:
            return None

        # Normalize bbox to 0-1 coordinates
        norm_bbox = (
            bbox[0] / frame_width,
            bbox[1] / frame_height,
            bbox[2] / frame_width,
            bbox[3] / frame_height,
        )

        # Check each exit zone
        best_match = None
        best_overlap = 0.0

        for edge in edges:
            overlap = edge.exit_zone.overlap_ratio(norm_bbox)
            if overlap > 0.3 and overlap > best_overlap:  # At least 30% overlap
                best_match = edge.to_camera
                best_overlap = overlap

        if best_match:
            return (best_match, best_overlap)

        return None

    def check_entry(
        self,
        camera_id: str,
        bbox: tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
    ) -> Optional[tuple[str, float]]:
        """Check if a track is entering through an overlap zone.

        Args:
            camera_id: Current camera
            bbox: Track bounding box in pixel coordinates
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels

        Returns:
            Tuple of (source_camera_id, overlap_ratio) if entering, else None
        """
        edges = self._camera_entries.get(camera_id, [])
        if not edges:
            return None

        # Normalize bbox to 0-1 coordinates
        norm_bbox = (
            bbox[0] / frame_width,
            bbox[1] / frame_height,
            bbox[2] / frame_width,
            bbox[3] / frame_height,
        )

        # Check each entry zone
        best_match = None
        best_overlap = 0.0

        for edge in edges:
            overlap = edge.entry_zone.overlap_ratio(norm_bbox)
            if overlap > 0.3 and overlap > best_overlap:
                best_match = edge.from_camera
                best_overlap = overlap

        if best_match:
            return (best_match, best_overlap)

        return None

    def get_handover_config(
        self, from_camera: str, to_camera: str
    ) -> Optional[HandoverEdge]:
        """Get handover configuration between two cameras.

        Args:
            from_camera: Source camera
            to_camera: Target camera

        Returns:
            HandoverEdge if exists, else None
        """
        for edge in self._edges:
            if edge.from_camera == from_camera and edge.to_camera == to_camera:
                return edge
        return None

    def get_max_handover_time(self, from_camera: str, to_camera: str) -> float:
        """Get maximum handover time between two cameras.

        Args:
            from_camera: Source camera
            to_camera: Target camera

        Returns:
            Maximum handover time in seconds
        """
        edge = self.get_handover_config(from_camera, to_camera)
        if edge:
            return edge.max_handover_sec
        return 3.0  # Default

    def get_connected_cameras(self, camera_id: str) -> list[str]:
        """Get cameras that are connected (overlap) with a camera.

        Args:
            camera_id: Camera identifier

        Returns:
            List of connected camera IDs
        """
        connected = set()
        for edge in self._edges:
            if edge.from_camera == camera_id:
                connected.add(edge.to_camera)
            elif edge.to_camera == camera_id:
                connected.add(edge.from_camera)
        return list(connected)

    def visualize_zones(
        self,
        camera_id: str,
        frame_width: int,
        frame_height: int,
    ) -> list[tuple[tuple[int, int, int, int], str, str]]:
        """Get zone rectangles for visualization.

        Args:
            camera_id: Camera identifier
            frame_width: Frame width
            frame_height: Frame height

        Returns:
            List of (bbox_pixels, zone_type, target_camera) tuples
        """
        result = []

        # Exit zones
        for target, zone in self.get_exit_zones(camera_id):
            bbox = (
                int(zone.x1 * frame_width),
                int(zone.y1 * frame_height),
                int(zone.x2 * frame_width),
                int(zone.y2 * frame_height),
            )
            result.append((bbox, "exit", target))

        # Entry zones
        for source, zone in self.get_entry_zones(camera_id):
            bbox = (
                int(zone.x1 * frame_width),
                int(zone.y1 * frame_height),
                int(zone.x2 * frame_width),
                int(zone.y2 * frame_height),
            )
            result.append((bbox, "entry", source))

        return result
