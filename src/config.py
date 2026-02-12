"""Configuration loader for Home Person ID system."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ExclusionZone:
    """Exclusion zone where detections should be ignored."""

    x1: float  # Normalized 0-1
    y1: float
    x2: float
    y2: float


@dataclass
class CameraConfig:
    """Configuration for a single camera."""

    id: str
    name: str
    rtsp_url: str
    fps: Optional[int] = 5
    exclusion_zones: list[ExclusionZone] = field(default_factory=list)
    use_nvdec: bool = False  # Use Jetson NVDEC hardware decoder


@dataclass
class CameraOverlap:
    """Configuration for camera overlap zones."""

    cameras: list[str]
    cam1_exit_zone: list[float]  # [x1, y1, x2, y2] normalized
    cam2_entry_zone: list[float]
    max_handover_sec: float = 3.0


@dataclass
class ZonePolygon:
    """Polygon definition for a zone on a specific camera."""

    polygon: list[list[float]]  # [[x1,y1], [x2,y2], ...] normalized 0-1


@dataclass
class ZoneConfig:
    """A room/area zone visible on one or more cameras."""

    name: str
    cameras: dict[str, ZonePolygon]  # camera_id -> polygon
    max_handover_sec: float = 5.0
    exit_destination: Optional[str] = None  # Estimated location when person leaves this zone


@dataclass
class ZonesConfig:
    """All zones configuration."""

    zones: list[ZoneConfig] = field(default_factory=list)


@dataclass
class CameraTopologyConfig:
    """Camera topology configuration."""

    # Use room-based zones for handovers
    use_zones: bool = True
    overlaps: list[CameraOverlap] = field(default_factory=list)
    # Enable handover: when a track is lost on one camera and appears on another
    # in the same zone, link them as the same person.
    enable_handover: bool = True
    # Enable cross-camera identity propagation: when two cameras simultaneously
    # see exactly 1 person each in a shared zone, transfer identity from the
    # face-identified track to the unidentified one.
    enable_cross_camera_propagation: bool = True


@dataclass
class MotionConfig:
    """Motion detection configuration."""

    enabled: bool = True
    history: int = 500
    var_threshold: int = 50
    min_area_ratio: float = 0.01
    cooldown_sec: float = 5.0
    # Process frames at this height for motion detection
    processing_height: int = 360
    # Use CUDA-accelerated motion detection (requires OpenCV with CUDA support)
    use_cuda: bool = False
    # Resize frame on CPU before GPU upload (saves ~25MB GPU memory per camera)
    # Trade-off: slightly higher CPU usage, lower GPU memory
    resize_on_cpu: bool = False


@dataclass
class DetectionConfig:
    """Person detection configuration."""

    enabled: bool = True
    # YOLO model to use (e.g., yolov8n.pt, yolov8s.pt, yolo11n.pt, yolo11s.pt)
    # Smaller models are faster: yolov8n (fastest) < yolov8s < yolov8m < yolov8l
    model: str = "yolov8n.pt"
    # Minimum confidence for detections
    confidence_threshold: float = 0.5
    # NMS IoU threshold (lower = fewer overlapping boxes)
    nms_iou_threshold: float = 0.4
    # Only run detection every N frames (1 = every frame, 2 = every other frame)
    # Higher values reduce CPU usage but may miss fast-moving objects
    frame_skip: int = 1
    # Number of CPU threads for inference (0 = auto)
    num_threads: int = 0
    # Device for YOLO inference: "cuda", "cpu", or None (auto-detect)
    # Use "cpu" to force CPU inference for testing/comparison
    device: Optional[str] = None
    # Use TensorRT native backend (bypasses PyTorch/Ultralytics, lowest memory)
    # Only works with .engine or .trt model files exported with nms=True
    use_tensorrt_native: bool = False
    # Use hardware-accelerated JPEG encoding (pynvjpeg) on Jetson
    use_nvjpeg: bool = False


@dataclass
class TrackingConfig:
    """Single-camera tracking configuration."""

    track_thresh: float = 0.5  # Deprecated: use detection.confidence_threshold
    track_buffer: int = 30
    match_thresh: float = 0.3
    # Seconds before removing stationary track (0 = no timeout, keep forever)
    stationary_timeout: float = 60.0


@dataclass
class ReIDConfig:
    """Cross-camera Re-ID configuration."""

    enabled: bool = True  # Set to False to disable Re-ID and save memory
    # Models: osnet_x1_0 (best), osnet_x0_75, osnet_x0_5, osnet_x0_25 (smallest)
    model: str = "osnet_x1_0"
    similarity_threshold: float = 0.65
    # Max time (seconds) to re-identify a track after it was lost.
    # Longer times allow for long-term tracking but may increase false matches.
    # Note: Global tracker uses a shorter grace period for global_track_id continuity.
    max_reappear_time_sec: float = 3600.0
    # Grace period (seconds) to maintain the same global_track_id after a track is lost.
    # If a person returns within this time, they keep their ID. After this, they get a new ID.
    global_id_grace_period: float = 60.0
    gallery_size: int = 10
    min_consecutive_matches: int = 3
    min_crop_height: int = 100
    min_visibility: float = 0.5
    # Path for temporary crop storage (reduces memory usage)
    crop_cache_path: str = "data/cache/reid_crops"
    # Skip Re-ID for detections with overlapping bounding boxes (IoU > threshold).
    # Helps avoid storing embeddings when people are close together (e.g., parent+child).
    # Set to 0.0 to disable. Recommended: 0.15
    skip_overlapping_iou: float = 0.15
    # Use TensorRT native backend (bypasses PyTorch, lowest memory)
    # Requires TensorRT engine: python tools/convert_reid_to_trt.py
    use_tensorrt_native: bool = False
    # Path to TensorRT engine file (only used when use_tensorrt_native=True)
    trt_model: str = "models/osnet_ain_x1_0.engine"
    # Seconds between cross-camera zone identity propagation checks.
    # When two cameras share a zone and each sees exactly 1 person,
    # identity is transferred from the face-identified track to the unidentified one.
    cross_camera_interval: float = 2.0
    # Seconds between gallery rechecks for unidentified tracks.
    # Periodically checks unidentified tracks against the shared Re-ID gallery,
    # so identification on one camera propagates to others even without zone overlap.
    gallery_recheck_interval: float = 10.0
    # Log track_recovered events to the database.
    # These fire frequently when ByteTrack briefly loses a detection.
    # Disable to reduce event noise.
    log_track_recovery: bool = False


@dataclass
class FaceRecognitionConfig:
    """Face recognition configuration."""

    enabled: bool = True
    model: str = "buffalo_l"
    similarity_threshold: float = 0.6
    min_face_size: int = 80
    # Minimum face size for multi-face detection (used to filter Re-ID crops).
    # Lower than min_face_size because we only need to detect presence, not quality.
    multi_face_min_size: int = 20
    detection_interval: int = 10
    # Interval for re-checking Re-ID identified tracks with face recognition
    # (to confirm identity with primary method). Set higher than detection_interval.
    reid_confirmation_interval: int = 30
    # Detection input size (det_size x det_size). Smaller = less GPU memory but shorter detection range.
    # 640: ~900MB GPU, full frame detection | 480: ~600MB | 320: ~400MB, good for person crops
    det_size: int = 640
    # Use TensorRT acceleration via ONNX Runtime on Jetson (legacy)
    use_tensorrt: bool = False
    # Max memory for TensorRT engine (bytes), 0 = default (usually 1GB)
    trt_max_workspace_size: int = 0
    # Use native TensorRT backend (lower memory, requires pre-converted .engine files)
    use_tensorrt_native: bool = False
    # Path to TensorRT detection engine (det_10g.engine)
    trt_det_model: str = "models/det_10g.engine"
    # Path to TensorRT recognition engine (w600k_r50.engine)
    trt_rec_model: str = "models/w600k_r50.engine"
    # Directory to store registered face images
    faces_dir: str = "data/faces"


@dataclass
class UnidentifiedFacesConfig:
    """Configuration for unidentified faces capture and review."""

    enabled: bool = True
    # Directory to store face images
    images_path: str = "data/unidentified_faces"
    # Maximum number of unidentified faces to keep per camera
    max_per_camera: int = 20
    # Minimum quality score (0-1) to save a face
    min_quality_score: float = 0.3
    # Minimum sharpness score (0-1) to save a face - rejects motion/focus blur
    min_sharpness_score: float = 0.4
    # Minimum face size (pixels) for face bbox width
    min_face_size: int = 60
    # Minimum similarity threshold to have a best match (below face threshold but reasonable)
    min_match_score: float = 0.3
    # Embedding similarity threshold for diversity check (skip if too similar to existing)
    embedding_similarity_threshold: float = 0.7
    # Days to keep unidentified faces before cleanup
    retention_days: int = 30
    # Queue size for background processing
    queue_size: int = 100


@dataclass
class MQTTConfig:
    """MQTT configuration for Home Assistant integration."""

    enabled: bool = True
    broker: str = "localhost"
    port: int = 1883
    username: Optional[str] = None
    password: Optional[str] = None
    topic_prefix: str = "home/person"


@dataclass
class DatabaseConfig:
    """Database configuration."""

    path: str = "data/database.db"
    event_retention_days: int = 30
    archive_tracks_after_hours: int = 24


@dataclass
class SnapshotConfig:
    """Snapshot configuration."""

    enabled: bool = True
    path: str = "data/snapshots"
    save_on_new_track: bool = True
    save_on_identification: bool = True
    save_unknown_only: bool = False


@dataclass
class DebugConfig:
    """Debug configuration for development and diagnostics."""

    # Show Re-ID debug overlay on preview frames
    reid_overlay: bool = False
    # Enable detailed Re-ID decision logging at DEBUG level
    log_reid_details: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    file: Optional[str] = None


@dataclass
class APIConfig:
    """API server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class Config:
    """Main configuration class."""

    cameras: list[CameraConfig] = field(default_factory=list)
    camera_topology: CameraTopologyConfig = field(default_factory=CameraTopologyConfig)
    zones: ZonesConfig = field(default_factory=ZonesConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    reid: ReIDConfig = field(default_factory=ReIDConfig)
    face_recognition: FaceRecognitionConfig = field(default_factory=FaceRecognitionConfig)
    unidentified_faces: UnidentifiedFacesConfig = field(default_factory=UnidentifiedFacesConfig)
    mqtt: MQTTConfig = field(default_factory=MQTTConfig)
    api: APIConfig = field(default_factory=APIConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    snapshots: SnapshotConfig = field(default_factory=SnapshotConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def get_camera(self, camera_id: str) -> Optional[CameraConfig]:
        """Get camera configuration by ID."""
        for camera in self.cameras:
            if camera.id == camera_id:
                return camera
        return None


def load_config(config_path: str | Path) -> Config:
    """Load configuration from YAML file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    # Parse cameras
    cameras = []
    for cam_data in data.get("cameras", []):
        # Parse exclusion zones if present
        zones_data = cam_data.pop("exclusion_zones", [])
        exclusion_zones = []
        for zone in zones_data:
            exclusion_zones.append(ExclusionZone(**zone))
        cameras.append(CameraConfig(**cam_data, exclusion_zones=exclusion_zones))

    # Parse camera topology
    topology_data = data.get("camera_topology", {})
    overlaps = []
    for overlap_data in topology_data.get("overlaps", []):
        overlaps.append(CameraOverlap(**overlap_data))
    camera_topology = CameraTopologyConfig(
        use_zones=topology_data.get("use_zones", True),
        overlaps=overlaps,
        enable_handover=topology_data.get("enable_handover", True),
        enable_cross_camera_propagation=topology_data.get("enable_cross_camera_propagation", True),
    )

    # Parse zones configuration
    zones_data = data.get("zones", [])
    zone_configs = []
    for zone_data in zones_data:
        cameras_dict = {}
        for camera_id, camera_zone_data in zone_data.get("cameras", {}).items():
            cameras_dict[camera_id] = ZonePolygon(polygon=camera_zone_data.get("polygon", []))
        zone_configs.append(
            ZoneConfig(
                name=zone_data.get("name", ""),
                cameras=cameras_dict,
                max_handover_sec=zone_data.get("max_handover_sec", 5.0),
                exit_destination=zone_data.get("exit_destination"),
            )
        )
    zones_config = ZonesConfig(zones=zone_configs)

    # Parse other configs
    motion = MotionConfig(**data.get("motion", {}))
    detection = DetectionConfig(**data.get("detection", {}))
    tracking = TrackingConfig(**data.get("tracking", {}))
    reid = ReIDConfig(**data.get("reid", {}))
    face_recognition = FaceRecognitionConfig(**data.get("face_recognition", {}))
    unidentified_faces = UnidentifiedFacesConfig(**data.get("unidentified_faces", {}))
    mqtt = MQTTConfig(**data.get("mqtt", {}))
    api = APIConfig(**data.get("api", {}))
    database = DatabaseConfig(**data.get("database", {}))
    snapshots = SnapshotConfig(**data.get("snapshots", {}))
    debug = DebugConfig(**data.get("debug", {}))
    logging_config = LoggingConfig(**data.get("logging", {}))

    return Config(
        cameras=cameras,
        camera_topology=camera_topology,
        zones=zones_config,
        motion=motion,
        detection=detection,
        tracking=tracking,
        reid=reid,
        face_recognition=face_recognition,
        unidentified_faces=unidentified_faces,
        mqtt=mqtt,
        api=api,
        database=database,
        snapshots=snapshots,
        debug=debug,
        logging=logging_config,
    )
