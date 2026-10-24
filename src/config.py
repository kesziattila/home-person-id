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
    # "center": drop a detection whose center is inside (TV, mirror);
    # "contained": only one whose WHOLE box is inside — an object people stand in front of
    # (a coat on a hook: a person's box reaches past the zone, the coat's does not)
    mode: str = "center"

    def __post_init__(self):
        if self.mode not in ("center", "contained"):
            raise ValueError(f"exclusion zone mode must be 'center' or 'contained', got {self.mode!r}")

    def excludes(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """Whether a detection box (normalized 0-1) is excluded by this zone."""
        if self.mode == "contained":
            return self.x1 <= x1 and self.y1 <= y1 and x2 <= self.x2 and y2 <= self.y2
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2


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
    # Entry handover: an unnamed track walking INTO this zone gets the name of the one person whose
    # track on the same camera was lost <= entry_handover_sec before it appeared (0 = off; stairs:
    # TUNING_LOG 2026-10-21). entry_handover_cameras limits it to these cameras (empty = all of the zone)
    entry_handover_sec: float = 0.0
    entry_handover_cameras: list[str] = field(default_factory=list)


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
    # Minimum confidence for detections. Keep below tracking.track_thresh so
    # ByteTrack's second (low-confidence) association stage receives detections.
    confidence_threshold: float = 0.15
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

    # High/low split: detections >= track_thresh can start tracks; lower ones
    # (down to detection.confidence_threshold) only keep confirmed tracks alive
    track_thresh: float = 0.5
    track_buffer: int = 30
    # Minimum IoU between detection and predicted track box (lower = more lenient)
    match_thresh: float = 0.3
    # A low-confidence box covered by a stage-1-matched box at least this much (IoA)
    # is a partial/duplicate detection of that person and may not revive another track
    duplicate_ioa: float = 0.6
    # Seconds before removing stationary track (0 = no timeout, keep forever)
    stationary_timeout: float = 60.0
    # Post-overlap un-swap (ByteTracker._check_swaps): when two tracks separate after
    # overlapping (> unswap_overlap, intersection / smaller box), each collects
    # unswap_samples Re-ID samples while isolated; if they match the OTHER track's
    # pre-overlap appearance better by > unswap_margin (mean cosine), the two swap ids.
    # 0 disables.
    unswap_margin: float = 0.15
    unswap_samples: int = 3
    unswap_overlap: float = 0.15
    # OC-SORT (Cao et al. 2023) on top of ByteTrack's two stages; replay 10-23..10-24 in TUNING_LOG
    # (all three together: -21 % tracks, swaps 3 -> 2, both un-swapped; OCR alone adds a swap).
    # OCM: weight of the motion-direction agreement (direction over velocity_delta_t frames) in
    # the first association; 0 disables
    velocity_weight: float = 0.2
    velocity_delta_t: int = 3
    # OCR: a box left unmatched is tried against each unmatched track's last observed box
    recover_last_observation: bool = True
    # ORU: after missed frames, re-run the Kalman filter along a straight path to the new box
    reupdate_after_occlusion: bool = True
    # Body signature (src/tracking/body_signature.py: proportions + relative height from pose keypoints)
    # after a merge-loss (a track lost its box while overlapping another, both back later): SHADOW ONLY,
    # logged as `body_check` track events with `would_swap`, never swaps. Pre = median of each track's
    # standing isolated samples in the body_window_frames before the merge, post = after both are back
    # (decided at body_post_samples each, or at the window end with >= 3). would_swap: bodies apart
    # >= body_min_apart and straight - crosswise >= body_margin (10-23..24 offline: 8/8 swaps, 0/31 false).
    body_check: bool = True
    body_window_frames: int = 170          # ~30 s at ~5.7 tracker updates/s per camera
    body_post_samples: int = 10
    body_min_apart: float = 0.5
    body_margin: float = 0.2
    # camera -> 7 numbers: expected standing pixel height at a foot point (tools/fit_height_model.py);
    # a camera without one gets no relative-height feature
    body_height_model: dict = field(default_factory=dict)


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
    # Re-ID evidence only for isolated tracks: overlap (intersection / smaller box)
    # with every other box <= this (src/tracking/attribution.py). 0.0 disables.
    max_overlap: float = 0.15
    # Use TensorRT native backend (bypasses PyTorch, lowest memory)
    # Requires TensorRT engine: python tools/convert_reid_to_trt.py
    use_tensorrt_native: bool = False
    # Path to TensorRT engine file (only used when use_tensorrt_native=True)
    trt_model: str = "models/osnet_ain_x1_0.engine"
    # Seconds between cross-camera zone identity propagation checks.
    # When two cameras share a zone and each sees exactly 1 person,
    # identity is transferred from the face-identified track to the unidentified one.
    cross_camera_interval: float = 2.0
    # Seconds between gallery rechecks for unidentified tracks (the cap of the schedule below).
    # Periodically checks unidentified tracks against the shared Re-ID gallery,
    # so identification on one camera propagates to others even without zone overlap.
    gallery_recheck_interval: float = 10.0
    # Young tracks are rechecked more often: interval = clip(age_factor * track age,
    # min_interval, gallery_recheck_interval). Offline 2026-10-21 (3594 replayed tracks,
    # 10-17..10-21): 0.25 / 0.5 s named 139 more tracks than fixed 10 s, recheck median 60 s -> 5 s,
    # 0 new contradicted names (fixed 2 s: 119 more, 7 contradicted). 0 = fixed interval
    gallery_recheck_age_factor: float = 0.25
    gallery_recheck_min_interval: float = 0.5
    # Minimum seconds between shared-gallery embedding updates of one identified track
    # (each update runs grayscale + multi-face checks; gallery keeps only gallery_size)
    gallery_update_interval_sec: float = 1.0
    # A named track feeds the shared gallery only if its Re-ID sample resembles the
    # person's face-confirmed appearance (agreed by face on two different tracks):
    # cosine >= this (src/tracking/attribution.PersonAppearance)
    anchor_similarity: float = 0.55
    # Feed the gallery (and the face-confirmed appearance anchors) only from boxes whose pose
    # skeleton stands (attribution.posture; box-only detectors are not gated). Offline 2026-10-23
    # (captures 10-18..23, 2564 face-labelled boxes, gallery >= 1 h earlier, sitting queries,
    # 0.70): all postures 46 % right / 12 % wrong names, standing only 28 % / 0.7 % — on the sofa
    # the crop is mostly sofa, and Ati and Judit in white tops matched each other at 0.73-0.76
    gallery_needs_standing: bool = True
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
    # Minimum lead of the best person over the best OTHER person (top1 - top2)
    # required to accept a match. 0 = threshold only.
    min_margin: float = 0.0
    # A single attempt this strong (top1 >= strong_threshold, margin >= strong_min_margin)
    # names the track at once instead of FACE_CONSECUTIVE_MATCHES accepted attempts.
    # 0 = off (always wait for the streak).
    strong_threshold: float = 0.0
    strong_min_margin: float = 0.15
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
    max_per_camera: int = 50
    # Face quality model (eDifFIQA-tiny ONNX, CPU): one learned score of how
    # usable a face is for recognition — replaced the hand-made sharpness /
    # noise / frontalness / brightness metrics (2026-10-18)
    quality_model: str = "models/fiqa/ediffiqa_tiny_jun2024.onnx"
    # Below this the face is not saved (phone-occluded 0.22-0.37, faded 0.11-0.28)
    min_quality_score: float = 0.3
    # From this the face is marked "Learning" (good enough to enroll): usable
    # live crops 0.29-0.70 (low end = looking-down heads), enrolled gallery
    # faces 0.45-0.68, grainy dawn crops up to 0.45
    learning_min_quality: float = 0.45
    # Minimum face size (pixels, min side of the detected face box — not the
    # margin crop, which is ~1.6x larger and let 44 px faces through until
    # 2026-10-18). The quality model upscales everything to 112x112 and cannot
    # see resolution; enrolled faces today were ~70-200 px
    min_face_size: int = 64
    # Minimum similarity threshold to have a best match (below face threshold but reasonable)
    min_match_score: float = 0.3
    # Embedding similarity threshold for diversity check against other tracks'
    # suggestions (a track keeps ONE suggestion, its best crop; a better crop
    # archives the previous one)
    embedding_similarity_threshold: float = 0.7
    # Days to keep unidentified faces before cleanup
    retention_days: int = 30
    # Queue size for background processing
    queue_size: int = 100
    # Enrolment value of a suggestion (UI "Recommended +N" / "Risky"). 2026-10-19 offline on
    # 150 labelled candidates: coverage predicted the real gain of enrolling far better than
    # quality (Spearman 0.71 vs 0.14). Coverage = current suggestions of OTHER tracks at
    # least this similar (cosine) — the recurring situation the face stands for
    enrol_coverage_similarity: float = 0.5
    # "Recommended" from this coverage (and the closest person leading by enrol_min_margin):
    # 92 / 150 marked, 91 % helped, 89 % of the total gain (quality >= 0.45: 18 %).
    # Calibrated on 20 crops per track; the live list holds one per track — re-check
    enrol_min_coverage: int = 5
    # Below this lead of the closest person over the runner-up the suggestion is "Risky"
    # (the one seriously harmful candidate: Judit face 0.17 vs Virag 0.20)
    enrol_min_margin: float = 0.10
    # "Recommended" also needs this face quality: blurred/grainy faces around the 0.45
    # learning threshold still helped a little offline, but degraded embeddings resemble
    # each other (a junk crop named 10 Ati faces Virag) and a stranger's blurred face could
    # match them — a gallery image should be a good picture too (2026-10-19, #1399 at 0.45)
    enrol_min_quality: float = 0.50


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
    max_bytes: int = 20_000_000  # Rotate log file when it exceeds this size
    backup_count: int = 5  # Number of rotated files to keep (app.log.1 .. .5)


@dataclass
class VLMConfig:
    """VLM (Visual Language Model) configuration for scene understanding."""

    enabled: bool = False
    # Base URL of the OpenAI-compatible server (e.g. llama.cpp: llama-server --port 8080)
    url: str = "http://localhost:8080"
    model: str = "qwen3-vl"
    timeout_sec: float = 10.0
    # Maximum tokens in response (keep low: JSON outputs are ~15-25 tokens)
    max_tokens: int = 30
    # Resize images so the longest edge does not exceed this value before sending.
    # Reduces input tokens and network transfer. Set to 0 to disable resizing.
    max_image_size: int = 512
    # Run activity/appearance analysis on unidentified tracks
    analyze_unidentified: bool = True
    # Run periodic activity analysis per active track
    analyze_activity: bool = True
    # Minimum seconds between VLM calls per global_track_id
    activity_interval_sec: float = 10.0
    # Enable periodic house overview (camera scenes + aggregated summary)
    house_overview: bool = True
    # Seconds between house overview generations
    overview_interval_sec: float = 30.0
    # Resize overview frames to this longest-edge before sending (0 = use max_image_size)
    overview_image_size: int = 256
    # Max tokens for the overview response (must cover summary + N camera scenes)
    overview_max_tokens: int = 150


@dataclass
class APIConfig:
    """API server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    # Merge/hide movement-timeline zone stays shorter than this many seconds
    timeline_min_stay_sec: float = 5.0


@dataclass
class MetricsConfig:
    """On-disk metrics for offline threshold tuning (tools/analyze_face_scores.py)."""

    # Log per-person face similarity of every face attempt (JSONL, one file per UTC day)
    face_scores_enabled: bool = True
    path: str = "data/metrics"
    # Minimum seconds between logged attempts of the same track (bounds volume)
    face_scores_min_interval_sec: float = 1.0
    # Log ByteTrack births / revivals / suppressed duplicates with nearest tracks
    # (explains track fragmentation; tools/analyze_track_events.py)
    track_events_enabled: bool = True
    # Log every tracker input (detections per processed frame) to <path>/detections/
    # so tools/replay_tracking.py can re-run ByteTrack offline with other settings
    detection_log_enabled: bool = True
    # Include Re-ID embeddings (float16) for appearance-association experiments
    detection_log_reid: bool = True
    # ~100-300 MB per active day with Re-ID: shorter than other metrics
    detection_log_retention_days: int = 7
    # Full frames of logged records while detections/capture_until is in the future
    # (scripts/remote/capture_frames.sh; tools/redetect.py). Hard size cap; 0 disables
    frame_capture_max_gb: float = 20.0
    # ~0.5 MB per frame, ~10 frames/s with two busy cameras: keep only a few days
    frame_capture_retention_days: int = 3
    # Event capture (src/tracking/event_capture.py): downscaled frames of the last
    # event_capture_pre_sec in RAM per camera; an un-swap, a merge (track turns uncertain) or
    # 3+ live tracks saves the pre-roll + event_capture_post_sec as JPEGs (same layout as the
    # manual capture, for tools/redetect.py). Runtime off: file detections/event_capture_off
    event_capture_enabled: bool = True
    event_capture_scale: float = 0.5
    event_capture_pre_sec: float = 5.0
    event_capture_post_sec: float = 10.0
    event_capture_max_event_sec: float = 30.0
    event_capture_cooldown_sec: float = 60.0
    event_capture_max_gb: float = 5.0
    # Delete daily metric files older than this
    retention_days: int = 30
    # Identity audit samples (annotated frame + full-res track crops) for offline
    # review of what the system labelled (skill review-identities)
    audit_enabled: bool = True
    # Seconds between periodic samples per camera while tracks are in view
    audit_interval_sec: float = 60.0
    # A label change samples immediately, but at most this often per camera
    audit_change_min_interval_sec: float = 5.0
    # Annotated frame width (crops stay full resolution, capped in height)
    audit_max_width: int = 1280
    audit_crop_max_height: int = 640
    # Images are large (~0.3-1 MB per sample): keep them shorter than other metrics
    audit_retention_days: int = 7
    # Age samples, person gallery ages and age vetoes (<path>/age/, age.enabled only)
    age_log_enabled: bool = True
    # Measured times of every profiled module (profiler.measure) to <path>/perf/,
    # one record per perf_interval_sec (~3 KB; ~4 MB/day at 60 s)
    perf_log_enabled: bool = True
    perf_interval_sec: float = 60.0
    # One record per ended track: seconds from track start to first name / first face, method,
    # name changes, revocations (<path>/naming/, tools/analyze_naming.py)
    naming_log_enabled: bool = True
    # One record per Re-ID gallery sample a track buffers: the box's overlap with live / recently
    # lost neighbours at that moment (<path>/reid_feed/; why a cross-person sample got in)
    reid_feed_log_enabled: bool = True
    # One record per gallery recheck: every person's Re-ID gallery score, the best, matched, the
    # track's current name and streak (<path>/gallery_scores/; gallery threshold / margin tuning)
    gallery_scores_log_enabled: bool = True


@dataclass
class AgeConfig:
    """Apparent-age veto on naming (src/recognition/age_gate.py, MiVOLO).

    Each person's age is estimated from their face gallery images; a track's age is
    the median of a few face+body samples. A track is never named as a person whose
    gallery age differs from the track age by more than max_ratio (either way) — no
    hand-made age classes, it follows whoever is in the gallery.
    """

    enabled: bool = False
    # MiVOLO v2 d1_384: .onnx (ONNX Runtime, CPU) or .engine (TensorRT FP16 on the Jetson)
    model: str = "models/mivolo_v2/mivolo_d1_384.onnx"
    # 2026-10-19 on 27 hand-labelled live tracks: x2.5 after 3 samples vetoed the true
    # person 0 / 428 times and the child's name on all 17 adult tracks (x2.0: 4 % wrong)
    max_ratio: float = 2.5
    # Samples before a track's age may veto (single samples vary: child 4-45 y)
    min_samples: int = 3
    # Samples kept per track: the newest ones (rolling), so the age follows whoever the box
    # holds now (2026-10-23: a ByteTrack switch left Judit's 7 samples on Virag's box)
    max_samples: int = 7
    # Seconds between samples of one track (~55 ms each on the Jetson)
    sample_interval_sec: float = 1.0
    # Seconds between samples once max_samples are held (keeps the window fresh, cheaply)
    refresh_interval_sec: float = 5.0
    # Gallery images per person used for the person's age (median)
    gallery_images_per_person: int = 10
    # Sample only while the box's skeleton stands (attribution.posture; unknown posture counts
    # as standing): MiVOLO misjudges sitting/lying people, face and body alike (offline
    # 2026-10-23: x2.5 misses 35 % of sitting/lying face samples vs 3 % standing; live 18:02
    # Judit half-lying on the sofa: 126 body-only samples 7-28 y, face name revoked)
    needs_standing: bool = True
    # Veto face-confirmed names too. Off: a face already tells the child from the adults; of the
    # 23 face vetoes 10-20..10-23 ~14 were wrong (stale samples, sitting/lying) and the 6 right
    # ones were another person's face on the box (attribution: skeleton_face, FACE_BOX_MARGIN)
    # or a merged box. Gallery/handover names (appearance) stay vetoed
    veto_face: bool = False


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
    vlm: VLMConfig = field(default_factory=VLMConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    age: AgeConfig = field(default_factory=AgeConfig)

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
                entry_handover_sec=float(zone_data.get("entry_handover_sec", 0.0)),
                entry_handover_cameras=list(zone_data.get("entry_handover_cameras", [])),
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
    vlm = VLMConfig(**data.get("vlm", {}))
    metrics = MetricsConfig(**data.get("metrics", {}))
    age = AgeConfig(**data.get("age", {}))

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
        vlm=vlm,
        metrics=metrics,
        age=age,
    )
