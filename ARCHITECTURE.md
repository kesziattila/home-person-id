# Home Person ID - Architecture Document

This document describes the system architecture for use in future LLM prompts and development sessions.

## System Purpose

Identify and track persons in a home environment using multiple RTSP cameras. The system should:
1. Detect when a person appears on any camera
2. Track them continuously within each camera view
3. Maintain identity as they move between cameras
4. Identify known persons using face recognition
5. Use Re-ID to maintain identity when face isn't visible

## Processing Pipeline

The system uses a decoupled, multi-threaded pipeline to maximize throughput and minimize latency, especially on resource-constrained hardware like the NVIDIA Jetson. Performance profiling is integrated to monitor bottlenecks.

### 1. Frame Ingestion (Stream Thread)
- `RTSPClient` runs a background thread for each camera.
- It continuously `grabs` frames from the RTSP stream to prevent buffer buildup.
- It `retrieves` and decodes frames at the target FPS, placing them in a `frame_queue`.

### 2. Processing Orchestration (Main Thread)
- The main `run` loop collects frames from all camera clients.
- It performs **Motion Detection** (fast, CPU-based) for each frame.
    - **Optimization**: To handle high-resolution (1080p+) streams efficiently on Jetson, frames are downscaled (default: 360p) before background subtraction. This reduces the pixel count by ~90% while maintaining detection accuracy.
- If motion is detected OR there are active tracks on that camera:
    - It creates an `InferenceTask` and pushes it to the `InferenceQueue`.
- If no motion/tracks:
    - it updates the **Preview Buffer** with the raw frame and skips expensive ML.
- It also polls the `ResultQueue` for completed inference results and processes them (tracking, cross-camera coordination).
- **Performance Monitoring**: A global `profiler` measures execution time of key blocks and detects if the main loop is blocked (>500ms).

### 3. Heavy Inference (Inference Worker Thread)
- A dedicated background thread monitors the `InferenceQueue`.
- **Person Detection**: Runs YOLOv8 on the frame.
- **Batch Re-ID Extraction**: For all detections in the frame, it extracts Re-ID embeddings in a single batch. This offloads significant compute from the main thread.
- It pushes the detections and pre-computed embeddings back to the `ResultQueue`.

### 4. Tracking & Recognition (Main Thread)
- When an `InferenceResult` is received:
    - **Local Tracking**: ByteTrack updates track positions using detections and pre-computed Re-ID embeddings.
    - **Global Tracking**: `GlobalTrackManager` updates cross-camera state, using pre-computed Re-ID for handover and reappearance matching.
    - **Identity Linker**: Periodically runs **Face Recognition** to confirm or update identity. It uses pre-computed Re-ID embeddings for gallery updates, avoiding redundant extraction.
- Finally, it updates the **Preview Buffer** with full metadata (bounding boxes, names).

### 5. Web UI & Hardware Acceleration (API Thread)
- The **APIServer** (FastAPI) provides MJPEG streaming.
- **Optimization**: On Jetson, hardware-accelerated JPEG encoding via `nvJPEG` is used to reduce CPU load when generating the MJPEG stream.

## Component Details

### 1. Stream Management (`src/stream/`)

**RTSPClient** (`rtsp_client.py`)
- Connects to RTSP camera stream
- Reads frames in background thread
- Buffers frames (max 2) to prevent memory issues
- Handles reconnection on failure
- Rate limits to target FPS

**StreamManager** (`manager.py`)
- Manages multiple RTSPClient instances
- Provides iterator over frames from all cameras
- Handles camera lifecycle (start/stop)

### 2. Detection (`src/detection/`)

**MotionDetector** (`motion_detector.py`)
- Uses OpenCV MOG2 background subtraction
- Acts as processing gate (skip expensive ML when no motion)
- **Downscaling**: Automatically resizes high-resolution frames to `processing_height` (e.g., 360px) to ensure low CPU usage on edge devices.
- Implements cooldown period after motion stops
- One instance per camera (via MotionDetectorManager)
- **Stationary tracking**: When active tracks exist but no motion detected,
  processing continues to maintain tracks (marked as "stationary")

**PersonDetector** (`person_detector.py`)
- Wraps YOLOv8 (ultralytics library)
- Detects only persons (class 0)
- Returns list of Detection objects with bbox, confidence
- Lazy loads model on first use

### 3. Tracking (`src/tracking/`)

**LocalTrack / GlobalTrack** (`track.py`)
- LocalTrack: Track within single camera (managed by ByteTrack)
- GlobalTrack: Cross-camera identity (managed by GlobalTrackManager)
- Both store bbox, confidence, timestamps, embeddings

**ByteTracker** (`byte_tracker.py`)
- Single-camera multi-object tracker
- Uses Kalman filter for motion prediction
- IoU-based detection-to-track association
- Two-stage matching (high conf first, then low conf)
- Maintains track state (new, tracked, lost, removed)

**GlobalTrackManager** (`global_tracker.py`)
- Coordinates tracks across all cameras
- Creates global tracks from confirmed local tracks
- Handles camera handover (overlapping views)
- Uses Re-ID for cross-camera matching (non-overlapping views)
- Manages track lifecycle and cleanup

**HandoverManager** (`handover.py`)
- Defines camera overlap zones
- Checks if tracks exit/enter through zones
- Provides zone visualization for debugging

### 4. Recognition (`src/recognition/`)

**FaceRecognizer** (`face_recognizer.py`)
- Uses InsightFace library
- Detects faces and extracts 512-dim embeddings
- Compares embeddings using cosine similarity
- Finds best match from gallery of known persons

**ReIDExtractor** (`reid_extractor.py`)
- Uses OSNet model (torchreid library)
- Extracts 512-dim appearance features from person crops
- Computes quality score based on crop size, aspect ratio, blur
- Supports batch extraction for efficiency. Used in `InferenceWorker` to offload work from the main thread.

**EmbeddingGallery** (in `reid_extractor.py`)
- Stores multiple embeddings per track
- Uses median similarity for stable matching
- Automatically removes oldest when over capacity

**IdentityLinker** (`identity_linker.py`)
- Central component linking recognition to tracks
- Manages TrackIdentityState for each global track
- Runs face recognition on tracks periodically
- Updates Re-ID gallery for cross-camera matching
- Implements consecutive match requirement for stability
- Transfers identity during camera handover
- **Testability**: Supports dependency injection of ML models and repository.

### 5. Database (`src/database/`)

**Models** (`models.py`)
- Person: Known persons (name, created_at, is_active)
- FaceEmbedding: Face embeddings per person (512-dim blob)
- Track: Global tracks (id, person_id, embeddings, status)
- TrackSighting: Track appearances on cameras
- Event: Activity log (track_created, person_identified, etc.)
- Camera, CameraOverlap: Configuration storage

**Repository** (`repository.py`)
- Handles all SQLite/SQLAlchemy interactions
- **Testability**: Supports `:memory:` databases for isolated unit testing.
- CRUD operations for all models
- Embedding serialization (numpy <-> blob)
- Cleanup operations for old data

## Testing Strategy

The system uses a **Scenario-Based Testing Framework** to verify complex tracking logic without requiring real cameras or ML models.

### Key Testing Pillars

1.  **In-Memory Database**: Isolated SQLite databases for every test run to ensure no side effects.
2.  **ML Dependency Injection**: Mocked `FaceRecognizer` and `ReIDExtractor` allow testing identification logic without the 1GB+ model overhead.
3.  **Logic-Only Frame Processing**: `GlobalTrackManager` can process "virtual" frames (passing `frame=None`), enabling simulation of track movement across cameras via bounding boxes alone.
4.  **Scenario Test Base**: `GlobalTrackerScenarioTest` (in `tests/scenario_base.py`) provides helpers to:
    - Create local tracks with specific coordinates.
    - Simulate track appearances and disappearances on different cameras.
    - Assert on global track outcomes (handovers, Re-ID matches, identification).

See [TESTING_SCENARIOS.md](docs/TESTING_SCENARIOS.md) for detailed examples and guide.

### 6. Utils (`src/utils/`)

**Profiler** (`profiler.py`)
- High-precision timing using `time.perf_counter()`
- Measures average, max, and total execution time per module/block
- Tracks call counts
- Reports statistics every 10 seconds to logger
- Thread-safe using locks
- Controlled via `--perf-report` command line argument

### 7. Configuration (`src/config.py`)

Dataclasses for all configuration sections:
- CameraConfig, CameraTopologyConfig
- MotionConfig, TrackingConfig
- ReIDConfig, FaceRecognitionConfig
- MQTTConfig, DatabaseConfig, SnapshotConfig, APIConfig

Loaded from YAML file via `load_config()`.

### 7. Visualization & Web UI (`src/visualization/`, `src/api/`, `src/static/`)

**PreviewBuffer** (`visualization/preview.py`)
- Thread-safe buffer for latest frames and metadata per camera
- Used to decouple main processing loop from web streaming
- Stores copies of frames to prevent mutation issues

**Visualizer** (`visualization/preview.py`)
- Shared drawing logic for annotations
- Uses `TrackRenderer` from `src/preview.py` for consistent look and feel

**APIServer** (`api/server.py`)
- FastAPI-based web server
- Runs in a separate daemon thread
- Provides MJPEG streaming endpoints (`/api/v1/stream/{camera_id}`)
- Serves static dashboard (`index.html`)

**Web Dashboard** (`static/index.html`)
- Tailwind CSS based responsive UI
- Auto-discovers active cameras
- Displays real-time annotated streams

## Data Flow

```
1. StreamManager.get_frames()
   └── Yields Frame(image, timestamp, camera_id) from each camera

2. MotionDetector.detect(frame)
   └── Returns MotionResult(has_motion, motion_ratio)
   └── If no motion: skip to next frame

3. PersonDetector.detect(frame)
   └── Returns DetectionResult with list of Detection(bbox, confidence)
   └── If no detections: skip to next frame

4. ByteTracker.update(detections, frame)
   └── Returns TrackingResult(tracks, new_track_ids, lost_track_ids)
   └── tracks = list of confirmed LocalTrack objects

5. GlobalTrackManager.process_local_tracks(...)
   └── Creates/updates GlobalTrack for each LocalTrack
   └── Checks for camera handover
   └── Checks for Re-ID match with lost tracks
   └── Returns GlobalTrackingResult

6. IdentityLinker.process_track(global_track_id, frame, crop, bbox)
   └── Runs face recognition if due
   └── Updates Re-ID gallery
   └── Returns IdentificationResult

7. PreviewBuffer.update(camera_id, frame, metadata)
   └── Pushes annotated frame data to buffer for API server
```

## Key Algorithms

### ByteTrack Two-Stage Matching

```
1. Predict new positions of existing tracks (Kalman filter)
2. Split detections: high confidence (>0.5) and low confidence
3. First association:
   - Match high-conf detections to all tracks using IoU
   - Solve assignment problem (Hungarian algorithm)
4. Second association:
   - Match low-conf detections to remaining tracks
   - Helps maintain tracks through occlusions
5. Create new tracks for unmatched high-conf detections
6. Mark unmatched tracks as lost
```

### Re-ID Stable Matching

```
1. For each track, maintain gallery of N embeddings
2. When matching:
   - Compare query against ALL gallery embeddings
   - Return MEDIAN similarity (not max or mean)
3. Track consecutive matches:
   - If same person matched M times in a row → confirmed
   - If different person → reset counter
4. Quality gating:
   - Skip crops < 100px height
   - Skip heavily occluded (< 50% visible)
   - Skip motion blurred
```

### Camera Handover

```
1. Define overlap zones in config (normalized coordinates)
2. When track exits cam1's overlap zone:
   - Create PendingHandover with Re-ID embedding
   - Mark track as "lost"
3. When new track enters cam2's overlap zone:
   - Check pending handovers from connected cameras
   - If time window OK and Re-ID matches → link tracks
   - Else → create new global track
```

## State Management

### Track States
- NEW: Just created, not yet confirmed (< 3 hits)
- TRACKED: Actively being tracked (with motion)
- TRACKED (stationary): Actively tracked but no motion detected
  - Displayed with yellow color in preview
  - Shown with stationary duration counter
  - Removed after 60 seconds of no detection
- LOST: No detection match, but still in memory
- REMOVED: Marked for deletion

### Identity States (per GlobalTrack)
- Unidentified: person_id = None
- Tentative: candidate_person_id set, consecutive_matches < threshold
- Confirmed: person_id set, identification_confidence > 0

## Configuration Tuning

### For Better Face Recognition
- `face_recognition.min_face_size`: Increase if false matches (require larger faces)
- `face_recognition.similarity_threshold`: Increase if false positives
- `face_recognition.detection_interval`: Decrease for faster identification

### For Better Re-ID Stability
- `reid.min_consecutive_matches`: Increase if matches flicker (default 3)
- `reid.gallery_size`: Increase for more stable matching (default 10)
- `reid.similarity_threshold`: Decrease if too many misses (default 0.65)
- `reid.min_visibility`: Increase if bad crops cause issues (default 0.5)

### For Better Tracking
- `tracking.track_thresh`: Lower to detect more (may increase false positives)
- `tracking.track_buffer`: Increase to keep lost tracks longer (default 60)
- `tracking.match_thresh`: IoU threshold - lower for fast movement (default 0.5)

### For False Positive Reduction
- **Exclusion zones**: Define areas to ignore (e.g., coat racks, mirrors)
  ```yaml
  cameras:
    - id: living_room
      exclusion_zones:
        - x1: 0.85  # Right side of frame
          y1: 0.0
          x2: 1.0
          y2: 0.8
  ```
- **Static filter**: Ignores detections that don't move
  - `tracking.static_filter_enabled`: Enable/disable (default true)
  - `tracking.static_filter_seconds`: Seconds before filtering (default 10)
  - `tracking.min_movement_threshold`: Min movement to be "moving" (default 0.02)

### Preview Commands
- `--show-zones`: Display exclusion zones as red rectangles

## Known Limitations

1. **Face recognition requires clear frontal faces**
   - Works best when person looks at camera
   - Fails with back of head, profile views

2. **Re-ID struggles with similar clothing**
   - Family members in similar outfits may be confused
   - Relies on face recognition to disambiguate

3. **Single-threaded processing**
   - Cameras processed sequentially
   - Could be parallelized for better performance

4. **No TensorRT optimization yet**
   - Models run with PyTorch/ONNX
   - Phase 5 planned for Jetson optimization

## Future Development Notes

### Phase 4: MQTT Integration
- Add `src/events/mqtt_publisher.py`
- Publish events to Home Assistant topics
- Topics: track/new, track/identified, track/handover, occupancy

### Phase 5: API & Optimization
- [x] Add `src/api/server.py` with FastAPI
- [x] REST endpoints for cameras and streaming
- [ ] Export models to TensorRT for Jetson

### Phase 6: Web UI
- [x] Simple web interface for monitoring
- [x] Live camera preview with annotations
- [ ] Event history viewer
