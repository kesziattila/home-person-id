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
- If motion is detected OR there are active/recently-lost tracks on that camera:
    - It creates an `InferenceTask` and pushes it to the `InferenceQueue`.
    - LOST tracks within the grace period (`reid.global_id_grace_period`, default 60s) keep detection running so ByteTrack can re-detect a briefly lost person.
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
    - **Global Tracking**: `GlobalTrackManager` updates cross-camera state. It treats `global_track_id` as a session-based identifier, assigning a new ID if a person is missing for more than a grace period (default 60s), while maintaining person identity via the Re-ID gallery.
    - **Identity Linker**: Periodically runs **Face Recognition** to confirm or update identity. It uses pre-computed Re-ID embeddings for gallery updates and maintains "best guess" identities for unidentified persons.
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
- Handles camera handover (overlapping views) — configurable via `enable_handover`
- Cross-camera zone identity propagation: when two cameras simultaneously see exactly 1 person each in a shared zone, transfers identity from the face-identified track to the unidentified one — configurable via `enable_cross_camera_propagation`
- Uses Re-ID for cross-camera matching within a short grace period
- Assigns new `global_track_id` after the grace period expires
- **Track recovery**: When a track goes LOST, the local-to-global mapping is kept alive so ByteTrack re-detection of the same local ID restores the global track with its identity. Mappings are cleaned up when the track transitions to REMOVED (grace period expired).
- **Detection gate**: `has_active_tracks()` includes LOST tracks within the grace period, preventing detection from stopping for stationary people whose tracks flicker
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
- Captures unidentified faces for manual review (via `UnidentifiedFaceManager`)
- **Testability**: Supports dependency injection of ML models and repository.

**UnidentifiedFaceManager** (`unidentified_face_manager.py`)
- Background thread manager for capturing faces below recognition threshold
- Non-blocking `submit()` method for main thread
- Computes quality score (size, sharpness, brightness)
- **FFT-based blur detection**: Uses frequency domain analysis to detect both motion blur and focus blur (Laplacian variance only catches focus blur). Performance: ~8-10ms per image.
- Diversity check to avoid storing duplicate faces (cosine similarity)
- Per-camera limit enforcement and automatic cleanup
- Stores face crops to disk and metadata to database

### 5. Database (`src/database/`)

**Models** (`models.py`)
- Person: Known persons (name, created_at, is_active)
- FaceEmbedding: Face embeddings per person (512-dim blob)
- Track: Global tracks (id, person_id, embeddings, status)
- TrackSighting: Track appearances on cameras
- Event: Activity log (track_created, person_identified, etc.)
- Camera, CameraOverlap: Configuration storage
- UnidentifiedFace: Faces below recognition threshold for manual review

**Repository** (`repository.py`)
- Handles all SQLite/SQLAlchemy interactions
- **Multi-threading**: Configured with `check_same_thread=False` and WAL mode for safe concurrent access
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

### 6. Frontend (`src/static/`)

The frontend is a modern, single-page application (SPA) built with **Vue.js** and styled with **Tailwind CSS**.

**Framework Choice (Vue.js):**
- **Simplicity & Performance:** Vue was chosen for its gentle learning curve and excellent performance. It's loaded directly from a CDN, avoiding the need for a complex build setup (like Webpack or Vite), which keeps the project simple.
- **Component-Based Architecture:** The UI is broken down into logical, reusable components (e.g., `App.js`, `Streams.js`, `PersonCard.js`), making the code clean, organized, and easy to maintain.
- **Declarative Rendering:** Vue's reactive data binding automatically updates the UI when data changes, eliminating manual DOM manipulation (`innerHTML`) and improving code readability.

**Core Architecture:**
- **`index.html`**: A minimal shell that loads Vue, Tailwind, and the main application script. It contains a single `<div id="app"></div>` where the Vue application is mounted.
- **`app.js`**: The main entry point. It initializes the Vue application, registers all the components, and mounts the root `App` component.
- **`components/`**: This directory contains all the Vue components.
    - **`App.js` (Root Component):** Manages global state, such as the active tab and the visibility of all modals. It acts as the central hub for communication between different parts of the UI.
    - **Tab Components (`Streams.js`, `Persons.js`, etc.):** Each tab in the UI has its own dedicated component, encapsulating all the logic and HTML for that view.
    - **UI Components (`PersonCard.js`, `EventDetailModal.js`, etc.):** Smaller, reusable components used by the tab components to display specific pieces of the UI.
- **`utils/`**: Contains shared JavaScript utility functions, such as `formatters.js`.

**Communication:**
- **Parent-to-Child:** Data is passed down from parent components to child components via **props** (e.g., `App.js` passes the `autoRefresh` setting to the active tab component).
- **Child-to-Parent:** Child components communicate with the parent by **emitting events** (e.g., when a person card is clicked, it emits a `show-person-detail` event, which the `App` component listens for to open the correct modal).

### 7. Utils (`src/utils/`)

**Profiler** (`profiler.py`)
- High-precision timing using `time.perf_counter()`
- Measures average, max, and total execution time per module/block
- Tracks call counts
- Reports statistics every 10 seconds to logger
- Thread-safe using locks
- Controlled via `--perf-report` command line argument

**Image Utils** (`image_utils.py`)
- `crop_with_margin(image, bbox, margin_ratio)`: Crop region with margin, returns (crop, adjusted_bbox)
- `crop_bbox(image, bbox)`: Simple bbox crop with bounds checking
- `crop_face_region(image, bbox, height_ratio)`: Crop upper portion for face detection
- `encode_jpeg(image, quality, use_nvjpeg)`: JPEG encoding with optional hardware acceleration
- Hardware-accelerated encoding via nvJPEG on Jetson

### 8. VLM Scene Understanding (`src/vlm/`)

Optional integration with a local OpenAI-compatible multimodal LLM server (e.g., Qwen3-VL via llama.cpp). Disabled by default; enabled by `vlm.enabled: true` in config.

**VLMClient** (`vlm_client.py`)
- Sends requests to `{url}/v1` using the `openai` SDK with `api_key="dummy"`.
- `_encode_image(image, max_size)`: resizes to longest-edge limit then base64-encodes as JPEG.
- `call(prompt, images, max_tokens, image_max_size)`: accepts per-call overrides for token budget and image size so different call types can be tuned independently.
- `call_text_only(prompt)`: text-only variant (used for testing / non-image calls).

**VLMAnalyzer** (`vlm_analyzer.py`)
- `analyze_person(crops)`: sends one or more person crops, parses `{"activity", "appearance", "gender", "age_group"}` JSON. Profiled as `VLM.analyze_person`.
- `generate_house_overview(person_states, frames, camera_names)`: **single-call** overview.
  - Builds a structured prompt listing cameras by number and name, plus person context (name, zone, activity/gender/age if known).
  - Sends all camera frames as images in a single request with `overview_image_size` resize and `overview_max_tokens` limit.
  - Parses `{"summary": "...", "scenes": {"cam_id": "...", ...}}` JSON. Falls back to raw text as summary if JSON parsing fails.
  - Profiled as `VLM.house_overview`.

**Events produced per cycle:**
- `camera_scene` event per camera (description from `scenes` key, snapshot saved before VLM call)
- `house_overview` event on `global` camera (full `HouseOverview` dict + `camera_snapshot_paths`)
- MQTT publish to `{topic_prefix}/overview`

**Token budget (default):**
- Per-person: `max_tokens=30`, image resized to 512px
- Overview: `overview_max_tokens=150`, images resized to 256px (~95 tokens for 3 cameras)

### 9. Configuration (`src/config.py`)

Dataclasses for all configuration sections:
- CameraConfig, CameraTopologyConfig
- MotionConfig, TrackingConfig
- ReIDConfig, FaceRecognitionConfig
- UnidentifiedFacesConfig (capture faces below threshold for manual review)
- MQTTConfig, DatabaseConfig, SnapshotConfig, APIConfig
- VLMConfig (scene understanding; `overview_image_size` and `overview_max_tokens` tune the single-call house overview)

Loaded from YAML file via `load_config()`.

### 9. Visualization & API (`src/visualization/`, `src/api/`)

**PreviewBuffer** (`visualization/preview.py`)
- Thread-safe buffer for latest frames and metadata per camera
- Used to decouple main processing loop from web streaming
- Stores copies of frames to prevent mutation issues

**Visualizer** (`visualization/preview.py`)
- Shared drawing logic for annotations
- Uses `TrackRenderer` from `src/visualization/render.py` for consistent look and feel

**TrackRenderer** (`visualization/render.py`)
- Centralized rendering of tracks with labels and bounding boxes
- Used by both standalone preview (`preview.py`) and web visualizer
- Implements unified labeling logic (Face, Re-ID, unidentified)
- Handles stationary indicators and zone information in labels

**ZoneRenderer** (`visualization/render.py`)
- Centralized rendering of polygon zones on frames
- Used by standalone preview (`preview.py`)
- Optionally used by web visualizer (currently hidden by default)

**APIServer** (`api/server.py`)
- FastAPI-based web server
- Runs in a separate daemon thread
- Provides MJPEG streaming endpoints (`/api/v1/stream/{camera_id}`)
- Serves static dashboard (`index.html`) and all frontend assets (`/static/*`).
- Integrated with `ZoneManager` to provide zone context in labels

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
   └── Checks for camera handover (lost → reappear)
   └── Checks for Re-ID match with lost tracks
   └── Cross-camera zone identity propagation (simultaneous active tracks)
   └── Returns GlobalTrackingResult

6. IdentityLinker.process_track(global_track_id, frame, crop, bbox)
   └── Runs face recognition if due
   └── Updates Re-ID gallery
   └── Returns IdentificationResult

7. PreviewBuffer.update(camera_id, frame, metadata)
   └── Pushes annotated frame data to buffer for API server

8. VLMAnalyzer (background threads, when vlm.enabled)
   ├── analyze_person(crops) → VLMResult per active track (activity, appearance, gender, age_group)
   └── generate_house_overview(person_states, frames) → HouseOverview
       ├── Single call: all camera frames + person context → {"summary", "scenes"} JSON
       ├── Saves camera_scene events + house_overview event to DB
       └── Publishes to MQTT {topic_prefix}/overview
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

Controlled by `camera_topology.enable_handover` (default: true).

```
1. Define polygon zones in config (normalized coordinates)
2. When track is lost in a zone visible by other cameras:
   - Create PendingHandover with Re-ID embedding
   - Mark track as "lost"
3. When new track appears in the same zone on another camera:
   - Check pending handovers from connected cameras
   - If time window OK and Re-ID matches → link tracks
   - Else → create new global track
```

### Cross-Camera Zone Identity Propagation

Controlled by `camera_topology.enable_cross_camera_propagation` (default: true).
Throttled by `reid.cross_camera_interval` (default: 2 seconds).

```
1. After processing all active tracks, check shared zones
2. For each zone visible on multiple cameras:
   - If camera A sees exactly 1 person AND camera B sees exactly 1 person
   - And one is face-identified while the other is unidentified
   - Transfer identity via handover method (rank 1, upgradeable by reid/face)
3. Update database for the newly identified track
```

## State Management

### Track States
- NEW: Just created, not yet confirmed (< 3 hits)
- TRACKED: Actively being tracked (with motion)
- TRACKED (stationary): Actively tracked but no motion detected
  - Displayed with orange color in preview (`S:10s`)
  - Shown with stationary duration counter
  - Removed after 60 seconds of no detection (if `stationary_timeout` configured)
- LOST: No detection match, but still in memory. Local-to-global mapping is kept alive for ByteTrack recovery. Detection continues running (`has_active_tracks` returns True) within the grace period.
- REMOVED: Marked for deletion. Local-to-global mapping cleaned up.

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
- `tracking.track_buffer`: Increase to keep lost tracks longer (default 30). At 10 FPS: 30=3s, 60=6s, 90=9s.
- `tracking.match_thresh`: IoU threshold — lower for fast movement, higher (0.5-0.6) to reduce flickering for stationary people (default 0.3)

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

4. **TensorRT optimization for YOLO, Face Recognition, and Re-ID**
   - YOLOv8, InsightFace, and OSNet Re-ID models can be optimized with TensorRT for 2-3x speedup on Jetson.
   - The system uses `onnxruntime` with `TensorrtExecutionProvider` for high-performance inference of Face and Re-ID models.
   - Memory limits for TensorRT workspace can be configured per module.
   - All modules (Detection, Face, Re-ID) are warmed up with dummy images on startup to ensure smooth initial processing.

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
- [x] Event history viewer (with clickable filtering, track ID display, keyboard shortcuts)
