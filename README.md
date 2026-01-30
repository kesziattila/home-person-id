# Home Person ID System

A Python-based person identification system for home environments that processes RTSP camera streams, tracks persons across multiple cameras, identifies known persons via face recognition, and maintains identity using Re-ID when faces aren't visible.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RTSP Camera Streams                                │
│                    (front_door, hallway, living_room)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Stream Manager (per camera)                          │
│                    - Threaded RTSP readers                                   │
│                    - Frame buffering (grab/retrieve pattern)                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Motion Detection Gate (MOG2)                            │
│                    - Skip processing when no motion                          │
│                    - ~90% compute savings when idle                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │ (only if motion detected)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Person Detection (YOLOv8-nano)                          │
│                    - Detects persons in frame                                │
│                    - Returns bounding boxes + confidence                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Single-Camera Tracking (ByteTrack)                        │
│                    - Kalman filter motion prediction                         │
│                    - IoU-based detection association                         │
│                    - Maintains local track IDs per camera                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Global Track Manager                                    │
│                    - Cross-camera track coordination                         │
│                    - Camera handover (overlapping views)                     │
│                    - Re-ID matching (non-overlapping views)                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Identity Linker                                       │
│                    - Face Recognition (InsightFace)                          │
│                    - Re-ID Features (OSNet)                                  │
│                    - Gallery-based stable matching                           │
│                    - Links tracks to known persons                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SQLite Database                                      │
│                    - Persons, face embeddings                                │
│                    - Tracks, events, sightings                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Face Recognition + Tracking Hybrid
- **Face recognition is the source of truth** for identity
- **Re-ID maintains identity** when face isn't visible
- **Tracking prevents redundant identification** (identify once, track continuously)

### 2. Re-ID Stability Improvements
Based on previous experience with unstable Re-ID matching:
- **Gallery-based matching**: Store 10 embeddings per person, keyed by person name
- **Multi-person safety**: Skip Re-ID when multiple persons in frame to avoid confusion
- **Quality gating**: Skip low-quality crops (small, blurry, occluded)
- **Face as anchor**: Only face-identified tracks are stored in Re-ID gallery
- **Persistent gallery**: Gallery entries persist across matches (not deleted on match)

### 3. Motion Detection Gate
- Skip expensive ML processing when no activity
- ~95% of frames in typical home have no motion
- Reduces power consumption on edge devices

### 4. Debug Image Saving
Debug images are saved to `data/debug/reid/` for troubleshooting:
- `face_recognized_*.jpg` - When face is recognized (with bounding box)
- `gallery_stored_*.jpg` - When person leaves and Re-ID is stored
- `reid_match_*.jpg` - Side-by-side comparison when Re-ID match found

### 5. Web User Interface
A real-time web dashboard is available for monitoring all camera streams:
- **Live Streams**: High-resolution MJPEG streaming with real-time annotations.
- **Unified View**: See all cameras in a single responsive grid.
- **Detailed Metadata**: Displays same tracking info as CLI preview (Face/Re-ID confidence).
- **Zero-Config**: Automatically discovers cameras from system configuration.
- **Person Management**: Add, edit, and delete known persons with face image upload.
- **Unidentified Faces**: Review faces that fall below recognition threshold, assign to persons or dismiss.

Default access: `http://localhost:8000`

### 6. Unidentified Faces Feature
When a face is detected but doesn't match any known person (below similarity threshold):
- Face is automatically captured with quality scoring (size, sharpness, brightness)
- **FFT-based blur detection** rejects motion blur and focus blur (~8-10ms overhead)
- Best-match person and confidence are recorded for quick review
- Diversity check prevents storing duplicate faces
- Per-camera limits keep storage manageable (configurable)
- Web UI provides one-click assignment to existing persons

## Testing

To ensure the stability of tracking logic, the system includes a scenario-based testing framework.

### Running Scenario Tests

These tests simulate person movement across cameras using in-memory databases and mocked ML models, making them fast and reliable.

```bash
# Run all scenario tests
PYTHONPATH=. python3 tests/test_global_tracking_scenarios.py
```

For more details on how to write and run tests, see [TESTING_SCENARIOS.md](docs/TESTING_SCENARIOS.md).

## Project Structure

```
home-person-id/
├── config/
│   └── config.yaml              # Main configuration
├── src/
│   ├── main.py                  # Entry point, main processing loop
│   ├── cli.py                   # CLI commands
│   ├── config.py                # Configuration dataclasses
│   ├── api/
│   │   └── server.py            # FastAPI web server
│   ├── visualization/
│   │   └── preview.py           # Shared visualization and frame buffering
│   ├── static/
│   │   └── index.html           # Web UI dashboard
│   ├── stream/
│   │   ├── rtsp_client.py       # RTSP camera reader (threaded)
│   │   └── manager.py           # Multi-camera stream manager
│   ├── detection/
│   │   ├── motion_detector.py   # MOG2 background subtraction
│   │   └── person_detector.py   # YOLOv8 person detection
│   ├── tracking/
│   │   ├── track.py             # Track data classes (LocalTrack, GlobalTrack)
│   │   ├── byte_tracker.py      # ByteTrack single-camera tracker
│   │   ├── global_tracker.py    # Cross-camera track manager
│   │   └── handover.py          # Camera overlap zone logic
│   ├── recognition/
│   │   ├── face_recognizer.py   # InsightFace face detection/recognition
│   │   ├── reid_extractor.py    # OSNet Re-ID feature extraction
│   │   ├── reid_gallery.py      # Shared Re-ID gallery manager
│   │   ├── identity_linker.py   # Links recognition to tracks
│   │   └── unidentified_face_manager.py  # Captures unidentified faces
│   └── database/
│       ├── models.py            # SQLAlchemy ORM models
│       └── repository.py        # Database CRUD operations
│   ├── utils/
│   │   ├── image_utils.py       # Crop utilities and JPEG encoding
│   │   └── profiler.py          # Performance profiling
├── data/
│   ├── database.db              # SQLite database (created on first run)
│   ├── faces/                   # Reference face images
│   ├── snapshots/               # Event snapshots
│   ├── unidentified_faces/      # Captured unidentified faces for review
│   └── debug/reid/              # Debug images for Re-ID troubleshooting
├── models/                      # ML model weights (downloaded on first run)
├── requirements.txt
├── Dockerfile
└── docker-compose.yaml
```

## Technology Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Person Detection | YOLOv8 (ultralytics) | Detect persons in frames |
| Single-cam Tracking | ByteTrack (custom impl) | Track persons within camera |
| Face Recognition | InsightFace | Detect faces, extract embeddings |
| Person Re-ID | OSNet (torchreid) | Cross-camera appearance matching |
| Database | SQLite + SQLAlchemy | Store persons, tracks, events |
| Streaming | OpenCV | RTSP camera reading |

## Installation & Testing

**Note: Always run the application and CLI tools within the virtual environment and with PYTHONPATH set:**
```bash
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.
# Then run your command, e.g.:
python -m src.main
```

### 1. Install Dependencies

```bash
cd /home/akeszi/hobby/home-person-id

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Cameras

Edit `config/config.yaml` with your camera URLs:

```yaml
cameras:
  - id: front_door
    name: "Front Door"
    rtsp_url: "rtsp://user:pass@192.168.1.100:554/stream1"
    fps: 5
    # Optional: exclusion zones (normalized 0-1 coordinates)
    exclusion_zones:
      - x1: 0.85
        y1: 0.0
        x2: 1.0
        y2: 0.8
```

### 3. Test Camera Connection

```bash
python -m src.cli test-camera front_door
```

### 4. Enroll Known Persons

```bash
# From image files (supports wildcards)
python -m src.cli add-person "John Doe" --images "./photos/john/*.jpg"

# Via live capture (opens camera window, press 'c' to capture)
python -m src.cli add-person "Jane Doe" --camera front_door --capture 5

# Add more faces to existing person
python -m src.cli add-faces --person-id 1 --images "./more_photos/*.jpg"

# Add faces via interactive camera capture
python -m src.cli add-faces --person-id 1 --camera front_door --interactive
```

### 5. Preview Camera with Detection

```bash
# Basic preview
python -m src.cli preview --camera front_door

# With Re-ID enabled (shows Re-ID scores, saves debug images)
python -m src.cli preview --camera front_door --reid

# With scaling (for large monitors)
python -m src.cli preview --camera front_door --scale 0.5

# Show exclusion zones
python -m src.cli preview --camera front_door --show-zones

# All options combined
python -m src.cli preview --camera living_room --reid --scale 0.5 --show-zones
```

Preview display shows:
- Person bounding boxes with track IDs
- Identified persons: `Name (F:0.85)` for face, `Name (R:0.72)` for Re-ID
- Unidentified: `#5 R:0.45` showing track ID and Re-ID confidence
- Zones: Zone name in brackets e.g., `[living_room]`
- Stationary: `S:10s` indicator with orange color for unidentified
- Motion status and track counts
- Re-ID gallery size (when `--reid` enabled)

Press 'q' to quit.

### 6. Run Full System

```bash
python -m src.main --config config/config.yaml
```

To enable performance reporting, use the `--perf-report` flag:
```bash
python -m src.main --perf-report
```

Once running, the Web UI is available at `http://localhost:8000`. You can configure the host and port in `config.yaml`:

```yaml
api:
  host: "0.0.0.0"
  port: 8000
```

### 7. Query Events

```bash
# Recent events
python -m src.cli events --last 1h

# Current occupancy
python -m src.cli occupancy

# Active tracks
python -m src.cli tracks --active

# List known persons
python -m src.cli list-persons
```

## Testing Without Cameras

To test with a video file instead of RTSP:

```yaml
# In config/config.yaml
cameras:
  - id: test
    name: "Test Video"
    rtsp_url: "/path/to/test_video.mp4"  # Local file path works too
    fps: 5
```

## Docker Usage

```bash
# Build and run with GPU support
docker-compose up --build

# Or run with local MQTT broker for testing
docker-compose --profile mqtt up
```

## Configuration Reference

See `config/config.yaml` for all options. Key settings:

| Setting | Description | Default |
|---------|-------------|---------|
| `motion.enabled` | Enable motion gate | true |
| `motion.cooldown_sec` | Continue processing after motion stops | 5 |
| `tracking.track_thresh` | Detection confidence threshold | 0.5 |
| `tracking.match_thresh` | IoU threshold for tracking (lower for fast movement) | 0.3 |
| `reid.similarity_threshold` | Re-ID match threshold | 0.65 |
| `reid.gallery_size` | Embeddings stored per person | 10 |
| `reid.max_reappear_time_sec` | Gallery entry expiration time | 300 |
| `face_recognition.similarity_threshold` | Face match threshold | 0.6 |
| `face_recognition.min_face_size` | Minimum face size (pixels) | 40 |
| `face_recognition.detection_interval` | Frames between face checks | 10 |
| `unidentified_faces.enabled` | Capture unidentified faces | true |
| `unidentified_faces.max_per_camera` | Max faces to keep per camera | 20 |
| `unidentified_faces.min_quality_score` | Min quality score (0-1) | 0.3 |
| `unidentified_faces.min_sharpness_score` | Min sharpness for blur rejection (0-1) | 0.4 |
| `unidentified_faces.min_face_size` | Min face size for capture (pixels) | 60 |
| `unidentified_faces.retention_days` | Days to keep before cleanup | 30 |

## Re-ID Gallery Behavior

The Re-ID system helps identify persons who leave and return:

1. **Face identification required first**: Only face-identified tracks are stored in Re-ID gallery
2. **Gallery keyed by person name**: Same person returning creates same gallery entry
3. **Multi-person safety**: Re-ID is skipped when multiple persons in frame
4. **Persistent entries**: Gallery entries are NOT deleted on successful match (person can leave again)
5. **Time-based expiration**: Entries expire after `max_reappear_time_sec` (default 5 min)
6. **Debug images**: Enable `--reid` flag to save debug images to `data/debug/reid/`

## Future Improvements (Not Yet Implemented)

- [ ] **Phase 4**: MQTT publisher for Home Assistant integration
- [x] **Phase 5**: FastAPI REST server & Web UI
- [x] **Phase 5**: TensorRT optimization for Jetson (YOLO & Re-ID)

## Troubleshooting

### Models not downloading
Models are downloaded on first use. Ensure internet connection and sufficient disk space.

### CUDA out of memory
Reduce batch size or use CPU:
```python
# In person_detector.py, force CPU
self._model.to('cpu')
```

### RTSP connection fails
- Check camera URL format
- Try VLC to verify stream works
- Some cameras need specific URL paths (e.g., `/stream1`, `/h264`)

### Re-ID matches unstable
- Ensure only one person in frame when testing Re-ID
- Check debug images in `data/debug/reid/` to verify quality
- Increase `reid.gallery_size` for more stable matching
- Ensure good lighting for quality crops

### Tracking creates too many IDs
- Lower `tracking.match_thresh` (e.g., 0.2) for fast-moving persons
- Increase `tracking.track_buffer` to keep lost tracks longer
