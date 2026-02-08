# Scenario-Based Testing Framework

To improve the stability of global tracking, we have introduced a scenario-based testing framework. This allows testing complex tracking logic (like cross-camera handovers and Re-ID reappearance) without requiring live camera streams, real ML models, or a persistent database.

## Key Components

### 1. In-Memory Database
The `Repository` class now supports `":memory:"` as a database path. This creates a fresh, isolated database for each test run.

```python
from src.database.repository import Repository
repo = Repository(":memory:")
```

### 2. Dependency Injection for ML Models
`IdentificationManager` and `IdentityLinker` now accept optional `face_recognizer` and `reid_extractor` instances in their constructors. This allows passing mocked versions that return predictable results without the overhead of loading heavy models.

```python
import numpy as np
from unittest.mock import MagicMock
from src.recognition.identity_linker import IdentityLinker

mock_reid = MagicMock()
mock_reid.extract.return_value = (np.zeros(512), 0.9)
linker = IdentityLinker(
    face_config=face_config, 
    reid_config=reid_config, 
    repository=repo, 
    reid_extractor=mock_reid
)
```

### 3. Logic-Only Frame Processing
`GlobalTrackManager.process_local_tracks` now accepts `frame=None`. When no frame is provided, it uses cached or default frame dimensions for zone calculations and skips operations that strictly require image data (unless they are mocked).

## Creating a New Scenario Test

Inherit from `GlobalTrackerScenarioTest` in `tests/scenario_base.py` to get a pre-configured environment.

### Example: Handover Scenario

```python
def test_handover(self):
    # 1. Setup overlap zones
    overlap = CameraOverlap(cameras=["cam1", "cam2"], ...)
    self.gtm.topology_config.overlaps.append(overlap)
    self.gtm._load_handover_zones()

    # 2. Simulate track on cam1
    lt1 = self.create_local_track(track_id=1, camera_id="cam1", bbox=(100, 100, 200, 200))
    res1 = self.process_frame("cam1", [lt1])
    global_id = res1.new_global_tracks[0]

    # 3. Move to exit zone and lose track
    lt1.bbox = (1800, 100, 1900, 300) 
    self.process_frame("cam1", [lt1])
    self.process_frame("cam1", [], lost_ids=[1])

    # 4. Appear on cam2
    lt2 = self.create_local_track(track_id=2, camera_id="cam2", bbox=(50, 100, 150, 300))
    res2 = self.process_frame("cam2", [lt2])

    # 5. Assert handover success
    self.assertEqual(res2.handovers_completed[0][0], global_id)
```

## Running Tests

```bash
# Global tracking scenarios (handover, Re-ID)
PYTHONPATH=. python3 tests/test_global_tracking_scenarios.py

# Cross-camera zone identity propagation
PYTHONPATH=. python -m pytest tests/test_cross_camera_zone_propagation.py -v
```

## Cross-Camera Zone Propagation Tests

`tests/test_cross_camera_zone_propagation.py` covers zone-based identity propagation between simultaneously active tracks:

- **Propagation**: Face-identified track on one camera propagates identity to unidentified track on another camera in the same zone
- **Single-person constraint**: No propagation when multiple persons on one camera in the zone
- **Already identified**: No change when both tracks are identified
- **Precedence**: Face-identified tracks are not downgraded by handover propagation
- **No zones**: No propagation when zone manager is not configured
- **Throttling**: Propagation respects `reid.cross_camera_interval`
