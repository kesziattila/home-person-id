
import pytest
import numpy as np
from unittest.mock import MagicMock
from src.recognition.unidentified_face_manager import UnidentifiedFaceManager, UnidentifiedFaceTask
from src.config import Config
from src.database.repository import Repository

def test_reproduce_json_serializable_error():
    # Use real in-memory repository to exercise JSON path
    repository = Repository(":memory:")
    
    # Mock config
    config = Config()
    config.unidentified_faces.enabled = True
    config.unidentified_faces.images_path = "data/test_unidentified"
    
    # Initialize manager
    manager = UnidentifiedFaceManager(config.unidentified_faces, repository)
    
    # Create a task with numpy float32 values
    # In the error log: 'quality_score': np.float32(0.52994657)
    task = UnidentifiedFaceTask(
        camera_id="test_cam",
        face_crop=np.zeros((100, 100, 3), dtype=np.uint8),
        embedding=np.random.rand(512).astype(np.float32),
        face_bbox=(10, 10, 120, 60),
        track_id="global_15",
        best_match_person_id=6,
        best_match_score=np.float32(0.5271) # Numpy float32
    )
    
    # We want to test _process_task directly to see if it fails when calling create_event
    # or if we can catch the numpy types in the call to create_event
    
    # Do not mock create_event; use real Repository path
    
    # Mock quality computation to pass checks
    manager._compute_quality = MagicMock(return_value=(0.9, 0.9))
    manager._is_diverse = MagicMock(return_value=True)
    manager._save_image = MagicMock(return_value="data/test_unidentified/test.jpg")
    
    # Manually trigger process_task (usually runs in a thread)
    manager._process_task(task)
    
    # Verify event persisted and extra_data is JSON-safe
    events = repository.get_events(event_type="unidentified_face_saved", limit=5)
    assert len(events) >= 1
    ev = events[0]
    assert ev.extra_data is not None
    # Ensure values are base Python types
    for key, value in ev.extra_data.items():
        assert not isinstance(value, (np.float32, np.float64, np.int32, np.int64)), f"Key {key} has numpy type {type(value)}"

if __name__ == "__main__":
    pytest.main([__file__])
