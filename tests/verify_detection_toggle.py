import sys
import os
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from src.config import load_config
from src.main import PersonIDSystem
import yaml

def test_disabled_detection():
    # Create a temporary config with detection disabled
    config_data = {
        "cameras": [
            {
                "id": "test_cam",
                "name": "Test Camera",
                "rtsp_url": "rtsp://localhost:554/stream"
            }
        ],
        "detection": {
            "enabled": False,
            "model": "yolov8n.pt"
        },
        "reid": {"enabled": False},
        "face_recognition": {"enabled": False}
    }
    
    config_path = Path("tests/temp_config.yaml")
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
        
    try:
        print("Initializing system with detection disabled...")
        system = PersonIDSystem(str(config_path))
        
        assert system.person_detector is None, "Person detector should be None when disabled"
        print("Person detector is None as expected.")
        
        # Test start (warmup)
        print("Warming up...")
        # We don't want to actually start the stream manager as it will try to connect to RTSP
        # So we just test that person_detector.warmup() is not called
        system.identity_linker.warmup = lambda: print("IdentityLinker warmup called")
        
        # This should not crash even if person_detector is None
        system.person_detector = None # Ensure it's None
        
        # Mocking stream_manager.start to avoid network issues
        system.stream_manager.start = lambda: print("StreamManager start mocked")
        system.api_server.start = lambda: print("APIServer start mocked")
        
        system.start()
        print("System started successfully with detection disabled.")
        
        # Test inference worker with a dummy task
        import numpy as np
        from src.main import InferenceTask
        
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        task = InferenceTask(
            camera_id="test_cam",
            frame=dummy_frame,
            motion_result=None,
            has_active_tracks=False,
            timestamp=0.0
        )
        
        system._inference_queue.put(task)
        
        # We can't easily run the whole thread in a test, but we can call the worker logic once
        # if we modify it to be more testable, but for now we've checked the initialization and start.
        
    finally:
        if config_path.exists():
            config_path.unlink()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        test_disabled_detection()
        print("\nSUCCESS: System handles disabled detection correctly.")
    except Exception as e:
        print(f"\nFAILURE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
