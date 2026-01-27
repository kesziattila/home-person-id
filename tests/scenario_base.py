import unittest
from unittest.mock import MagicMock
import numpy as np
from datetime import datetime
from src.tracking.track import LocalTrack, TrackState
from src.tracking.global_tracker import GlobalTrackManager
from src.database.repository import Repository
from src.config import Config, CameraTopologyConfig, ReIDConfig, FaceRecognitionConfig, ZonesConfig
from src.recognition.identity_linker import IdentityLinker

class GlobalTrackerScenarioTest(unittest.TestCase):
    def setUp(self):
        # Use in-memory database
        self.repository = Repository(":memory:")
        
        # Setup minimal config
        self.config = Config()
        self.config.reid = ReIDConfig(enabled=True)
        self.config.face_recognition = FaceRecognitionConfig(enabled=False) # Disable for simple tracking tests
        self.config.camera_topology = CameraTopologyConfig(overlaps=[])
        
        # Mock ML models to avoid loading them
        self.mock_face_recognizer = MagicMock()
        self.mock_reid_extractor = MagicMock()
        # Mock Re-ID extraction to return a dummy embedding
        self.mock_reid_extractor.extract.return_value = (np.zeros(512), 0.9)
        self.mock_reid_extractor.compare.return_value = 1.0 # Default perfect match

        # Create IdentityLinker with mocked models
        self.identity_linker = IdentityLinker(
            face_config=self.config.face_recognition,
            reid_config=self.config.reid,
            repository=self.repository,
            face_recognizer=self.mock_face_recognizer,
            reid_extractor=self.mock_reid_extractor
        )

        # Create GlobalTrackManager
        self.gtm = GlobalTrackManager(
            topology_config=self.config.camera_topology,
            reid_config=self.config.reid,
            identity_linker=self.identity_linker,
            repository=self.repository
        )

    def create_local_track(self, track_id, camera_id, bbox=(100, 100, 200, 200), hits=5):
        return LocalTrack(
            track_id=track_id,
            camera_id=camera_id,
            bbox=bbox,
            hits=hits,
            state=TrackState.TRACKED
        )

    def process_frame(self, camera_id, local_tracks, new_ids=None, lost_ids=None):
        if new_ids is None:
            new_ids = [t.track_id for t in local_tracks]
        if lost_ids is None:
            lost_ids = []
            
        # Create dummy frame for Re-ID extraction if not provided
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
            
        return self.gtm.process_local_tracks(
            camera_id=camera_id,
            local_tracks=local_tracks,
            frame=frame,
            new_track_ids=new_ids,
            lost_track_ids=lost_ids
        )
