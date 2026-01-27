import unittest
from unittest.mock import MagicMock
import numpy as np
import signal
from tests.scenario_base import GlobalTrackerScenarioTest
from src.config import ZonesConfig, ZoneConfig, ZonePolygon, ReIDConfig
from src.tracking.track import LocalTrack, TrackState
from src.tracking.global_tracker import GlobalTrackManager

def timeout_handler(signum, frame):
    raise TimeoutError("Test timed out!")

class TestGlobalTrackerHandover(GlobalTrackerScenarioTest):
    def setUp(self):
        super().setUp()
        # Set a 10-second timeout for each test
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)
        
        # Configure a zone for handover
        # Zone "hallway" visible on cam1 and cam2
        self.zone_name = "hallway"
        self.zones_config = ZonesConfig(
            zones=[
                ZoneConfig(
                    name=self.zone_name,
                    cameras={
                        "cam1": ZonePolygon(polygon=[[0.8, 0.0], [1.0, 0.0], [1.0, 1.0], [0.8, 1.0]]),
                        "cam2": ZonePolygon(polygon=[[0.0, 0.0], [0.2, 0.0], [0.2, 1.0], [0.0, 1.0]])
                    },
                    max_handover_sec=5.0
                )
            ]
        )
        
        # Re-initialize GTM with zones
        self.gtm = GlobalTrackManager(
            topology_config=self.config.camera_topology,
            reid_config=self.config.reid,
            identity_linker=self.identity_linker,
            repository=self.repository,
            zones_config=self.zones_config
        )
        
        # Enable zones in topology
        self.gtm.topology_config.use_zones = True
        
        # Set standard frame dimensions to avoid large dummy frames
        self.gtm._frame_dimensions["cam1"] = (1920, 1080)
        self.gtm._frame_dimensions["cam2"] = (1920, 1080)

    def tearDown(self):
        signal.alarm(0) # Disable alarm
        super().tearDown()

    def test_zone_handover_reid_match_success(self):
        """
        Reproduce the fix for ValueError: list.remove(x): x not in list
        This happens when a zone-based handover matches via Re-ID.
        """
        cam1 = "cam1"
        cam2 = "cam2"
        
        # Use a very small frame to save memory
        small_frame = np.zeros((8, 8, 3), dtype=np.uint8)
        
        # 1. Track appears on cam1
        lt1 = self.create_local_track(track_id=1, camera_id=cam1, bbox=(100, 100, 200, 200))
        # Mock crop for Re-ID (minimal size)
        lt1.last_crop = np.zeros((8, 8, 3), dtype=np.uint8)
        
        res1 = self.gtm.process_local_tracks(cam1, [lt1], frame=small_frame, new_track_ids=[1], lost_track_ids=[])
        global_id = res1.new_global_tracks[0]
        
        # 2. Add Re-ID embedding to the track via identity_linker
        dummy_embedding = np.ones(512, dtype=np.float32)
        state = self.identity_linker.get_track_state(global_id)
        # Use a more realistic way to add embeddings that match EmbeddingGallery structure
        state.update_reid_gallery(dummy_embedding, 0.9)
        
        # 3. Track moves to zone on cam1 and is lost
        # Hallway on cam1 is [0.8, 0.0] to [1.0, 1.0] -> x: 1536 to 1920
        # Bottom-center should be in the zone.
        # let's put it at center of zone: x=0.9*1920=1728, y=0.5*1080=540
        # bbox: center_x=1728, bottom=540. Width/height=100.
        # x1=1678, y1=440, x2=1778, y2=540
        lt1.bbox = (1678, 440, 1778, 540)
        # Update GTM with last bbox
        self.gtm.process_local_tracks(cam1, [lt1], frame=small_frame, new_track_ids=[], lost_track_ids=[])
        
        # Now lose it
        self.gtm.process_local_tracks(cam1, [], frame=small_frame, new_track_ids=[], lost_track_ids=[1])
        
        # Verify it's in pending handovers
        self.assertEqual(len(self.gtm._pending_handovers), 1)
        
        # 4. Track appears on cam2 in the same zone
        # Hallway on cam2 is [0.0, 0.0] to [0.2, 1.0] -> x: 0 to 384
        # center of zone: x=0.1*1920=192, y=0.5*1080=540
        # x1=142, y1=440, x2=242, y2=540
        lt2 = self.create_local_track(track_id=2, camera_id=cam2, bbox=(142, 440, 242, 540))
        lt2.last_crop = np.zeros((8, 8, 3), dtype=np.uint8)
        
        # Mock Re-ID extractor to return the same embedding
        self.mock_reid_extractor.extract.return_value = (dummy_embedding, 0.9)
        
        # Process frame on cam2 - this should trigger the handover match
        res2 = self.gtm.process_local_tracks(
            camera_id=cam2,
            local_tracks=[lt2],
            frame=small_frame,
            new_track_ids=[2],
            lost_track_ids=[]
        )
            
        # Verify handover completed
        self.assertEqual(len(res2.handovers_completed), 1)
        self.assertEqual(res2.handovers_completed[0][0], global_id)
        self.assertEqual(len(self.gtm._pending_handovers), 0)

if __name__ == "__main__":
    unittest.main()
