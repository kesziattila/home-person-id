import unittest
import numpy as np
import time
from datetime import datetime, timedelta
from tests.scenario_base import GlobalTrackerScenarioTest

class TestGlobalTrackGracePeriod(GlobalTrackerScenarioTest):
    def test_reid_match_within_grace_period(self):
        """Test that Re-ID matches and keeps same ID within grace period."""
        cam1 = "cam1"
        cam2 = "cam2"
        
        # Set short grace period for testing
        self.gtm.reid_config.global_id_grace_period = 2.0
        
        # 1. Track appears on cam1
        lt1 = self.create_local_track(track_id=1, camera_id=cam1)
        res1 = self.process_frame(cam1, [lt1])
        global_id = res1.new_global_tracks[0]
        
        # Ensure track is registered and has embedding
        state = self.identity_linker.register_track(global_id)
        dummy_emb = np.random.rand(512).astype(np.float32)
        state.update_reid_gallery(dummy_emb, 0.9)

        # 2. Track is lost on cam1
        self.process_frame(cam1, [], new_ids=[], lost_ids=[1])
        
        # 3. Track reappears on cam2 immediately
        self.mock_reid_extractor.extract.return_value = (dummy_emb, 0.9)
        
        lt2 = self.create_local_track(track_id=2, camera_id=cam2)
        lt2.last_crop = np.zeros((10, 10, 3), dtype=np.uint8)
        res2 = self.process_frame(cam2, [lt2], new_ids=[2])
        
        # Verify it keeps the same ID
        self.assertEqual(self.gtm.get_global_track_for_local(cam2, 2).track_id, global_id)

    def test_reid_match_outside_grace_period(self):
        """Test that Re-ID match outside grace period gets a NEW ID."""
        cam1 = "cam1"
        cam2 = "cam2"
        
        # Set VERY short grace period
        self.gtm.reid_config.global_id_grace_period = 0.1
        
        # 1. Track appears on cam1
        lt1 = self.create_local_track(track_id=1, camera_id=cam1)
        res1 = self.process_frame(cam1, [lt1])
        global_id = res1.new_global_tracks[0]
        
        # Ensure track is registered and has embedding
        state = self.identity_linker.register_track(global_id)
        dummy_emb = np.random.rand(512).astype(np.float32)
        state.update_reid_gallery(dummy_emb, 0.9)

        # 2. Track is lost on cam1
        self.process_frame(cam1, [], new_ids=[], lost_ids=[1])
        
        # Manually backdate the last_seen to simulate time passing beyond grace period
        gt_obj = self.gtm.get_global_track(global_id)
        gt_obj.last_seen = datetime.now() - timedelta(seconds=1)
        
        # 3. Track reappears on cam2
        self.mock_reid_extractor.extract.return_value = (dummy_emb, 0.9)
        
        lt2 = self.create_local_track(track_id=2, camera_id=cam2)
        lt2.last_crop = np.zeros((10, 10, 3), dtype=np.uint8)
        res2 = self.process_frame(cam2, [lt2], new_ids=[2])
        
        # Verify it gets a NEW ID
        new_global_id = res2.new_global_tracks[0]
        self.assertNotEqual(new_global_id, global_id)
        self.assertEqual(self.gtm.get_global_track_for_local(cam2, 2).track_id, new_global_id)

if __name__ == "__main__":
    unittest.main()
