import numpy as np
from tests.scenario_base import GlobalTrackerScenarioTest
from src.config import CameraOverlap

class TestGlobalTrackingScenarios(GlobalTrackerScenarioTest):
    def test_single_camera_track_creation(self):
        """Test that a new local track creates a new global track."""
        camera_id = "cam1"
        local_track = self.create_local_track(track_id=1, camera_id=camera_id)
        
        result = self.process_frame(camera_id, [local_track])
        
        self.assertEqual(len(result.new_global_tracks), 1)
        global_track_id = result.new_global_tracks[0]
        self.assertIn(global_track_id, self.gtm._tracks)
        
        # Verify it's linked correctly
        gt_obj = self.gtm.get_global_track_for_local(camera_id, 1)
        self.assertIsNotNone(gt_obj)
        self.assertEqual(gt_obj.track_id, global_track_id)

    def test_handover_between_cameras(self):
        """Test handover between two cameras with overlapping zones."""
        cam1 = "cam1"
        cam2 = "cam2"
        
        # Configure topology with overlap
        # cam1 exit zone: right side
        # cam2 entry zone: left side
        overlap = CameraOverlap(
            cameras=[cam1, cam2],
            cam1_exit_zone=[0.8, 0.0, 1.0, 1.0], # x1, y1, x2, y2 normalized
            cam2_entry_zone=[0.0, 0.0, 0.2, 1.0],
            max_handover_sec=5.0
        )
        self.gtm.topology_config.overlaps.append(overlap)
        self.gtm._load_handover_zones() # Refresh internal zones

        # 1. Track appears on cam1
        lt1 = self.create_local_track(track_id=10, camera_id=cam1, bbox=(100, 100, 200, 200))
        res1 = self.process_frame(cam1, [lt1])
        global_id = res1.new_global_tracks[0]

        # 2. Track moves to exit zone on cam1
        lt1.bbox = (1800, 100, 1900, 300) # Assuming 1920x1080 default
        self.process_frame(cam1, [lt1], new_ids=[])

        # 3. Track is lost on cam1
        self.process_frame(cam1, [], new_ids=[], lost_ids=[10])

        # 4. Track appears on cam2 in entry zone
        lt2 = self.create_local_track(track_id=20, camera_id=cam2, bbox=(50, 100, 150, 300))
        res2 = self.process_frame(cam2, [lt2], new_ids=[20])

        # Verify handover
        self.assertEqual(len(res2.handovers_completed), 1)
        self.assertEqual(res2.handovers_completed[0][0], global_id)
        gt_obj = self.gtm.get_global_track_for_local(cam2, 20)
        self.assertIsNotNone(gt_obj)
        self.assertEqual(gt_obj.track_id, global_id)

    def test_reid_reappearance_cross_camera(self):
        """Test that Re-ID can match a track that reappears on a different camera."""
        cam1 = "cam1"
        cam2 = "cam2"
        
        # 1. Track appears on cam1
        lt1 = self.create_local_track(track_id=1, camera_id=cam1)
        res1 = self.process_frame(cam1, [lt1])
        global_id = res1.new_global_tracks[0]
        
        # Ensure track is registered in identity_linker (normally done by GTM)
        state = self.identity_linker.register_track(global_id)
        
        # Mock Re-ID embedding for this track
        dummy_emb = np.random.rand(512).astype(np.float32)
        state.update_reid_gallery(dummy_emb, 0.9)

        # 2. Track is lost on cam1
        self.process_frame(cam1, [], new_ids=[], lost_ids=[1])
        
        # 3. Track reappears on cam2
        # Mock extractor to return the same embedding
        self.mock_reid_extractor.extract.return_value = (dummy_emb, 0.9)
        self.mock_reid_extractor.compare.return_value = 0.95 # High similarity
        
        lt2 = self.create_local_track(track_id=2, camera_id=cam2)
        lt2.last_crop = np.zeros((10, 10, 3), dtype=np.uint8) # Add crop for Re-ID trigger
        res2 = self.process_frame(cam2, [lt2], new_ids=[2])
        
        # Verify Re-ID match
        gt_obj = self.gtm.get_global_track_for_local(cam2, 2)
        self.assertIsNotNone(gt_obj)
        self.assertEqual(gt_obj.track_id, global_id)

if __name__ == "__main__":
    import unittest
    unittest.main()
