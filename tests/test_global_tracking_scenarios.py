import numpy as np
from tests.scenario_base import GlobalTrackerScenarioTest
from src.tracking.track import TrackState


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

    def test_lost_track_becomes_removed(self):
        """Test that a lost track eventually becomes REMOVED."""
        cam1 = "cam1"

        # 1. Track appears
        lt1 = self.create_local_track(track_id=1, camera_id=cam1)
        res1 = self.process_frame(cam1, [lt1])
        global_id = res1.new_global_tracks[0]

        # 2. Track is lost — enters GRACE state
        self.process_frame(cam1, [], new_ids=[], lost_ids=[1])
        self.assertEqual(self.gtm._tracks[global_id].state, TrackState.GRACE)

        # ByteTrack permanently removes — GRACE → LOST
        self.process_frame(cam1, [], new_ids=[], lost_ids=[], removed_ids=[1])
        self.assertEqual(self.gtm._tracks[global_id].state, TrackState.LOST)

        # 3. After grace period, should be REMOVED
        from datetime import datetime, timedelta
        self.gtm._tracks[global_id].last_seen = datetime.now() - timedelta(seconds=120)
        import time
        self.gtm._cleanup_lost_tracks(time.time())
        self.assertEqual(self.gtm._tracks[global_id].state, TrackState.REMOVED)


if __name__ == "__main__":
    import unittest
    unittest.main()
