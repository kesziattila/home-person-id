"""A new track is named from the Re-ID gallery only after reid.min_consecutive_matches.

Live 2026-10-23 09:04:41 UTC (health watch, teleport alert): living_room track
global_63745 was born on a legs-only box of a child in the hanging chair. The gallery
recheck counted Ati 1/3 (sim 0.72) — and 14 ms later the separate birth path in
GlobalTrackManager (`_try_gallery_match`) named it Ati from that single crop. Ati's
zone flipped work_room -> living_room -> work_room twice until the age veto revoked
the name 3.6 s later. Two naming paths, only one of them honoured the streak; the
recheck path already runs on the birth frame (young tracks every 0.5 s), so the
streak costs about one second.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.config import CameraTopologyConfig, FaceRecognitionConfig, ReIDConfig
from src.database.repository import Repository
from src.recognition.identity_linker import IdentityLinker
from src.tracking.global_tracker import GlobalTrackManager
from src.tracking.track import LocalTrack, TrackState


class TestBirthGalleryStreak(unittest.TestCase):

    def setUp(self):
        self.repository = Repository(":memory:")
        reid_config = ReIDConfig(enabled=True, min_consecutive_matches=3)
        extractor = MagicMock()
        extractor.extract.return_value = (np.zeros(512), 0.9)
        extractor.compare.return_value = 1.0
        self.linker = IdentityLinker(
            face_config=FaceRecognitionConfig(enabled=False),
            reid_config=reid_config,
            repository=self.repository,
            face_recognizer=MagicMock(),
            reid_extractor=extractor,
        )
        gallery = MagicMock()
        gallery.match_new_track.return_value = MagicMock(
            matched=True, person_name="Ati", score=0.72, best_score=0.72, db_id=1)
        self.linker._id_manager._reid_gallery_manager = gallery
        self.gtm = GlobalTrackManager(
            topology_config=CameraTopologyConfig(overlaps=[]),
            reid_config=reid_config,
            identity_linker=self.linker,
            repository=self.repository,
        )
        self.person = self.repository.create_person("Ati")

    def _frame(self, new_ids):
        track = LocalTrack(track_id=203, camera_id="living_room", bbox=(280, 560, 560, 715),
                           hits=5, state=TrackState.TRACKED,
                           last_crop=np.zeros((155, 280, 3), dtype=np.uint8))
        return self.gtm.process_local_tracks(
            camera_id="living_room", local_tracks=[track],
            frame=np.zeros((720, 1280, 3), dtype=np.uint8),
            new_track_ids=new_ids, lost_track_ids=[], removed_track_ids=[])

    def test_single_birth_match_does_not_name_the_track(self):
        result = self._frame(new_ids=[203])
        state = self.linker.get_track_state(result.new_global_tracks[0])
        self.assertFalse(state.is_identified)

    def test_birth_track_named_after_three_consecutive_matches(self):
        t0 = 1_000_000.0
        with patch("src.recognition.identity_linker.time.time", return_value=t0):
            result = self._frame(new_ids=[203])
        state = self.linker.get_track_state(result.new_global_tracks[0])
        state.first_seen = t0
        named_at = None
        for age in (0.6, 1.2, 1.8, 2.4):
            with patch("src.recognition.identity_linker.time.time", return_value=t0 + age):
                self._frame(new_ids=[])
            if state.is_identified:
                named_at = age
                break
        self.assertEqual(state.person_id, self.person.id)
        self.assertLessEqual(named_at, 1.8)


if __name__ == "__main__":
    unittest.main()
