"""Evidence attribution (src/tracking/attribution.py): one rule per evidence kind.

Audit reviews on 2026-10-17 found three mislabels with one root cause — evidence
credited to a box it may not belong to — each produced by a different ad-hoc
check: a gallery match on a new track half-covering another person (hard-coded
num_persons=1), a zone handover while the other person's track was briefly lost
and her feet were outside the zone, and an adult face picked in a child's box.
Expected now:
- appearance (Re-ID) evidence only for isolated tracks (overlap measured over the
  smaller box), on every path incl. new-track gallery match;
- location evidence only when each camera sees exactly one person, briefly lost
  tracks included, regardless of zone membership;
- face evidence only from the face the box owns (tests/test_face_multi_face.py).
"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from src.config import CameraTopologyConfig, FaceRecognitionConfig, ReIDConfig
from src.database.repository import Repository
from src.recognition.identity_linker import IdentityLinker
from src.recognition.reid_gallery import MatchResult
from src.tracking.attribution import isolated_track_ids, overlap_of_smaller
from src.tracking.global_tracker import GlobalTrackManager
from src.tracking.track import GlobalTrack, LocalTrack, TrackState
from tests.test_cross_camera_zone_propagation import (
    _make_global_track_manager, _make_zone_config, _register_active_track,
)


def _box(track_id, bbox):
    return SimpleNamespace(track_id=track_id, bbox=bbox)


# ------------------------------------------------------------ appearance rule

def test_small_box_inside_large_box_is_not_isolated_even_at_low_iou():
    adult = _box(1, (0, 0, 400, 800))
    child = _box(2, (250, 400, 400, 800))          # IoU 0.19, but fully inside the adult box
    far = _box(3, (900, 0, 1100, 600))
    assert overlap_of_smaller(adult.bbox, child.bbox) == 1.0
    assert isolated_track_ids([adult, child, far], 0.15) == {3}


def test_zero_threshold_disables_the_check():
    assert isolated_track_ids([_box(1, (0, 0, 10, 10)), _box(2, (0, 0, 10, 10))], 0.0) == {1, 2}


def _manager():
    repository = Repository(":memory:")
    repository.create_person("Ati")
    reid_config = ReIDConfig(enabled=True, skip_overlapping_iou=0.15)
    extractor = MagicMock()
    extractor.extract.return_value = (np.zeros(512), 0.9)
    linker = IdentityLinker(
        face_config=FaceRecognitionConfig(enabled=False), reid_config=reid_config,
        repository=repository, face_recognizer=MagicMock(), reid_extractor=extractor,
    )
    linker.reid_gallery_manager.match_new_track = MagicMock(
        return_value=MatchResult(matched=True, person_name="Ati", score=0.78)
    )
    gtm = GlobalTrackManager(
        topology_config=CameraTopologyConfig(overlaps=[]), reid_config=reid_config,
        identity_linker=linker, repository=repository,
    )
    return gtm, linker


def _track(track_id, bbox):
    return LocalTrack(track_id=track_id, camera_id="ground_floor", bbox=bbox, hits=5,
                      state=TrackState.TRACKED, last_crop=np.zeros((100, 60, 3), dtype=np.uint8))


def _process(gtm, tracks, new_ids):
    gtm.process_local_tracks(camera_id="ground_floor", local_tracks=tracks,
                             frame=np.zeros((480, 640, 3), dtype=np.uint8),
                             new_track_ids=new_ids, lost_track_ids=[])


def _person(gtm, linker, local_id):
    return linker.get_track_state(gtm._local_to_global[("ground_floor", local_id)]).person_id


def test_new_track_overlapping_another_gets_no_appearance_identity():
    # Live 10:05 UTC: Virag's new box half-covered Ati and was gallery-matched "Ati" (0.78)
    gtm, linker = _manager()
    ati = _track(72, (100, 100, 300, 260))
    _process(gtm, [ati], [72])
    linker.reid_gallery_manager.match_new_track.reset_mock()

    _process(gtm, [ati, _track(77, (200, 150, 320, 330))], [77])

    linker.reid_gallery_manager.match_new_track.assert_not_called()
    assert _person(gtm, linker, 77) is None


def test_isolated_new_track_still_gets_gallery_identity():
    gtm, linker = _manager()
    _process(gtm, [_track(77, (400, 150, 520, 330))], [77])
    assert _person(gtm, linker, 77) is not None


# -------------------------------------------------------------- location rule

def test_no_handover_while_the_other_person_is_briefly_lost():
    # Live 10:29 UTC: ground_floor's other track was dropped for a moment and the
    # living_room camera's zone missed Virag's feet -> Ati's name went to Virag's box
    zones = _make_zone_config("living_room", ["cam_a", "cam_b"])
    manager = _make_global_track_manager(zones_config=zones)
    _register_active_track(manager, "cam_a", 1, "global_1", bbox=(100, 100, 300, 500),
                           person_id=1, identified_by="face", confidence=0.7)
    receiver = _register_active_track(manager, "cam_b", 1, "global_2", bbox=(200, 150, 400, 600))
    manager._tracks["global_3"] = GlobalTrack(
        track_id="global_3", state=TrackState.GRACE, current_camera_id="cam_b",
        last_seen=datetime.now() - timedelta(seconds=1),
    )

    manager._try_zone_identity_propagation(manager._update_track_zones())

    assert receiver.person_id is None


def test_handover_once_the_lost_track_is_long_gone():
    zones = _make_zone_config("living_room", ["cam_a", "cam_b"])
    manager = _make_global_track_manager(zones_config=zones)
    _register_active_track(manager, "cam_a", 1, "global_1", bbox=(100, 100, 300, 500),
                           person_id=1, identified_by="face", confidence=0.7)
    receiver = _register_active_track(manager, "cam_b", 1, "global_2", bbox=(200, 150, 400, 600))
    manager._tracks["global_3"] = GlobalTrack(
        track_id="global_3", state=TrackState.GRACE, current_camera_id="cam_b",
        last_seen=datetime.now() - timedelta(seconds=30),
    )

    manager._try_zone_identity_propagation(manager._update_track_zones())

    assert receiver.person_id == 1


def test_no_handover_when_a_second_person_is_outside_the_zone():
    zones = _make_zone_config("living_room", ["cam_a", "cam_b"])
    manager = _make_global_track_manager(zones_config=zones)
    _register_active_track(manager, "cam_a", 1, "global_1", bbox=(100, 100, 300, 500),
                           person_id=1, identified_by="face", confidence=0.7)
    manager._tracks["global_9"] = GlobalTrack(          # on cam_a but in no zone
        track_id="global_9", state=TrackState.TRACKED, current_camera_id="cam_a",
    )
    receiver = _register_active_track(manager, "cam_b", 1, "global_2", bbox=(200, 150, 400, 600))

    manager._try_zone_identity_propagation(manager._update_track_zones())

    assert receiver.person_id is None
