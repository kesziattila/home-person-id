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
    reid_config = ReIDConfig(enabled=True, max_overlap=0.15)
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


def test_neighbour_lost_a_moment_ago_still_counts():
    """Live 2026-10-18 15:49-16:05 UTC: ByteTrack kept dropping Ati's track for 1-3 s
    while Virag sat next to him; each drop made her box "isolated" (297 of 299
    isolated frames in replay) and two of her tracks were gallery-named Ati."""
    gtm, linker = _manager()
    ati, virag = _track(72, (100, 100, 300, 260)), _track(77, (200, 150, 320, 330))
    _process(gtm, [ati, virag], [72, 77])
    _process(gtm, [ati, virag], [])
    linker.process_track = MagicMock()

    gtm.process_local_tracks(camera_id="ground_floor", local_tracks=[virag],
                             frame=np.zeros((480, 640, 3), dtype=np.uint8),
                             new_track_ids=[], lost_track_ids=[72])
    _process(gtm, [virag], [])

    assert linker.process_track.call_args.kwargs["isolated"] is False


def test_new_track_where_a_person_was_just_lost_stays_matchable():
    # Live 2026-10-17 11:14 UTC: Ati's track was dropped and his new track came back
    gtm, linker = _manager()
    _process(gtm, [_track(72, (100, 100, 300, 260))], [72])
    gtm.process_local_tracks(camera_id="ground_floor", local_tracks=[],
                             frame=np.zeros((480, 640, 3), dtype=np.uint8),
                             new_track_ids=[], lost_track_ids=[72])
    linker.reid_gallery_manager.match_new_track.reset_mock()

    _process(gtm, [_track(80, (105, 100, 305, 260))], [80])

    linker.reid_gallery_manager.match_new_track.assert_called()
    assert _person(gtm, linker, 80) is not None


def test_lost_neighbour_boxes_count_like_live_boxes():
    track = _box(1, (0, 0, 100, 200))
    assert isolated_track_ids([track], 0.15, {1: [(50, 0, 150, 200)]}) == set()
    assert isolated_track_ids([track], 0.15, {1: [(400, 0, 500, 200)]}) == {1}


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


# ---------------------------------------------------------------- appearance settled

def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_appearance_change_blocks_strong_face_for_settle_period():
    """2026-10-19 17:02: Judit's track switched onto Ati (Re-ID vs EMA 0.49) and one strong Ati
    face named it. Same-person samples stay >= 0.79 (p1), so they never unsettle a track."""
    from src.tracking.attribution import APPEARANCE_SETTLE_SEC, appearance_settled, update_appearance
    from src.tracking.track import LocalTrack
    track = LocalTrack(track_id=1, camera_id="c", bbox=(0, 0, 1, 1))
    judit, ati = _unit([1, 0, 0]), _unit([0.45, 0.89, 0])          # cosine 0.45
    for t in range(10):
        update_appearance(track, judit + 0.1 * _unit([0, 0, 1]), now=float(t))
    assert appearance_settled(track, now=10.0)                     # small variation: settled
    update_appearance(track, ati, now=10.0)
    assert not appearance_settled(track, now=10.0 + APPEARANCE_SETTLE_SEC - 0.1)
    assert appearance_settled(track, now=10.0 + APPEARANCE_SETTLE_SEC)


def test_byte_tracker_keeps_track_appearance():
    from src.config import TrackingConfig
    from src.detection.person_detector import Detection, DetectionResult
    from src.tracking.byte_tracker import ByteTracker
    tracker = ByteTracker("c", TrackingConfig())
    det = DetectionResult(detections=[Detection(bbox=(100.0, 100.0, 200.0, 400.0), confidence=0.9)],
                          frame_shape=(1000, 1000, 3))
    emb = _unit([1, 2, 3])
    for _ in range(5):
        tracker.update(det, None, {0: (emb, 1.0)})
    track = tracker.get_track(1)
    assert np.allclose(track.appearance / np.linalg.norm(track.appearance), emb)
    assert track.appearance_changed_at == 0.0
