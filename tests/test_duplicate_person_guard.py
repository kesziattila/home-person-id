"""Tests for the two fixes from the live double-Rozika incident (2026-06-06).

Observed live: a second person lying on the floor scored 0.65-0.68 (just above
the 0.65 Re-ID threshold) against a clean Rozika gallery via single-frame
gallery rechecks, producing two simultaneous "Rozika" tracks on one camera.

Fix 1 — same-camera duplicate-person guard: two simultaneous tracks on the
SAME camera cannot be the same physical person, so gallery matches proposing
a person held by a LIVE track there are rejected
(GlobalTrackManager.persons_tracked_on_camera; lost tracks never block).

Fix 2 — reid.min_consecutive_matches is now enforced in the gallery recheck
(it was parsed but never applied): a single noisy frame cannot flip identity.
"""

from unittest.mock import MagicMock

import numpy as np

from src.config import FaceRecognitionConfig, ReIDConfig
from src.recognition.identity_linker import IdentityLinker
from src.tracking.track import TrackState


def _make_linker(min_consecutive_matches: int = 1) -> tuple[IdentityLinker, MagicMock, MagicMock]:
    reid_config = ReIDConfig(
        enabled=True,
        gallery_recheck_interval=0.0,
        min_consecutive_matches=min_consecutive_matches,
    )
    face_config = FaceRecognitionConfig(enabled=False)
    repo = MagicMock()

    mock_reid_extractor = MagicMock()
    mock_reid_extractor.extract.return_value = (np.zeros(512), 0.9)

    linker = IdentityLinker(
        face_config=face_config,
        reid_config=reid_config,
        repository=repo,
        reid_extractor=mock_reid_extractor,
    )
    mock_gallery = MagicMock()
    linker._id_manager._reid_gallery_manager = mock_gallery
    return linker, repo, mock_gallery


def _mock_match(mock_gallery, repo, name="Rozika", person_id=4, score=0.66):
    match_result = MagicMock()
    match_result.matched = True
    match_result.person_name = name
    match_result.score = score
    match_result.best_score = score
    match_result.db_id = 1
    mock_gallery.match_new_track.return_value = match_result

    person = MagicMock()
    person.id = person_id
    repo.get_person_by_name.return_value = person


def _mock_no_match(mock_gallery):
    match_result = MagicMock()
    match_result.matched = False
    mock_gallery.match_new_track.return_value = match_result


CROP = np.zeros((100, 50, 3), dtype=np.uint8)


class TestSameCameraDuplicatePersonGuard:
    def test_recheck_rejected_when_person_is_held_on_this_camera(self):
        linker, repo, mock_gallery = _make_linker()
        _mock_match(mock_gallery, repo, person_id=4)
        state = linker.register_track("global_2")
        assert linker._try_gallery_identification(state, CROP, "cam_a", frozenset({4})) is False
        assert state.person_id is None

    def test_recheck_allowed_when_person_is_not_held(self):
        linker, repo, mock_gallery = _make_linker()
        _mock_match(mock_gallery, repo, person_id=4)
        state = linker.register_track("global_2")
        assert linker._try_gallery_identification(state, CROP, "cam_a", frozenset({1})) is True
        assert state.person_id == 4


def _gtm_with(*tracks):
    """GlobalTrackManager with (track_id, camera, state, person_id) tracks."""
    from datetime import datetime
    from src.tracking.track import GlobalTrack
    from tests.test_cross_camera_zone_propagation import _make_global_track_manager
    manager = _make_global_track_manager()
    for track_id, camera, state, person_id in tracks:
        manager._tracks[track_id] = GlobalTrack(track_id=track_id, state=state, current_camera_id=camera,
                                                last_seen=datetime.now())
        linker_state = manager.identity_linker.register_track(track_id)
        linker_state.person_id = person_id
    return manager


class TestPersonsTrackedOnCamera:
    def test_live_tracks_on_the_camera_count(self):
        manager = _gtm_with(("g1", "cam_a", TrackState.TRACKED, 4), ("g2", "cam_a", TrackState.NEW, None))
        assert manager.persons_tracked_on_camera("cam_a") == {4}

    def test_other_camera_does_not_count(self):
        manager = _gtm_with(("g1", "cam_b", TrackState.TRACKED, 4))  # overlapping views are legitimate
        assert manager.persons_tracked_on_camera("cam_a") == frozenset()

    def test_asking_track_is_excluded(self):
        manager = _gtm_with(("g1", "cam_a", TrackState.TRACKED, 4))
        assert manager.persons_tracked_on_camera("cam_a", exclude="g1") == frozenset()

    def test_just_lost_track_does_not_block_its_person_coming_back(self):
        # Live 2026-10-17 11:14 UTC: ByteTrack dropped Ati's living_room track; 0.9 s
        # later his new track matched the gallery as Ati (0.80) and was rejected because
        # the lost track still counted as "active" -> he stayed unnamed.
        manager = _gtm_with(("g_lost", "cam_a", TrackState.GRACE, 1))
        assert manager.persons_tracked_on_camera("cam_a") == frozenset()
        assert manager._persons_on_camera("cam_a") == 1  # location evidence still counts him


class TestGalleryRecheckConsecutiveMatches:
    def test_single_hit_does_not_confirm_with_min_3(self):
        linker, repo, mock_gallery = _make_linker(min_consecutive_matches=3)
        _mock_match(mock_gallery, repo, person_id=4, score=0.66)
        state = linker.register_track("global_1")

        assert linker._try_gallery_identification(state, CROP, "cam_a") is False
        assert state.person_id is None
        assert state.gallery_consecutive_matches == 1

    def test_confirms_on_third_consecutive_hit(self):
        linker, repo, mock_gallery = _make_linker(min_consecutive_matches=3)
        _mock_match(mock_gallery, repo, person_id=4)
        state = linker.register_track("global_1")

        assert linker._try_gallery_identification(state, CROP, "cam_a") is False
        assert linker._try_gallery_identification(state, CROP, "cam_a") is False
        assert linker._try_gallery_identification(state, CROP, "cam_a") is True
        assert state.person_id == 4

    def test_miss_resets_streak(self):
        linker, repo, mock_gallery = _make_linker(min_consecutive_matches=2)
        state = linker.register_track("global_1")

        _mock_match(mock_gallery, repo, person_id=4)
        assert linker._try_gallery_identification(state, CROP, "cam_a") is False

        _mock_no_match(mock_gallery)
        assert linker._try_gallery_identification(state, CROP, "cam_a") is False
        assert state.gallery_consecutive_matches == 0

        _mock_match(mock_gallery, repo, person_id=4)
        assert linker._try_gallery_identification(state, CROP, "cam_a") is False  # streak restarted
        assert linker._try_gallery_identification(state, CROP, "cam_a") is True

    def test_candidate_switch_resets_streak(self):
        linker, repo, mock_gallery = _make_linker(min_consecutive_matches=2)
        state = linker.register_track("global_1")

        _mock_match(mock_gallery, repo, name="Rozika", person_id=4)
        assert linker._try_gallery_identification(state, CROP, "cam_a") is False

        _mock_match(mock_gallery, repo, name="Ati", person_id=1)
        assert linker._try_gallery_identification(state, CROP, "cam_a") is False
        assert state.gallery_candidate_person_id == 1
        assert state.gallery_consecutive_matches == 1

    def test_min_consecutive_1_confirms_immediately(self):
        linker, repo, mock_gallery = _make_linker(min_consecutive_matches=1)
        _mock_match(mock_gallery, repo, person_id=4)
        state = linker.register_track("global_1")
        assert linker._try_gallery_identification(state, CROP, "cam_a") is True
