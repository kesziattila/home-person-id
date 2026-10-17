"""Tests for the two fixes from the live double-Rozika incident (2026-06-06).

Observed live: a second person lying on the floor scored 0.65-0.68 (just above
the 0.65 Re-ID threshold) against a clean Rozika gallery via single-frame
gallery rechecks, producing two simultaneous "Rozika" tracks on one camera.

Fix 1 — same-camera duplicate-person guard: two simultaneous tracks on the
SAME camera cannot be the same physical person, so gallery matches proposing
an already-present person are rejected.

Fix 2 — reid.min_consecutive_matches is now enforced in the gallery recheck
(it was parsed but never applied): a single noisy frame cannot flip identity.
"""

import time
from unittest.mock import MagicMock

import numpy as np

from src.config import FaceRecognitionConfig, ReIDConfig
from src.recognition.identity_linker import IdentityLinker


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
    def test_rejected_when_person_already_active_on_same_camera(self):
        linker, repo, mock_gallery = _make_linker()
        # Rozika already actively tracked on cam_a
        holder = linker.register_track("global_1")
        holder.person_id = 4
        holder.last_camera_id = "cam_a"
        holder.last_seen_time = time.time()

        _mock_match(mock_gallery, repo, person_id=4)
        new_state = linker.register_track("global_2")
        assert linker._try_gallery_identification(new_state, CROP, "cam_a") is False
        assert new_state.person_id is None

    def test_allowed_when_person_is_on_different_camera(self):
        linker, repo, mock_gallery = _make_linker()
        holder = linker.register_track("global_1")
        holder.person_id = 4
        holder.last_camera_id = "cam_b"  # overlapping camera views are legitimate
        holder.last_seen_time = time.time()

        _mock_match(mock_gallery, repo, person_id=4)
        new_state = linker.register_track("global_2")
        assert linker._try_gallery_identification(new_state, CROP, "cam_a") is True
        assert new_state.person_id == 4

    def test_allowed_when_same_camera_holder_is_stale(self):
        linker, repo, mock_gallery = _make_linker()
        holder = linker.register_track("global_1")
        holder.person_id = 4
        holder.last_camera_id = "cam_a"
        holder.last_seen_time = time.time() - 60.0  # lost long ago

        _mock_match(mock_gallery, repo, person_id=4)
        new_state = linker.register_track("global_2")
        assert linker._try_gallery_identification(new_state, CROP, "cam_a") is True

    def test_asking_track_is_excluded_from_scan(self):
        linker, _, _ = _make_linker()
        state = linker.register_track("global_1")
        state.person_id = 4
        state.last_camera_id = "cam_a"
        state.last_seen_time = time.time()
        assert linker.is_person_active_on_camera(
            "cam_a", exclude_track_id="global_1", person_id=4
        ) is False

    def test_person_name_lookup_for_new_track_path(self):
        linker, _, _ = _make_linker()
        state = linker.register_track("global_1")
        state.person_id = 4
        state.last_camera_id = "cam_a"
        state.last_seen_time = time.time()
        linker._id_manager.get_person_name = MagicMock(return_value="Rozika")
        assert linker.is_person_active_on_camera("cam_a", person_name="Rozika") is True
        assert linker.is_person_active_on_camera("cam_a", person_name="Ati") is False


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
