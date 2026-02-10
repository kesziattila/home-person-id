"""Tests for periodic gallery recheck of unidentified tracks.

Covers:
- Unidentified track gets identified via gallery recheck
- Gallery recheck is time-gated (respects interval)
- Already-identified tracks skip gallery recheck
- No gallery manager -> no crash
- Database and events updated on match
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config import FaceRecognitionConfig, ReIDConfig
from src.recognition.identity_linker import IdentityLinker, TrackIdentityState


def _make_identity_linker(
    gallery_recheck_interval: float = 0.0,
) -> tuple[IdentityLinker, MagicMock, MagicMock]:
    """Create an IdentityLinker with mocked dependencies.

    Returns:
        (linker, mock_repository, mock_gallery_manager)
    """
    reid_config = ReIDConfig(
        enabled=True,
        gallery_recheck_interval=gallery_recheck_interval,
    )
    face_config = FaceRecognitionConfig(enabled=False)
    repo = MagicMock()

    # Create linker with mocked extractor
    mock_reid_extractor = MagicMock()
    mock_reid_extractor.extract.return_value = (np.zeros(512), 0.9)

    linker = IdentityLinker(
        face_config=face_config,
        reid_config=reid_config,
        repository=repo,
        reid_extractor=mock_reid_extractor,
    )

    # Mock the gallery manager
    mock_gallery = MagicMock()
    linker._id_manager._reid_gallery_manager = mock_gallery

    return linker, repo, mock_gallery


class TestGalleryRecheck:
    """Test _try_gallery_identification in IdentityLinker."""

    def test_unidentified_track_gets_identified(self):
        """Unidentified track matched via gallery recheck."""
        linker, repo, mock_gallery = _make_identity_linker(gallery_recheck_interval=0.0)

        # Register unidentified track
        state = linker.register_track("global_1")

        # Mock gallery match
        match_result = MagicMock()
        match_result.matched = True
        match_result.person_name = "Alice"
        match_result.score = 0.75
        match_result.best_score = 0.75
        match_result.db_id = 42
        mock_gallery.match_new_track.return_value = match_result

        # Mock person lookup
        mock_person = MagicMock()
        mock_person.id = 10
        repo.get_person_by_name.return_value = mock_person

        crop = np.zeros((100, 50, 3), dtype=np.uint8)
        success = linker._try_gallery_identification(state, crop, 1, "cam_a")

        assert success is True
        assert state.person_id == 10
        assert state.identified_by == "reid_gallery"
        assert state.identification_confidence == 0.75

        # Check DB updated
        repo.update_track.assert_called_with("global_1", person_id=10)

        # Check event emitted
        repo.create_event.assert_called_once()
        call_kwargs = repo.create_event.call_args[1]
        assert call_kwargs["event_type"] == "reid_match"
        assert call_kwargs["extra_data"]["match_policy"] == "gallery_recheck"

    def test_gallery_recheck_respects_interval(self):
        """Gallery recheck should not fire before interval elapses."""
        linker, repo, mock_gallery = _make_identity_linker(gallery_recheck_interval=10.0)

        state = linker.register_track("global_1")
        # Set last check to now
        state.last_gallery_check = time.time()

        crop = np.zeros((100, 50, 3), dtype=np.uint8)
        success = linker._try_gallery_identification(state, crop, 1, "cam_a")

        assert success is False
        # Gallery should not be called
        mock_gallery.match_new_track.assert_not_called()

    def test_gallery_recheck_fires_after_interval(self):
        """Gallery recheck should fire when interval has elapsed."""
        linker, repo, mock_gallery = _make_identity_linker(gallery_recheck_interval=5.0)

        state = linker.register_track("global_1")
        # Set last check to 10 seconds ago
        state.last_gallery_check = time.time() - 10.0

        # Mock no match
        match_result = MagicMock()
        match_result.matched = False
        mock_gallery.match_new_track.return_value = match_result

        crop = np.zeros((100, 50, 3), dtype=np.uint8)
        success = linker._try_gallery_identification(state, crop, 1, "cam_a")

        assert success is False
        # Gallery should have been called
        mock_gallery.match_new_track.assert_called_once()

    def test_already_identified_skips_gallery_recheck(self):
        """Identified tracks should not trigger gallery recheck in process_track."""
        linker, repo, mock_gallery = _make_identity_linker(gallery_recheck_interval=0.0)

        state = linker.register_track("global_1")
        state.person_id = 10
        state.identified_by = "face"
        state.identification_confidence = 0.9

        crop = np.zeros((100, 50, 3), dtype=np.uint8)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # process_track should return early for identified tracks
        result = linker.process_track(
            "global_1", frame, crop, (10, 10, 50, 90),
            num_persons_in_frame=1,
        )

        # Should not call gallery recheck
        mock_gallery.match_new_track.assert_not_called()
        assert result.is_confirmed is True
        assert result.person_id == 10

    def test_no_gallery_manager_no_crash(self):
        """No gallery manager configured -> returns False gracefully."""
        linker, repo, mock_gallery = _make_identity_linker(gallery_recheck_interval=0.0)

        # Remove gallery manager
        linker._id_manager._reid_gallery_manager = None

        state = linker.register_track("global_1")
        crop = np.zeros((100, 50, 3), dtype=np.uint8)

        success = linker._try_gallery_identification(state, crop, 1, "cam_a")
        assert success is False

    def test_gallery_match_no_person_in_db(self):
        """Gallery matches person name but person not found in DB."""
        linker, repo, mock_gallery = _make_identity_linker(gallery_recheck_interval=0.0)

        state = linker.register_track("global_1")

        match_result = MagicMock()
        match_result.matched = True
        match_result.person_name = "Unknown_Person"
        match_result.score = 0.7
        match_result.best_score = 0.7
        match_result.db_id = None
        mock_gallery.match_new_track.return_value = match_result

        # Person not found
        repo.get_person_by_name.return_value = None

        crop = np.zeros((100, 50, 3), dtype=np.uint8)
        success = linker._try_gallery_identification(state, crop, 1, "cam_a")

        assert success is False
        assert state.person_id is None

    def test_gallery_recheck_integrated_in_process_track(self):
        """Gallery recheck fires within process_track for unidentified tracks."""
        linker, repo, mock_gallery = _make_identity_linker(gallery_recheck_interval=0.0)

        state = linker.register_track("global_1")

        # Mock gallery match
        match_result = MagicMock()
        match_result.matched = True
        match_result.person_name = "Bob"
        match_result.score = 0.8
        match_result.best_score = 0.8
        match_result.db_id = 5
        mock_gallery.match_new_track.return_value = match_result

        mock_person = MagicMock()
        mock_person.id = 20
        repo.get_person_by_name.return_value = mock_person

        crop = np.zeros((100, 50, 3), dtype=np.uint8)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        result = linker.process_track(
            "global_1", frame, crop, (10, 10, 50, 90),
            num_persons_in_frame=1,
            camera_id="cam_a",
        )

        assert result.is_confirmed is True
        assert result.person_id == 20
        assert result.method == "reid_gallery"
        assert state.identified_by == "reid_gallery"
