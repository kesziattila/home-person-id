"""Tests for Re-ID cross-camera matching fix and identification hierarchy.

Covers:
- confirm_identity() precedence (upgrade/downgrade behavior)
- _find_track_for_person() helper
- match_reid_cross_camera() restructured logic
- CrossCameraMatch type with Optional matched_track_id
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config import FaceRecognitionConfig, ReIDConfig
from src.recognition.identity_linker import (
    IDENTIFICATION_PRECEDENCE,
    IdentityLinker,
    TrackIdentityState,
)
from src.recognition.identification_manager import CrossCameraMatch


# ==================== IDENTIFICATION_PRECEDENCE ====================


class TestIdentificationPrecedence:
    """Test the precedence constant."""

    def test_precedence_order(self):
        assert IDENTIFICATION_PRECEDENCE["none"] < IDENTIFICATION_PRECEDENCE["handover"]
        assert IDENTIFICATION_PRECEDENCE["handover"] < IDENTIFICATION_PRECEDENCE["reid"]
        assert IDENTIFICATION_PRECEDENCE["reid"] < IDENTIFICATION_PRECEDENCE["face"]

    def test_reid_and_reid_gallery_equal(self):
        assert IDENTIFICATION_PRECEDENCE["reid"] == IDENTIFICATION_PRECEDENCE["reid_gallery"]


# ==================== confirm_identity() ====================


class TestConfirmIdentity:
    """Test confirm_identity with precedence guard."""

    def _make_state(self, method: str = "none", person_id=None) -> TrackIdentityState:
        state = TrackIdentityState(global_track_id="test_track")
        if person_id is not None:
            state.person_id = person_id
            state.identified_by = method
            state.identification_confidence = 0.5
        return state

    def test_upgrade_none_to_handover(self):
        state = self._make_state("none")
        state.confirm_identity(1, 0.7, "handover")
        assert state.person_id == 1
        assert state.identified_by == "handover"

    def test_upgrade_handover_to_reid(self):
        state = self._make_state("handover", person_id=1)
        state.confirm_identity(1, 0.8, "reid")
        assert state.identified_by == "reid"
        assert state.identification_confidence == 0.8

    def test_upgrade_reid_to_face(self):
        state = self._make_state("reid", person_id=1)
        state.confirm_identity(1, 0.9, "face")
        assert state.identified_by == "face"
        assert state.identification_confidence == 0.9

    def test_upgrade_handover_to_face(self):
        state = self._make_state("handover", person_id=1)
        state.confirm_identity(1, 0.85, "face")
        assert state.identified_by == "face"

    def test_downgrade_face_to_reid_blocked(self):
        state = self._make_state("face", person_id=1)
        state.identification_confidence = 0.9
        state.confirm_identity(2, 0.7, "reid")
        # Should NOT change
        assert state.person_id == 1
        assert state.identified_by == "face"
        assert state.identification_confidence == 0.9

    def test_downgrade_face_to_handover_blocked(self):
        state = self._make_state("face", person_id=1)
        state.confirm_identity(2, 0.6, "handover")
        assert state.person_id == 1
        assert state.identified_by == "face"

    def test_downgrade_reid_to_handover_blocked(self):
        state = self._make_state("reid", person_id=1)
        state.confirm_identity(2, 0.6, "handover")
        assert state.person_id == 1
        assert state.identified_by == "reid"

    def test_same_rank_reid_to_reid_gallery_allowed(self):
        state = self._make_state("reid", person_id=1)
        state.confirm_identity(2, 0.8, "reid_gallery")
        # Same rank — should update
        assert state.person_id == 2
        assert state.identified_by == "reid_gallery"

    def test_upgrade_none_to_face(self):
        state = self._make_state("none")
        state.confirm_identity(5, 0.95, "face")
        assert state.person_id == 5
        assert state.identified_by == "face"
        assert state.identified_at is not None


# ==================== CrossCameraMatch ====================


class TestCrossCameraMatch:
    """Test CrossCameraMatch dataclass."""

    def test_default_matched_track_id_is_none(self):
        match = CrossCameraMatch(similarity=0.8, person_name="Alice")
        assert match.matched_track_id is None

    def test_with_matched_track_id(self):
        match = CrossCameraMatch(
            matched_track_id="global_42", similarity=0.8, person_name="Alice"
        )
        assert match.matched_track_id == "global_42"

    def test_gallery_only_match(self):
        """Gallery-only match: has person_name but no matched_track_id."""
        match = CrossCameraMatch(similarity=0.75, person_name="Bob")
        assert match.matched_track_id is None
        assert match.person_name == "Bob"
        assert match.similarity == 0.75


# ==================== _find_track_for_person() ====================


class TestFindTrackForPerson:
    """Test _find_track_for_person helper."""

    def _make_linker(self):
        """Create a minimal IdentityLinker with mocked dependencies."""
        face_config = FaceRecognitionConfig(enabled=False)
        reid_config = ReIDConfig(enabled=False)
        repo = MagicMock()
        linker = IdentityLinker(face_config, reid_config, repo)
        return linker

    def test_finds_matching_track(self):
        linker = self._make_linker()
        # Register tracks
        state1 = linker.register_track("global_1")
        state1.person_id = 10
        state1.identified_by = "face"

        state2 = linker.register_track("global_2")
        state2.person_id = 20
        state2.identified_by = "face"

        # Mock _get_person_name
        linker._get_person_name = MagicMock(side_effect=lambda pid: {10: "Alice", 20: "Bob"}.get(pid))

        result = linker._find_track_for_person("Alice", ["global_1", "global_2"])
        assert result == "global_1"

    def test_returns_none_when_no_match(self):
        linker = self._make_linker()
        state1 = linker.register_track("global_1")
        state1.person_id = 10
        state1.identified_by = "face"

        linker._get_person_name = MagicMock(return_value="Alice")

        result = linker._find_track_for_person("Bob", ["global_1"])
        assert result is None

    def test_skips_unidentified_tracks(self):
        linker = self._make_linker()
        # Unidentified track (person_id is None)
        linker.register_track("global_1")

        linker._get_person_name = MagicMock(return_value=None)

        result = linker._find_track_for_person("Alice", ["global_1"])
        assert result is None

    def test_handles_missing_track(self):
        linker = self._make_linker()
        result = linker._find_track_for_person("Alice", ["nonexistent_track"])
        assert result is None


# ==================== match_reid_cross_camera() restructured ====================


class TestMatchReidCrossCamera:
    """Test restructured match_reid_cross_camera()."""

    def _make_linker(self):
        face_config = FaceRecognitionConfig(enabled=False)
        reid_config = ReIDConfig(enabled=True, similarity_threshold=0.65, min_visibility=0.3)
        repo = MagicMock()
        linker = IdentityLinker(face_config, reid_config, repo)
        return linker

    def test_per_track_match_wins(self):
        """Per-track match with high similarity should be returned."""
        linker = self._make_linker()

        # Register candidate track with a gallery
        state = linker.register_track("global_1")
        state.person_id = 10
        state.identified_by = "face"

        # Add a fake embedding to the per-track gallery
        fake_emb = np.random.randn(512).astype(np.float32)
        fake_emb /= np.linalg.norm(fake_emb)
        state.reid_gallery.add(fake_emb, 0.9, time.time())

        linker._get_person_name = MagicMock(return_value="Alice")

        # Mock reid_extractor to return the same embedding (perfect match)
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = (fake_emb, 0.9)
        linker._id_manager._reid_extractor = mock_extractor

        # No shared gallery
        linker._id_manager._reid_gallery_manager = None

        result = linker.match_reid_cross_camera(
            "new_track", np.zeros((100, 50, 3), dtype=np.uint8),
            ["global_1"],
            precomputed_reid=(fake_emb, 0.9),
        )

        assert result is not None
        assert result.matched_track_id == "global_1"
        assert result.person_name == "Alice"
        assert result.similarity > 0.65

    def test_gallery_match_resolves_to_candidate_track(self):
        """Gallery match should resolve to a candidate track if possible."""
        linker = self._make_linker()

        # Register candidate track
        state = linker.register_track("global_1")
        state.person_id = 10
        state.identified_by = "face"

        linker._get_person_name = MagicMock(return_value="Alice")

        # Mock gallery manager returning a match
        mock_gallery_mgr = MagicMock()
        mock_match_result = MagicMock()
        mock_match_result.matched = True
        mock_match_result.score = 0.8
        mock_match_result.person_name = "Alice"
        mock_match_result.best_score = 0.8
        mock_match_result.db_id = 42
        mock_gallery_mgr.match_new_track.return_value = mock_match_result
        linker._id_manager._reid_gallery_manager = mock_gallery_mgr

        # No per-track gallery (empty)
        # The candidate track has no reid_gallery entries, so per-track won't match

        result = linker.match_reid_cross_camera(
            "new_track", np.zeros((100, 50, 3), dtype=np.uint8),
            ["global_1"],
            precomputed_reid=(np.random.randn(512).astype(np.float32), 0.9),
        )

        assert result is not None
        assert result.matched_track_id == "global_1"  # Resolved!
        assert result.person_name == "Alice"

    def test_gallery_match_without_candidate_returns_person_only(self):
        """Gallery match with no matching candidate returns person_name but no track."""
        linker = self._make_linker()

        linker._get_person_name = MagicMock(return_value=None)

        # Mock gallery manager returning a match
        mock_gallery_mgr = MagicMock()
        mock_match_result = MagicMock()
        mock_match_result.matched = True
        mock_match_result.score = 0.8
        mock_match_result.person_name = "Alice"
        mock_match_result.best_score = 0.8
        mock_match_result.db_id = 42
        mock_gallery_mgr.match_new_track.return_value = mock_match_result
        linker._id_manager._reid_gallery_manager = mock_gallery_mgr

        # No candidates
        result = linker.match_reid_cross_camera(
            "new_track", np.zeros((100, 50, 3), dtype=np.uint8),
            [],
            precomputed_reid=(np.random.randn(512).astype(np.float32), 0.9),
        )

        assert result is not None
        assert result.matched_track_id is None
        assert result.person_name == "Alice"

    def test_no_match_returns_none(self):
        """No match should return None."""
        linker = self._make_linker()

        # No gallery manager
        linker._id_manager._reid_gallery_manager = None

        result = linker.match_reid_cross_camera(
            "new_track", np.zeros((100, 50, 3), dtype=np.uint8),
            [],
            precomputed_reid=(np.random.randn(512).astype(np.float32), 0.9),
        )

        assert result is None

    def test_multi_person_skipped(self):
        """Multiple persons in frame should skip matching."""
        linker = self._make_linker()

        result = linker.match_reid_cross_camera(
            "new_track", np.zeros((100, 50, 3), dtype=np.uint8),
            [],
            num_persons_in_frame=2,
        )

        assert result is None

    def test_gallery_does_not_overwrite_per_track_match(self):
        """A gallery match with lower score should not overwrite a per-track match."""
        linker = self._make_linker()

        # Register candidate track with high-scoring per-track match
        state = linker.register_track("global_1")
        state.person_id = 10
        state.identified_by = "face"

        fake_emb = np.random.randn(512).astype(np.float32)
        fake_emb /= np.linalg.norm(fake_emb)
        state.reid_gallery.add(fake_emb, 0.9, time.time())

        linker._get_person_name = MagicMock(return_value="Alice")

        # Mock gallery manager returning a lower-score match
        mock_gallery_mgr = MagicMock()
        mock_match_result = MagicMock()
        mock_match_result.matched = True
        mock_match_result.score = 0.7  # Lower than per-track (which will be ~1.0)
        mock_match_result.person_name = "Bob"
        mock_match_result.best_score = 0.7
        mock_match_result.db_id = 99
        mock_gallery_mgr.match_new_track.return_value = mock_match_result
        linker._id_manager._reid_gallery_manager = mock_gallery_mgr

        result = linker.match_reid_cross_camera(
            "new_track", np.zeros((100, 50, 3), dtype=np.uint8),
            ["global_1"],
            precomputed_reid=(fake_emb, 0.9),
        )

        assert result is not None
        assert result.matched_track_id == "global_1"
        assert result.person_name == "Alice"  # Per-track winner, not gallery's "Bob"


# ==================== transfer_identity uses "handover" ====================


class TestTransferIdentityHandover:
    """Test that transfer_identity now uses 'handover' method."""

    def test_transfer_sets_handover_method(self):
        face_config = FaceRecognitionConfig(enabled=False)
        reid_config = ReIDConfig(enabled=False)
        repo = MagicMock()
        linker = IdentityLinker(face_config, reid_config, repo)

        # Source track
        from_state = linker.register_track("global_1")
        from_state.person_id = 10
        from_state.identified_by = "face"
        from_state.identification_confidence = 0.9

        # Target track
        linker.register_track("global_2")

        linker.transfer_identity("global_1", "global_2")

        to_state = linker.get_track_state("global_2")
        assert to_state.person_id == 10
        assert to_state.identified_by == "handover"

    def test_handover_upgradeable_to_reid(self):
        face_config = FaceRecognitionConfig(enabled=False)
        reid_config = ReIDConfig(enabled=False)
        repo = MagicMock()
        linker = IdentityLinker(face_config, reid_config, repo)

        state = linker.register_track("global_1")
        state.person_id = 10
        state.identified_by = "handover"
        state.identification_confidence = 0.7

        state.confirm_identity(10, 0.8, "reid")
        assert state.identified_by == "reid"
        assert state.identification_confidence == 0.8
