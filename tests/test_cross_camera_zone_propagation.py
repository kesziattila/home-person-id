"""Tests for cross-camera zone identity propagation.

Covers:
- Identity propagation from face-identified track to unidentified track in shared zone
- No propagation when multiple persons are on one camera
- No propagation when both tracks are already identified
- Face-identified track not downgraded by handover propagation
- No propagation when no zones are configured
"""

import time
from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.config import (
    CameraTopologyConfig,
    FaceRecognitionConfig,
    ReIDConfig,
    ZoneConfig,
    ZonePolygon,
    ZonesConfig,
)
from src.recognition.identity_linker import IdentityLinker, TrackIdentityState
from src.tracking.global_tracker import GlobalTrackManager, PendingHandover
from src.tracking.track import GlobalTrack, LocalTrack, TrackState


def _make_zone_config(zone_name: str, camera_ids: list[str]) -> ZonesConfig:
    """Create a ZonesConfig with one zone covering full frame on each camera."""
    cameras = {}
    for cam_id in camera_ids:
        # Full-frame polygon
        cameras[cam_id] = ZonePolygon(polygon=[[0, 0], [1, 0], [1, 1], [0, 1]])
    return ZonesConfig(zones=[ZoneConfig(name=zone_name, cameras=cameras)])


def _make_global_track_manager(
    zones_config: ZonesConfig = None,
    cross_camera_interval: float = 0.0,
) -> GlobalTrackManager:
    """Create a GlobalTrackManager with mocked dependencies."""
    topology = CameraTopologyConfig(use_zones=True)
    reid_config = ReIDConfig(enabled=False, cross_camera_interval=cross_camera_interval)
    face_config = FaceRecognitionConfig(enabled=False)
    repo = MagicMock()

    identity_linker = IdentityLinker(face_config, reid_config, repo)

    manager = GlobalTrackManager(
        topology_config=topology,
        reid_config=reid_config,
        identity_linker=identity_linker,
        repository=repo,
        zones_config=zones_config,
    )
    return manager


def _register_active_track(
    manager: GlobalTrackManager,
    camera_id: str,
    local_track_id: int,
    global_track_id: str,
    bbox: tuple[float, float, float, float],
    person_id: int = None,
    identified_by: str = "none",
    confidence: float = 0.0,
):
    """Register an active track in the manager and identity linker."""
    # Create global track
    track = GlobalTrack(
        track_id=global_track_id,
        state=TrackState.TRACKED,
        current_camera_id=camera_id,
        current_local_track_id=local_track_id,
        cameras_seen=[camera_id],
    )
    manager._tracks[global_track_id] = track
    manager._local_to_global[(camera_id, local_track_id)] = global_track_id
    manager._last_bboxes[(camera_id, local_track_id)] = bbox
    manager._frame_dimensions[camera_id] = (1920, 1080)

    # Register identity state
    state = manager.identity_linker.register_track(global_track_id)
    if person_id is not None:
        state.person_id = person_id
        state.identified_by = identified_by
        state.identification_confidence = confidence

    return state


class TestZoneIdentityPropagation:
    """Test _try_zone_identity_propagation."""

    def test_propagates_identity_from_identified_to_unidentified(self):
        """Face-identified on cam A, unidentified on cam B -> propagate."""
        zones = _make_zone_config("living_room", ["cam_a", "cam_b"])
        manager = _make_global_track_manager(zones_config=zones)

        # cam_a: face-identified person
        _register_active_track(
            manager, "cam_a", 1, "global_1",
            bbox=(100, 100, 300, 500),
            person_id=42, identified_by="face", confidence=0.9,
        )

        # cam_b: unidentified person
        state_b = _register_active_track(
            manager, "cam_b", 1, "global_2",
            bbox=(200, 150, 400, 600),
        )

        manager._try_zone_identity_propagation()

        assert state_b.person_id == 42
        assert state_b.identified_by == "handover"

    def test_propagates_from_b_to_a(self):
        """Face-identified on cam B, unidentified on cam A -> propagate."""
        zones = _make_zone_config("living_room", ["cam_a", "cam_b"])
        manager = _make_global_track_manager(zones_config=zones)

        # cam_a: unidentified
        state_a = _register_active_track(
            manager, "cam_a", 1, "global_1",
            bbox=(100, 100, 300, 500),
        )

        # cam_b: face-identified
        _register_active_track(
            manager, "cam_b", 1, "global_2",
            bbox=(200, 150, 400, 600),
            person_id=7, identified_by="face", confidence=0.85,
        )

        manager._try_zone_identity_propagation()

        assert state_a.person_id == 7
        assert state_a.identified_by == "handover"

    def test_no_propagation_multiple_persons_on_one_camera(self):
        """Two persons on cam_a in the zone -> no propagation."""
        zones = _make_zone_config("living_room", ["cam_a", "cam_b"])
        manager = _make_global_track_manager(zones_config=zones)

        # cam_a: 2 persons
        _register_active_track(
            manager, "cam_a", 1, "global_1",
            bbox=(100, 100, 300, 500),
            person_id=42, identified_by="face", confidence=0.9,
        )
        _register_active_track(
            manager, "cam_a", 2, "global_3",
            bbox=(500, 100, 700, 500),
        )

        # cam_b: 1 unidentified person
        state_b = _register_active_track(
            manager, "cam_b", 1, "global_2",
            bbox=(200, 150, 400, 600),
        )

        manager._try_zone_identity_propagation()

        # Should NOT propagate because cam_a has 2 persons in the zone
        assert state_b.person_id is None
        assert state_b.identified_by == "none"

    def test_no_propagation_both_identified(self):
        """Both tracks already identified -> no change."""
        zones = _make_zone_config("living_room", ["cam_a", "cam_b"])
        manager = _make_global_track_manager(zones_config=zones)

        _register_active_track(
            manager, "cam_a", 1, "global_1",
            bbox=(100, 100, 300, 500),
            person_id=42, identified_by="face", confidence=0.9,
        )

        state_b = _register_active_track(
            manager, "cam_b", 1, "global_2",
            bbox=(200, 150, 400, 600),
            person_id=99, identified_by="reid", confidence=0.8,
        )

        manager._try_zone_identity_propagation()

        # state_b should keep its own identity
        assert state_b.person_id == 99
        assert state_b.identified_by == "reid"

    def test_no_propagation_both_unidentified(self):
        """Both tracks unidentified -> no change (nothing to propagate)."""
        zones = _make_zone_config("living_room", ["cam_a", "cam_b"])
        manager = _make_global_track_manager(zones_config=zones)

        state_a = _register_active_track(
            manager, "cam_a", 1, "global_1",
            bbox=(100, 100, 300, 500),
        )
        state_b = _register_active_track(
            manager, "cam_b", 1, "global_2",
            bbox=(200, 150, 400, 600),
        )

        manager._try_zone_identity_propagation()

        assert state_a.person_id is None
        assert state_b.person_id is None

    def test_face_identified_not_downgraded_by_propagation(self):
        """Face-identified track on cam B should not be overwritten by handover."""
        zones = _make_zone_config("living_room", ["cam_a", "cam_b"])
        manager = _make_global_track_manager(zones_config=zones)

        # cam_a: face-identified as person 42
        _register_active_track(
            manager, "cam_a", 1, "global_1",
            bbox=(100, 100, 300, 500),
            person_id=42, identified_by="face", confidence=0.9,
        )

        # cam_b: already face-identified as person 99
        state_b = _register_active_track(
            manager, "cam_b", 1, "global_2",
            bbox=(200, 150, 400, 600),
            person_id=99, identified_by="face", confidence=0.85,
        )

        manager._try_zone_identity_propagation()

        # Both identified -> no propagation attempt
        assert state_b.person_id == 99
        assert state_b.identified_by == "face"

    def test_no_propagation_without_zone_manager(self):
        """No zone manager configured -> method returns immediately."""
        manager = _make_global_track_manager(zones_config=None)

        state_a = _register_active_track(
            manager, "cam_a", 1, "global_1",
            bbox=(100, 100, 300, 500),
            person_id=42, identified_by="face", confidence=0.9,
        )
        state_b = _register_active_track(
            manager, "cam_b", 1, "global_2",
            bbox=(200, 150, 400, 600),
        )

        manager._try_zone_identity_propagation()

        # No propagation
        assert state_b.person_id is None

    def test_db_updated_on_propagation(self):
        """Repository.update_track is called when identity is propagated."""
        zones = _make_zone_config("living_room", ["cam_a", "cam_b"])
        manager = _make_global_track_manager(zones_config=zones)

        _register_active_track(
            manager, "cam_a", 1, "global_1",
            bbox=(100, 100, 300, 500),
            person_id=42, identified_by="face", confidence=0.9,
        )
        _register_active_track(
            manager, "cam_b", 1, "global_2",
            bbox=(200, 150, 400, 600),
        )

        manager._try_zone_identity_propagation()

        manager.repository.update_track.assert_called_once_with("global_2", person_id=42)

    def test_lost_tracks_excluded(self):
        """LOST tracks should not participate in propagation."""
        zones = _make_zone_config("living_room", ["cam_a", "cam_b"])
        manager = _make_global_track_manager(zones_config=zones)

        _register_active_track(
            manager, "cam_a", 1, "global_1",
            bbox=(100, 100, 300, 500),
            person_id=42, identified_by="face", confidence=0.9,
        )
        # Mark as LOST
        manager._tracks["global_1"].state = TrackState.LOST

        state_b = _register_active_track(
            manager, "cam_b", 1, "global_2",
            bbox=(200, 150, 400, 600),
        )

        manager._try_zone_identity_propagation()

        # Lost track should be excluded
        assert state_b.person_id is None

    def test_three_cameras_one_zone(self):
        """Three cameras sharing a zone, one identified -> propagates to both others."""
        zones = _make_zone_config("hallway", ["cam_a", "cam_b", "cam_c"])
        manager = _make_global_track_manager(zones_config=zones)

        _register_active_track(
            manager, "cam_a", 1, "global_1",
            bbox=(100, 100, 300, 500),
            person_id=42, identified_by="face", confidence=0.9,
        )

        state_b = _register_active_track(
            manager, "cam_b", 1, "global_2",
            bbox=(200, 150, 400, 600),
        )

        state_c = _register_active_track(
            manager, "cam_c", 1, "global_3",
            bbox=(300, 200, 500, 700),
        )

        manager._try_zone_identity_propagation()

        assert state_b.person_id == 42
        assert state_b.identified_by == "handover"
        assert state_c.person_id == 42
        assert state_c.identified_by == "handover"


class TestZonePropagationThrottling:
    """Test the throttle mechanism in process_local_tracks."""

    def test_throttle_respects_interval(self):
        """Propagation only runs when interval has elapsed."""
        zones = _make_zone_config("living_room", ["cam_a", "cam_b"])
        manager = _make_global_track_manager(
            zones_config=zones, cross_camera_interval=10.0
        )

        _register_active_track(
            manager, "cam_a", 1, "global_1",
            bbox=(100, 100, 300, 500),
            person_id=42, identified_by="face", confidence=0.9,
        )
        state_b = _register_active_track(
            manager, "cam_b", 1, "global_2",
            bbox=(200, 150, 400, 600),
        )

        # First call — should set _last_cross_camera_check but skip propagation
        # because interval is 10s and _last_cross_camera_check starts at 0
        # Actually, time.time() - 0 > 10.0 is True, so it WILL run on first call
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        manager.process_local_tracks(
            "cam_a",
            [LocalTrack(track_id=1, camera_id="cam_a", bbox=(100, 100, 300, 500), confidence=0.9)],
            frame,
            new_track_ids=[],
            lost_track_ids=[],
        )

        # After first call, identity should have been propagated
        assert state_b.person_id == 42

    def test_no_propagation_when_interval_not_elapsed(self):
        """Propagation should not run if interval hasn't elapsed."""
        zones = _make_zone_config("living_room", ["cam_a", "cam_b"])
        manager = _make_global_track_manager(
            zones_config=zones, cross_camera_interval=10.0
        )

        # Set last check to now (simulating it just ran)
        manager._last_cross_camera_check = time.time()

        _register_active_track(
            manager, "cam_a", 1, "global_1",
            bbox=(100, 100, 300, 500),
            person_id=42, identified_by="face", confidence=0.9,
        )
        state_b = _register_active_track(
            manager, "cam_b", 1, "global_2",
            bbox=(200, 150, 400, 600),
        )

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        manager.process_local_tracks(
            "cam_a",
            [LocalTrack(track_id=1, camera_id="cam_a", bbox=(100, 100, 300, 500), confidence=0.9)],
            frame,
            new_track_ids=[],
            lost_track_ids=[],
        )

        # Should NOT propagate because interval hasn't elapsed
        assert state_b.person_id is None


class TestTrackRecoveryAfterBriefLoss:
    """Test track recovery when ByteTrack re-detects after brief loss.

    Covers the scenario where a stationary person is briefly lost by ByteTrack
    (e.g., 1 missed detection frame) but then re-detected with the same local ID.
    The track should be restored to TRACKED instead of timing out.
    """

    def test_track_recovered_after_brief_loss(self):
        """Track LOST -> ByteTrack re-reports same local ID -> restored to TRACKED."""
        zones = _make_zone_config("living_room", ["cam_a", "cam_b"])
        manager = _make_global_track_manager(zones_config=zones)

        # Register a tracked person on cam_b
        _register_active_track(
            manager, "cam_b", 5, "global_1",
            bbox=(100, 100, 300, 500),
        )

        # Simulate brief loss: mark track as LOST and create PendingHandover
        manager._tracks["global_1"].state = TrackState.LOST
        manager._recently_lost_tracks.add("global_1")
        manager._pending_handovers.append(PendingHandover(
            global_track_id="global_1",
            from_camera="cam_b",
            zone_name="living_room",
            exit_time=time.time(),
            max_handover_sec=5.0,
            local_track_id=5,
        ))

        # ByteTrack re-detects same local ID (not in new_track_ids — it's a continued track)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = manager.process_local_tracks(
            "cam_b",
            [LocalTrack(track_id=5, camera_id="cam_b", bbox=(100, 100, 300, 500), confidence=0.9)],
            frame,
            new_track_ids=[],
            lost_track_ids=[],
        )

        # Track should be restored to TRACKED
        assert manager._tracks["global_1"].state == TrackState.TRACKED
        assert "global_1" not in manager._recently_lost_tracks
        # Should appear in active tracks
        active_ids = [t.track_id for t in result.active_tracks]
        assert "global_1" in active_ids

    def test_pending_handover_cancelled_on_recovery(self):
        """PendingHandover removed when track recovers on same camera."""
        zones = _make_zone_config("living_room", ["cam_a", "cam_b"])
        manager = _make_global_track_manager(zones_config=zones)

        _register_active_track(
            manager, "cam_b", 5, "global_1",
            bbox=(100, 100, 300, 500),
        )

        # Create pending handover (simulating _handle_lost_local_track)
        manager._tracks["global_1"].state = TrackState.LOST
        manager._recently_lost_tracks.add("global_1")
        manager._pending_handovers.append(PendingHandover(
            global_track_id="global_1",
            from_camera="cam_b",
            zone_name="living_room",
            exit_time=time.time(),
            max_handover_sec=5.0,
            local_track_id=5,
        ))
        assert len(manager._pending_handovers) == 1

        # ByteTrack re-detects
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        manager.process_local_tracks(
            "cam_b",
            [LocalTrack(track_id=5, camera_id="cam_b", bbox=(100, 100, 300, 500), confidence=0.9)],
            frame,
            new_track_ids=[],
            lost_track_ids=[],
        )

        # PendingHandover should be cancelled
        assert len(manager._pending_handovers) == 0

    def test_has_active_tracks_includes_pending_handovers(self):
        """has_active_tracks returns True when pending handover exists from camera."""
        zones = _make_zone_config("living_room", ["cam_a", "cam_b"])
        manager = _make_global_track_manager(zones_config=zones)

        # No tracks, no pending -> False
        assert manager.has_active_tracks("cam_b") is False

        # Add pending handover from cam_b
        manager._pending_handovers.append(PendingHandover(
            global_track_id="global_1",
            from_camera="cam_b",
            zone_name="living_room",
            exit_time=time.time(),
            max_handover_sec=5.0,
            local_track_id=5,
        ))

        # Now should return True (keeps detection gate open)
        assert manager.has_active_tracks("cam_b") is True
        # But not for other cameras
        assert manager.has_active_tracks("cam_a") is False

    def test_mapping_cleaned_on_handover_timeout(self):
        """local-to-global mapping removed when PendingHandover times out."""
        zones = _make_zone_config("living_room", ["cam_a", "cam_b"])
        manager = _make_global_track_manager(zones_config=zones)

        _register_active_track(
            manager, "cam_b", 5, "global_1",
            bbox=(100, 100, 300, 500),
        )

        # Simulate _handle_lost_local_track with mapping kept alive
        manager._tracks["global_1"].state = TrackState.LOST
        manager._recently_lost_tracks.add("global_1")
        manager._pending_handovers.append(PendingHandover(
            global_track_id="global_1",
            from_camera="cam_b",
            zone_name="living_room",
            exit_time=time.time() - 10.0,  # Already expired
            max_handover_sec=5.0,
            local_track_id=5,
        ))

        # Mapping should still exist
        assert ("cam_b", 5) in manager._local_to_global

        # Run cleanup (called at start of process_local_tracks)
        manager._cleanup_pending_handovers(time.time())

        # Mapping should now be cleaned up
        assert ("cam_b", 5) not in manager._local_to_global
        assert ("cam_b", 5) not in manager._last_bboxes

    def test_mapping_cleaned_on_handover_match(self):
        """Old camera mapping removed when handover matches on new camera."""
        zones = _make_zone_config("living_room", ["cam_a", "cam_b"])
        manager = _make_global_track_manager(zones_config=zones)

        _register_active_track(
            manager, "cam_b", 5, "global_1",
            bbox=(100, 100, 300, 500),
        )

        # Simulate _handle_lost_local_track: mapping kept alive, pending handover created
        manager._tracks["global_1"].state = TrackState.LOST
        manager._recently_lost_tracks.add("global_1")
        manager._pending_handovers.append(PendingHandover(
            global_track_id="global_1",
            from_camera="cam_b",
            zone_name="living_room",
            exit_time=time.time(),
            max_handover_sec=5.0,
            local_track_id=5,
        ))

        # New track appears on cam_a -> handover match
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        manager.process_local_tracks(
            "cam_a",
            [LocalTrack(track_id=10, camera_id="cam_a", bbox=(200, 150, 400, 600), confidence=0.9)],
            frame,
            new_track_ids=[10],
            lost_track_ids=[],
        )

        # Old cam_b mapping should be removed
        assert ("cam_b", 5) not in manager._local_to_global
        # New cam_a mapping should exist
        assert ("cam_a", 10) in manager._local_to_global
        assert manager._local_to_global[("cam_a", 10)] == "global_1"
        # Track should be TRACKED on cam_a
        assert manager._tracks["global_1"].state == TrackState.TRACKED

    def test_normal_handover_still_works(self):
        """Regression: person leaves camera, appears on another -> handover succeeds."""
        zones = _make_zone_config("living_room", ["cam_a", "cam_b"])
        manager = _make_global_track_manager(zones_config=zones)

        _register_active_track(
            manager, "cam_a", 1, "global_1",
            bbox=(100, 100, 300, 500),
        )

        # Track lost on cam_a (leaves view) -> creates pending handover
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        manager.process_local_tracks(
            "cam_a",
            [],  # No tracks — person left
            frame,
            new_track_ids=[],
            lost_track_ids=[1],
        )

        # Should have pending handover
        assert len(manager._pending_handovers) == 1
        assert manager._pending_handovers[0].global_track_id == "global_1"
        assert manager._pending_handovers[0].from_camera == "cam_a"
        assert manager._tracks["global_1"].state == TrackState.LOST

        # Mapping should still exist (kept alive for recovery)
        assert ("cam_a", 1) in manager._local_to_global

        # New track appears on cam_b -> handover match
        result = manager.process_local_tracks(
            "cam_b",
            [LocalTrack(track_id=10, camera_id="cam_b", bbox=(200, 150, 400, 600), confidence=0.9)],
            frame,
            new_track_ids=[10],
            lost_track_ids=[],
        )

        # Handover should complete
        assert len(result.handovers_completed) == 1
        assert result.handovers_completed[0] == ("global_1", "cam_a", "cam_b")
        # Track should be TRACKED on cam_b
        assert manager._tracks["global_1"].state == TrackState.TRACKED
        # Old cam_a mapping should be cleaned up
        assert ("cam_a", 1) not in manager._local_to_global
        # New cam_b mapping should exist
        assert ("cam_b", 10) in manager._local_to_global
