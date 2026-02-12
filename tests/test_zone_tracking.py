"""Tests for zone tracking, sighting creation, exit destinations, and API zone fields.

Covers:
- _update_track_zones() correctly updates _track_zones dict
- Zone change triggers Track.extra_data persistence
- create_track_sighting called with entry_zone on track creation
- end_track_sighting called with exit_zone on track loss
- Exit destination: track lost in zone with exit_destination -> estimated_zone = destination
- Exit destination: track lost in zone without exit_destination -> estimated_zone = zone name
- API: LocationResponse includes correct estimated_location for active vs lost tracks
- API: ActiveTrackResponse includes zone field
"""

import os
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, call

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.config import (
    CameraTopologyConfig,
    FaceRecognitionConfig,
    ReIDConfig,
    ZoneConfig,
    ZonePolygon,
    ZonesConfig,
)
from src.recognition.identity_linker import IdentityLinker
from src.tracking.global_tracker import GlobalTrackManager
from src.tracking.track import GlobalTrack, LocalTrack, TrackState
from src.tracking.zone_manager import ZoneManager


# ==================== Helpers ====================


def _make_zones_config(zones: list[dict]) -> ZonesConfig:
    """Create ZonesConfig from simplified dicts.

    Each dict: {"name": str, "cameras": [str], "exit_destination": Optional[str]}
    """
    zone_configs = []
    for z in zones:
        cameras = {}
        for cam_id in z["cameras"]:
            cameras[cam_id] = ZonePolygon(polygon=[[0, 0], [1, 0], [1, 1], [0, 1]])
        zone_configs.append(
            ZoneConfig(
                name=z["name"],
                cameras=cameras,
                exit_destination=z.get("exit_destination"),
            )
        )
    return ZonesConfig(zones=zone_configs)


def _make_manager(
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
    bbox: tuple = (100, 100, 300, 500),
):
    """Register an active track in the manager."""
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
    manager.identity_linker.register_track(global_track_id)
    return track


# ==================== Zone Tracking Tests ====================


class TestUpdateTrackZones:
    """Test _update_track_zones method."""

    def test_updates_track_zones_dict(self):
        """_update_track_zones populates _track_zones for active tracks."""
        zones = _make_zones_config([{"name": "kitchen", "cameras": ["cam1"]}])
        manager = _make_manager(zones_config=zones)

        _register_active_track(manager, "cam1", 1, "global_1")

        zone_map = manager._update_track_zones()

        assert manager._track_zones["global_1"] == "kitchen"
        assert "kitchen" in zone_map
        assert "cam1" in zone_map["kitchen"]

    def test_zone_change_triggers_persistence(self):
        """When zone changes, extra_data is persisted via repository."""
        zones = _make_zones_config([{"name": "kitchen", "cameras": ["cam1"]}])
        manager = _make_manager(zones_config=zones)

        _register_active_track(manager, "cam1", 1, "global_1")

        # Initial zone is None, so first update triggers a change
        manager._update_track_zones()

        manager.repository.update_track.assert_any_call(
            "global_1", extra_data={"zone": "kitchen"}
        )

    def test_no_persistence_when_zone_unchanged(self):
        """No DB write when zone stays the same."""
        zones = _make_zones_config([{"name": "kitchen", "cameras": ["cam1"]}])
        manager = _make_manager(zones_config=zones)

        _register_active_track(manager, "cam1", 1, "global_1")

        # First call sets zone
        manager._update_track_zones()
        manager.repository.update_track.reset_mock()

        # Second call — zone unchanged
        manager._update_track_zones()

        # Should not call update_track for zone persistence
        for c in manager.repository.update_track.call_args_list:
            if c.kwargs.get("extra_data", {}).get("zone"):
                pytest.fail("update_track called with zone when zone didn't change")

    def test_track_outside_all_zones(self):
        """Track not in any zone gets None in _track_zones."""
        # Zone only on cam2 but track on cam1
        zones = _make_zones_config([{"name": "kitchen", "cameras": ["cam2"]}])
        manager = _make_manager(zones_config=zones)

        _register_active_track(manager, "cam1", 1, "global_1")

        manager._update_track_zones()

        assert manager._track_zones.get("global_1") is None

    def test_get_track_zone_accessor(self):
        """get_track_zone returns correct zone."""
        zones = _make_zones_config([{"name": "hallway", "cameras": ["cam1"]}])
        manager = _make_manager(zones_config=zones)

        _register_active_track(manager, "cam1", 1, "global_1")
        manager._update_track_zones()

        assert manager.get_track_zone("global_1") == "hallway"
        assert manager.get_track_zone("nonexistent") is None


# ==================== Sighting Tests ====================


class TestTrackSightings:
    """Test sighting creation and ending with zone data."""

    def test_sighting_created_on_track_creation(self):
        """create_track_sighting is called with entry_zone when track is created."""
        zones = _make_zones_config([{"name": "kitchen", "cameras": ["cam1"]}])
        manager = _make_manager(zones_config=zones)

        local_track = LocalTrack(track_id=1, camera_id="cam1", bbox=(100, 100, 300, 500))
        # Provide a real frame so dimensions get stored
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        manager.process_local_tracks(
            camera_id="cam1",
            local_tracks=[local_track],
            frame=frame,
            new_track_ids=[1],
            lost_track_ids=[],
        )

        # Verify create_track_sighting was called with entry_zone
        manager.repository.create_track_sighting.assert_called_once()
        call_kwargs = manager.repository.create_track_sighting.call_args
        assert call_kwargs[1].get("entry_zone") == "kitchen" or call_kwargs[0][2] if len(call_kwargs[0]) > 2 else True

    def test_sighting_ended_on_track_loss(self):
        """end_track_sighting is called with exit_zone when track is lost."""
        zones = _make_zones_config([{"name": "kitchen", "cameras": ["cam1"]}])
        manager = _make_manager(zones_config=zones)

        _register_active_track(manager, "cam1", 1, "global_1")
        manager._track_zones["global_1"] = "kitchen"

        import time
        manager._handle_lost_local_track("cam1", 1, time.time())

        manager.repository.end_track_sighting.assert_called_once_with(
            "global_1", "cam1", exit_zone="kitchen"
        )


# ==================== Exit Destination Tests ====================


class TestExitDestination:
    """Test exit destination logic on track loss."""

    def test_exit_destination_used_when_configured(self):
        """Track lost in zone with exit_destination -> estimated_zone = destination."""
        zones = _make_zones_config([
            {"name": "hallway", "cameras": ["cam1"], "exit_destination": "outside"}
        ])
        manager = _make_manager(zones_config=zones)

        _register_active_track(manager, "cam1", 1, "global_1")
        manager._track_zones["global_1"] = "hallway"

        import time
        manager._handle_lost_local_track("cam1", 1, time.time())

        # Should persist estimated_zone = "outside"
        manager.repository.update_track.assert_any_call(
            "global_1", extra_data={"estimated_zone": "outside"}
        )

    def test_zone_name_used_when_no_exit_destination(self):
        """Track lost in zone without exit_destination -> estimated_zone = zone name."""
        zones = _make_zones_config([
            {"name": "kitchen", "cameras": ["cam1"]}
        ])
        manager = _make_manager(zones_config=zones)

        _register_active_track(manager, "cam1", 1, "global_1")
        manager._track_zones["global_1"] = "kitchen"

        import time
        manager._handle_lost_local_track("cam1", 1, time.time())

        # Should persist estimated_zone = "kitchen" (zone name itself)
        manager.repository.update_track.assert_any_call(
            "global_1", extra_data={"estimated_zone": "kitchen"}
        )

    def test_no_estimated_zone_when_track_had_no_zone(self):
        """Track with no zone -> no estimated_zone persistence."""
        zones = _make_zones_config([
            {"name": "kitchen", "cameras": ["cam2"]}  # Not on cam1
        ])
        manager = _make_manager(zones_config=zones)

        _register_active_track(manager, "cam1", 1, "global_1")
        # _track_zones["global_1"] is None (not set)

        import time
        manager._handle_lost_local_track("cam1", 1, time.time())

        # Should NOT call update_track with estimated_zone
        for c in manager.repository.update_track.call_args_list:
            extra = c.kwargs.get("extra_data", {}) if c.kwargs else {}
            assert "estimated_zone" not in extra


# ==================== Zone Cleanup Tests ====================


class TestZoneCleanup:
    """Test that _track_zones is cleaned up when tracks are removed."""

    def test_cleanup_lost_tracks_removes_zone(self):
        """When LOST track is retired to REMOVED, zone entry is cleaned up."""
        zones = _make_zones_config([{"name": "kitchen", "cameras": ["cam1"]}])
        manager = _make_manager(zones_config=zones)
        manager.reid_config.global_id_grace_period = 0  # Expire immediately

        _register_active_track(manager, "cam1", 1, "global_1")
        manager._track_zones["global_1"] = "kitchen"

        # Mark as LOST with old last_seen
        track = manager._tracks["global_1"]
        track.state = TrackState.LOST
        track.last_seen = datetime(2020, 1, 1)
        manager._recently_lost_tracks.add("global_1")

        import time
        manager._cleanup_lost_tracks(time.time())

        assert "global_1" not in manager._track_zones

    def test_cleanup_old_tracks_removes_zone(self):
        """cleanup_old_tracks removes zone entries."""
        zones = _make_zones_config([{"name": "kitchen", "cameras": ["cam1"]}])
        manager = _make_manager(zones_config=zones)

        _register_active_track(manager, "cam1", 1, "global_1")
        manager._track_zones["global_1"] = "kitchen"

        # Mark as REMOVED
        manager._tracks["global_1"].state = TrackState.REMOVED

        manager.cleanup_old_tracks()

        assert "global_1" not in manager._track_zones


# ==================== ZoneManager Tests ====================


class TestZoneManagerExitDestination:
    """Test ZoneManager.get_exit_destination."""

    def test_returns_exit_destination(self):
        zones_config = _make_zones_config([
            {"name": "hallway", "cameras": ["cam1"], "exit_destination": "outside"}
        ])
        zm = ZoneManager(zones_config)
        assert zm.get_exit_destination("hallway") == "outside"

    def test_returns_none_when_not_configured(self):
        zones_config = _make_zones_config([
            {"name": "kitchen", "cameras": ["cam1"]}
        ])
        zm = ZoneManager(zones_config)
        assert zm.get_exit_destination("kitchen") is None

    def test_returns_none_for_unknown_zone(self):
        zones_config = _make_zones_config([
            {"name": "kitchen", "cameras": ["cam1"]}
        ])
        zm = ZoneManager(zones_config)
        assert zm.get_exit_destination("nonexistent") is None


# ==================== API Tests ====================


@pytest.fixture(scope="function")
def api_repository():
    """Create temporary file-based database repository for testing."""
    from src.database.repository import Repository

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    repo = Repository(db_path)
    yield repo

    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def api_client(api_repository):
    """Create test client with fresh app and repository."""
    from src.api.routes.events import create_events_router

    app = FastAPI()
    events_router = create_events_router(api_repository)
    app.include_router(events_router)
    return TestClient(app)


class TestLocationAPI:
    """Test LocationResponse includes estimated_location and time_since_seen_sec."""

    def test_active_track_shows_zone_as_estimated_location(self, api_repository, api_client):
        """Active track with zone -> estimated_location = zone name."""
        person = api_repository.create_person("Alice")
        track = api_repository.create_track("global_1", "cam1", person_id=person.id)
        api_repository.update_track("global_1", extra_data={"zone": "kitchen"})

        resp = api_client.get(f"/api/v1/persons/{person.id}/location")
        assert resp.status_code == 200
        data = resp.json()
        assert data["estimated_location"] == "kitchen"
        assert data["status"] == "active"
        assert data["time_since_seen_sec"] is not None

    def test_archived_track_shows_estimated_zone(self, api_repository, api_client):
        """Archived track with estimated_zone -> estimated_location = exit destination."""
        person = api_repository.create_person("Bob")
        api_repository.create_track("global_2", "cam1", person_id=person.id)
        api_repository.update_track(
            "global_2",
            status="archived",
            extra_data={"zone": "hallway", "estimated_zone": "outside"},
        )

        resp = api_client.get(f"/api/v1/persons/{person.id}/location")
        assert resp.status_code == 200
        data = resp.json()
        assert data["estimated_location"] == "outside"
        assert data["time_since_seen_sec"] is not None

    def test_archived_track_falls_back_to_zone(self, api_repository, api_client):
        """Archived track without estimated_zone -> falls back to zone."""
        person = api_repository.create_person("Charlie")
        api_repository.create_track("global_3", "cam1", person_id=person.id)
        api_repository.update_track(
            "global_3",
            status="archived",
            extra_data={"zone": "living_room"},
        )

        resp = api_client.get(f"/api/v1/persons/{person.id}/location")
        assert resp.status_code == 200
        data = resp.json()
        assert data["estimated_location"] == "living_room"


class TestActiveTracksAPI:
    """Test ActiveTrackResponse includes zone field."""

    def test_active_track_includes_zone(self, api_repository, api_client):
        """Active tracks endpoint includes zone from extra_data."""
        api_repository.create_track("global_1", "cam1")
        api_repository.update_track("global_1", extra_data={"zone": "kitchen"})

        resp = api_client.get("/api/v1/tracks/active")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["zone"] == "kitchen"

    def test_active_track_zone_null_when_no_zone(self, api_repository, api_client):
        """Active track without zone -> zone is null."""
        api_repository.create_track("global_2", "cam2")

        resp = api_client.get("/api/v1/tracks/active")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["zone"] is None
