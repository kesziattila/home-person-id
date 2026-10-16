"""Unit tests for API events routes."""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from fastapi import FastAPI

from src.database.repository import Repository
from src.api.routes.events import create_events_router


@pytest.fixture(scope="function")
def repository():
    """Create temporary file-based database repository for testing."""
    # Use temp file to avoid SQLite in-memory connection issues with TestClient
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    repo = Repository(db_path)
    yield repo

    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def client(repository):
    """Create test client with fresh app and repository."""
    app = FastAPI()
    events_router = create_events_router(repository)
    app.include_router(events_router)
    return TestClient(app)


@pytest.fixture
def sample_data(repository, client):
    """Populate database with sample data."""
    # Create persons
    person1 = repository.create_person("John Doe")
    person2 = repository.create_person("Jane Smith")

    # Create tracks
    track1 = repository.create_track("global_1", "camera1", person_id=person1.id)
    track2 = repository.create_track("global_2", "camera2", person_id=person2.id)
    track3 = repository.create_track("global_3", "camera1")  # Unidentified

    # Create events
    repository.create_event(
        camera_id="camera1",
        event_type="person_identified",
        track_id=track1.id,
        person_id=person1.id,
        confidence=0.95,
    )
    repository.create_event(
        camera_id="camera2",
        event_type="track_created",
        track_id=track2.id,
    )
    repository.create_event(
        camera_id="camera1",
        event_type="person_identified",
        track_id=track2.id,
        person_id=person2.id,
        confidence=0.88,
    )

    # Create track sightings
    repository.create_track_sighting(
        track_id=track1.id,
        camera_id="camera1",
        entry_zone="zone_a",
    )

    return {
        "person1": person1,
        "person2": person2,
        "track1": track1,
        "track2": track2,
        "track3": track3,
    }


class TestEventsEndpoint:
    """Tests for /api/v1/events endpoint."""

    def test_get_events(self, client, sample_data):
        """Test getting all events."""
        response = client.get("/api/v1/events")
        assert response.status_code == 200
        events = response.json()
        assert len(events) >= 3
        assert all("timestamp" in e for e in events)
        assert all("event_type" in e for e in events)

    def test_get_events_filter_by_camera(self, client, sample_data):
        """Test filtering events by camera."""
        response = client.get("/api/v1/events?camera_id=camera1")
        assert response.status_code == 200
        events = response.json()
        assert all(e["camera_id"] == "camera1" for e in events)

    def test_get_events_filter_by_person(self, client, sample_data):
        """Test filtering events by person."""
        person_id = sample_data["person1"].id
        response = client.get(f"/api/v1/events?person_id={person_id}")
        assert response.status_code == 200
        events = response.json()
        assert all(e["person_id"] == person_id for e in events)

    def test_get_events_filter_by_type(self, client, sample_data):
        """Test filtering events by type."""
        response = client.get("/api/v1/events?event_type=person_identified")
        assert response.status_code == 200
        events = response.json()
        assert all(e["event_type"] == "person_identified" for e in events)

    def test_get_events_limit(self, client, sample_data):
        """Test limiting number of events."""
        response = client.get("/api/v1/events?limit=2")
        assert response.status_code == 200
        events = response.json()
        assert len(events) <= 2


class TestPersonsEndpoint:
    """Tests for /api/v1/persons endpoint."""

    def test_get_persons(self, client, sample_data):
        """Test getting all persons."""
        response = client.get("/api/v1/persons")
        assert response.status_code == 200
        persons = response.json()
        assert len(persons) == 2
        assert all("id" in p for p in persons)
        assert all("name" in p for p in persons)
        assert any(p["name"] == "John Doe" for p in persons)
        assert any(p["name"] == "Jane Smith" for p in persons)

    def test_get_persons_with_face_count(self, client, sample_data):
        """Test that persons have face_count field."""
        response = client.get("/api/v1/persons")
        assert response.status_code == 200
        persons = response.json()
        assert all("face_count" in p for p in persons)

    def test_get_persons_active_only(self, client, sample_data, repository):
        """Test filtering active persons only."""
        # Deactivate one person
        repository.delete_person(sample_data["person1"].id)

        response = client.get("/api/v1/persons?active_only=true")
        assert response.status_code == 200
        persons = response.json()
        assert len(persons) == 1
        assert persons[0]["name"] == "Jane Smith"


class TestPersonLocationEndpoint:
    """Tests for /api/v1/persons/{person_id}/location endpoint."""

    def test_get_person_location_active(self, client, sample_data):
        """Test getting location of active person."""
        person_id = sample_data["person1"].id
        response = client.get(f"/api/v1/persons/{person_id}/location")
        assert response.status_code == 200
        location = response.json()
        assert location["person_id"] == person_id
        assert location["person_name"] == "John Doe"
        assert location["status"] == "active"
        assert location["current_camera_id"] == "camera1"

    def test_get_person_location_not_found(self, client):
        """Test getting location of non-existent person."""
        response = client.get("/api/v1/persons/999/location")
        assert response.status_code == 404


def _add_zone_change(repository, person_id, camera_id, ts, new_zone,
                     prev_zone=None, is_estimated=False, snapshot_path=None):
    """Insert a person_zone_change event with an explicit timestamp.

    create_event() defaults timestamp to utcnow(), so for deterministic
    ordering/durations we insert the Event row directly via a session.
    """
    from src.database.models import Event

    with repository.get_session() as session:
        event = Event(
            timestamp=ts,
            camera_id=camera_id,
            event_type="person_zone_change",
            person_id=person_id,
            snapshot_path=snapshot_path,
            extra_data={
                "prev_zone": prev_zone,
                "new_zone": new_zone,
                "is_estimated": is_estimated,
            },
        )
        session.add(event)
        session.commit()
        session.refresh(event)
        return event.id


class TestPersonTimelineEndpoint:
    """Tests for /api/v1/persons/{person_id}/timeline endpoint."""

    @pytest.fixture
    def person(self, repository):
        return repository.create_person("Walker")

    def test_collapses_consecutive_same_zone(self, client, repository, person):
        """Consecutive same-zone observations collapse into one stay segment."""
        base = datetime.utcnow() - timedelta(minutes=30)
        # living_room observed 4 times, then hallway once
        _add_zone_change(repository, person.id, "cam1", base, "living_room")
        _add_zone_change(repository, person.id, "cam1", base + timedelta(seconds=20), "living_room")
        _add_zone_change(repository, person.id, "cam1", base + timedelta(seconds=40), "living_room")
        _add_zone_change(repository, person.id, "cam1", base + timedelta(minutes=2), "living_room")
        _add_zone_change(repository, person.id, "cam2", base + timedelta(minutes=3), "hallway")

        response = client.get(f"/api/v1/persons/{person.id}/timeline")
        assert response.status_code == 200
        segments = response.json()["segments"]

        # newest first: hallway (current), then living_room
        assert len(segments) == 2
        assert segments[0]["zone"] == "hallway"
        assert segments[0]["is_current"] is True
        assert segments[0]["exited_at"] is None
        assert segments[1]["zone"] == "living_room"
        assert segments[1]["is_current"] is False
        # living_room duration = entered..hallway entry = 3 minutes
        assert segments[1]["duration_sec"] == pytest.approx(180.0)

    def test_exposes_snapshot_event_id(self, client, repository, person):
        """A non-estimated event with a snapshot surfaces its snapshot_event_id."""
        base = datetime.utcnow() - timedelta(minutes=10)
        ev_id = _add_zone_change(
            repository, person.id, "cam1", base, "kitchen",
            snapshot_path="/data/snapshots/x.jpg",
        )
        _add_zone_change(repository, person.id, "cam1", base + timedelta(minutes=1), "bedroom")

        response = client.get(f"/api/v1/persons/{person.id}/timeline")
        segments = response.json()["segments"]
        kitchen = next(s for s in segments if s["zone"] == "kitchen")
        assert kitchen["snapshot_event_id"] == ev_id

    def test_estimated_only_stay_is_marked_estimated(self, client, repository, person):
        """A stay built solely from estimated observations is flagged estimated."""
        base = datetime.utcnow() - timedelta(minutes=10)
        _add_zone_change(repository, person.id, "cam1", base, "outside", is_estimated=True)
        _add_zone_change(repository, person.id, "cam1", base + timedelta(minutes=1), "hallway")

        response = client.get(f"/api/v1/persons/{person.id}/timeline")
        segments = response.json()["segments"]
        outside = next(s for s in segments if s["zone"] == "outside")
        assert outside["is_estimated"] is True

    def test_merges_subthreshold_flicker(self, client, repository, person):
        """A short A->b->A flicker merges into a single A segment (threshold=5s)."""
        base = datetime.utcnow() - timedelta(minutes=10)
        # living_room, brief 2s hallway flicker, back to living_room, then kitchen
        _add_zone_change(repository, person.id, "cam1", base, "living_room")
        _add_zone_change(repository, person.id, "cam1", base + timedelta(minutes=1), "hallway")
        _add_zone_change(repository, person.id, "cam1", base + timedelta(minutes=1, seconds=2), "living_room")
        _add_zone_change(repository, person.id, "cam1", base + timedelta(minutes=5), "kitchen")

        response = client.get(f"/api/v1/persons/{person.id}/timeline")
        segments = response.json()["segments"]
        zones = [s["zone"] for s in segments]
        # flicker hallway is gone; one continuous living_room then kitchen
        assert zones == ["kitchen", "living_room"]
        lr = next(s for s in segments if s["zone"] == "living_room")
        assert lr["duration_sec"] == pytest.approx(300.0)  # base..kitchen entry

    def test_preserves_genuine_transition(self, client, repository, person):
        """A genuine A->B room change is preserved even if B is brief."""
        base = datetime.utcnow() - timedelta(minutes=10)
        _add_zone_change(repository, person.id, "cam1", base, "living_room")
        # 2s hallway then kitchen (different zones each side) -> not a flicker
        _add_zone_change(repository, person.id, "cam1", base + timedelta(minutes=1), "hallway")
        _add_zone_change(repository, person.id, "cam1", base + timedelta(minutes=1, seconds=2), "kitchen")

        response = client.get(f"/api/v1/persons/{person.id}/timeline")
        zones = [s["zone"] for s in response.json()["segments"]]
        assert zones == ["kitchen", "hallway", "living_room"]

    def test_empty_when_no_zone_changes(self, client, repository, person):
        """No zone-change events => empty segment list (not an error)."""
        response = client.get(f"/api/v1/persons/{person.id}/timeline")
        assert response.status_code == 200
        body = response.json()
        assert body["person_name"] == "Walker"
        assert body["segments"] == []

    def test_get_person_timeline_not_found(self, client):
        """Test getting timeline of non-existent person."""
        response = client.get("/api/v1/persons/999/timeline")
        assert response.status_code == 404


class TestActiveTracksEndpoint:
    """Tests for /api/v1/tracks/active endpoint."""

    def test_get_active_tracks(self, client, sample_data):
        """Test getting active tracks."""
        response = client.get("/api/v1/tracks/active")
        assert response.status_code == 200
        tracks = response.json()
        assert len(tracks) >= 3
        assert all("track_id" in t for t in tracks)
        assert all("camera_id" in t for t in tracks)
        assert all("duration_sec" in t for t in tracks)

    def test_get_active_tracks_person_info(self, client, sample_data):
        """Test that tracks include person information."""
        response = client.get("/api/v1/tracks/active")
        assert response.status_code == 200
        tracks = response.json()

        # Find track with person
        track_with_person = next(
            (t for t in tracks if t["person_id"] is not None), None
        )
        assert track_with_person is not None
        assert "person_name" in track_with_person
        assert track_with_person["person_name"] in ["John Doe", "Jane Smith"]

    def test_get_active_tracks_unidentified(self, client, sample_data):
        """Test that unidentified tracks are included."""
        response = client.get("/api/v1/tracks/active")
        assert response.status_code == 200
        tracks = response.json()

        # Find track without person
        unidentified_track = next(
            (t for t in tracks if t["person_id"] is None), None
        )
        assert unidentified_track is not None
        assert unidentified_track["track_id"] == "global_3"
