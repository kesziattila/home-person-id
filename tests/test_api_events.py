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


class TestPersonTimelineEndpoint:
    """Tests for /api/v1/persons/{person_id}/timeline endpoint."""

    def test_get_person_timeline(self, client, sample_data):
        """Test getting person timeline."""
        person_id = sample_data["person1"].id
        response = client.get(f"/api/v1/persons/{person_id}/timeline")
        assert response.status_code == 200
        timeline = response.json()
        assert timeline["person_id"] == person_id
        assert timeline["person_name"] == "John Doe"
        assert "sightings" in timeline
        assert len(timeline["sightings"]) >= 1

    def test_get_person_timeline_sighting_details(self, client, sample_data):
        """Test timeline sighting details."""
        person_id = sample_data["person1"].id
        response = client.get(f"/api/v1/persons/{person_id}/timeline")
        assert response.status_code == 200
        timeline = response.json()

        if timeline["sightings"]:
            sighting = timeline["sightings"][0]
            assert "camera_id" in sighting
            assert "entered_at" in sighting
            assert sighting["camera_id"] == "camera1"

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
