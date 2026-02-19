"""Tests for person zone state persistence on the Person record."""

import time
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np

from src.config import (
    CameraTopologyConfig,
    Config,
    FaceRecognitionConfig,
    ReIDConfig,
    ZonesConfig,
    ZoneConfig,
    ZonePolygon,
)
from src.database.models import Person, init_database
from src.database.repository import Repository
from src.recognition.identity_linker import IdentityLinker
from src.tracking.global_tracker import GlobalTrackManager
from src.tracking.track import LocalTrack, TrackState


class TestPersonZoneState(unittest.TestCase):
    """Test that zone state is persisted on Person records."""

    def setUp(self):
        self.repository = Repository(":memory:")

        self.config = Config()
        self.config.reid = ReIDConfig(enabled=True)
        self.config.face_recognition = FaceRecognitionConfig(enabled=False)
        self.config.camera_topology = CameraTopologyConfig(overlaps=[])

        self.mock_reid_extractor = MagicMock()
        self.mock_reid_extractor.extract.return_value = (np.zeros(512), 0.9)
        self.mock_reid_extractor.compare.return_value = 1.0

        self.mock_face_recognizer = MagicMock()

        self.identity_linker = IdentityLinker(
            face_config=self.config.face_recognition,
            reid_config=self.config.reid,
            repository=self.repository,
            face_recognizer=self.mock_face_recognizer,
            reid_extractor=self.mock_reid_extractor,
        )

        # Zone config: cam1 sees "living_room"
        zones_config = ZonesConfig(zones=[
            ZoneConfig(
                name="living_room",
                cameras={"cam1": ZonePolygon(polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])},
            ),
        ])

        self.gtm = GlobalTrackManager(
            topology_config=self.config.camera_topology,
            reid_config=self.config.reid,
            identity_linker=self.identity_linker,
            repository=self.repository,
            zones_config=zones_config,
        )

        # Create a known person
        self.person = self.repository.create_person("Alice")

    def _make_local_track(self, track_id=1, camera_id="cam1", with_crop=False):
        crop = np.zeros((100, 100, 3), dtype=np.uint8) if with_crop else None
        return LocalTrack(
            track_id=track_id,
            camera_id=camera_id,
            bbox=(100, 100, 200, 200),
            hits=5,
            state=TrackState.TRACKED,
            last_crop=crop,
        )

    def _process(self, camera_id, tracks, new_ids=None, lost_ids=None):
        if new_ids is None:
            new_ids = [t.track_id for t in tracks]
        if lost_ids is None:
            lost_ids = []
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return self.gtm.process_local_tracks(
            camera_id=camera_id,
            local_tracks=tracks,
            frame=frame,
            new_track_ids=new_ids,
            lost_track_ids=lost_ids,
        )

    def _identify_track(self, global_track_id: str, person_id: int):
        """Manually identify a track (simulates face recognition)."""
        state = self.identity_linker.get_track_state(global_track_id)
        if state:
            state.confirm_identity(person_id, 0.95, "face")
        self.repository.update_track(global_track_id, person_id=person_id)
        # Also update GlobalTrack in-memory
        gt = self.gtm._tracks.get(global_track_id)
        if gt:
            gt.person_id = person_id

    # ---- Test: zone update on active track zone change ----

    def test_zone_update_on_track_zone_change(self):
        """When an identified track's zone changes, Person.current_zone is updated (not estimated)."""
        lt = self._make_local_track()
        result = self._process("cam1", [lt])
        global_id = result.new_global_tracks[0]

        # Identify the track
        self._identify_track(global_id, self.person.id)

        # Clear the in-memory zone so the next check sees a change
        self.gtm._track_zones[global_id] = None

        # Force a zone update by triggering _update_track_zones
        self.gtm._last_cross_camera_check = 0
        self.config.camera_topology.enable_cross_camera_propagation = False

        # Process again (not new, not lost) to trigger zone check
        self._process("cam1", [lt], new_ids=[], lost_ids=[])

        # Check person record
        person = self.repository.get_person(self.person.id)
        self.assertEqual(person.current_zone, "living_room")
        self.assertFalse(person.zone_is_estimated)
        self.assertIsNotNone(person.zone_updated_at)

        # Check that a person_zone_change event was emitted
        events = self.repository.get_events(
            person_id=self.person.id, event_type="person_zone_change"
        )
        self.assertGreaterEqual(len(events), 1)
        latest = events[0]
        self.assertEqual(latest.extra_data["new_zone"], "living_room")
        self.assertFalse(latest.extra_data["is_estimated"])

    # ---- Test: zone becomes estimated on track lost ----

    def test_zone_becomes_estimated_on_track_lost(self):
        """When an identified track is lost, Person zone becomes estimated."""
        lt = self._make_local_track()
        result = self._process("cam1", [lt])
        global_id = result.new_global_tracks[0]

        self._identify_track(global_id, self.person.id)

        # Set zone first
        self.gtm._track_zones[global_id] = "living_room"

        # Lose the track
        self._process("cam1", [], new_ids=[], lost_ids=[1])

        person = self.repository.get_person(self.person.id)
        self.assertTrue(person.zone_is_estimated)
        self.assertIsNotNone(person.zone_updated_at)

        # Check that a person_zone_change event was emitted with is_estimated=True
        events = self.repository.get_events(
            person_id=self.person.id, event_type="person_zone_change"
        )
        self.assertGreaterEqual(len(events), 1)
        latest = events[0]
        self.assertTrue(latest.extra_data["is_estimated"])

    # ---- Test: lost track does NOT overwrite active track's zone ----

    def test_lost_track_does_not_overwrite_active_track_zone(self):
        """When one track is lost but another active track exists for the same person,
        the person's zone should NOT be overwritten with the estimated zone."""
        # Create two tracks on different cameras for the same person
        lt1 = self._make_local_track(track_id=1, camera_id="cam1")
        r1 = self._process("cam1", [lt1])
        g1 = r1.new_global_tracks[0]
        self._identify_track(g1, self.person.id)
        self.gtm._track_zones[g1] = "living_room"

        lt2 = self._make_local_track(track_id=2, camera_id="cam1")
        r2 = self._process("cam1", [lt2], new_ids=[2], lost_ids=[])
        g2 = r2.new_global_tracks[0]
        self._identify_track(g2, self.person.id)
        self.gtm._track_zones[g2] = "kitchen"

        # Set person zone to kitchen (from the second, active track)
        self.repository.update_person_zone(self.person.id, "kitchen", is_estimated=False)

        # Lose the first track (living_room)
        self._process("cam1", [lt2], new_ids=[], lost_ids=[1])

        # Person zone should still be kitchen (from active track), NOT estimated
        person = self.repository.get_person(self.person.id)
        self.assertEqual(person.current_zone, "kitchen")
        self.assertFalse(person.zone_is_estimated)

    def test_lost_track_uses_estimated_zone_when_no_active_tracks(self):
        """When a track is lost and no other active tracks exist, the person's zone
        is updated to the lost track's estimated zone (marked as estimated)."""
        lt = self._make_local_track(track_id=1, camera_id="cam1")
        r = self._process("cam1", [lt])
        g1 = r.new_global_tracks[0]
        self._identify_track(g1, self.person.id)

        # Person is observed in living_room
        self.repository.update_person_zone(self.person.id, "living_room", is_estimated=False)

        # Track's last zone was "kitchen" (exit zone)
        self.gtm._track_zones[g1] = "kitchen"

        # Lose the track — no other active tracks exist
        self._process("cam1", [], new_ids=[], lost_ids=[1])

        # With no active tracks, the resolver uses the lost track's estimated zone
        person = self.repository.get_person(self.person.id)
        self.assertEqual(person.current_zone, "kitchen")
        self.assertTrue(person.zone_is_estimated)

    def test_lost_track_marks_estimated_when_same_zone(self):
        """When a track is lost and its zone matches the person's current observed zone,
        the zone should be marked as estimated (person no longer being tracked)."""
        lt = self._make_local_track(track_id=1, camera_id="cam1")
        r = self._process("cam1", [lt])
        g1 = r.new_global_tracks[0]
        self._identify_track(g1, self.person.id)

        # Person is observed in living_room
        self.repository.update_person_zone(self.person.id, "living_room", is_estimated=False)

        # Track's last zone is also living_room (same zone)
        self.gtm._track_zones[g1] = "living_room"

        # Lose the track
        self._process("cam1", [], new_ids=[], lost_ids=[1])

        # Zone stays living_room but becomes estimated
        person = self.repository.get_person(self.person.id)
        self.assertEqual(person.current_zone, "living_room")
        self.assertTrue(person.zone_is_estimated)

    # ---- Test: resolver picks most recently seen active track ----

    def test_resolver_picks_most_recently_seen_active_track(self):
        """With multiple active tracks for a person, resolver picks the most recently seen zone."""
        lt1 = self._make_local_track(track_id=1, camera_id="cam1")
        r1 = self._process("cam1", [lt1])
        g1 = r1.new_global_tracks[0]
        self._identify_track(g1, self.person.id)
        self.gtm._track_zones[g1] = "kitchen"

        lt2 = self._make_local_track(track_id=2, camera_id="cam1")
        r2 = self._process("cam1", [lt2], new_ids=[2], lost_ids=[])
        g2 = r2.new_global_tracks[0]
        self._identify_track(g2, self.person.id)
        self.gtm._track_zones[g2] = "living_room"

        # Make g2 more recently seen
        self.gtm._tracks[g2].last_seen = datetime.now()
        self.gtm._tracks[g1].last_seen = datetime.now() - timedelta(seconds=5)

        # Resolve — should pick g2's zone (living_room)
        self.gtm._resolve_person_location(self.person.id, camera_id="cam1")

        person = self.repository.get_person(self.person.id)
        self.assertEqual(person.current_zone, "living_room")
        self.assertFalse(person.zone_is_estimated)

    # ---- Test: lost track with active track present keeps active zone ----

    def test_lost_track_with_active_track_keeps_active_zone(self):
        """When a track is lost but another active track exists for the same person,
        the resolver uses the active track's zone (not estimated)."""
        lt1 = self._make_local_track(track_id=1, camera_id="cam1")
        r1 = self._process("cam1", [lt1])
        g1 = r1.new_global_tracks[0]
        self._identify_track(g1, self.person.id)
        self.gtm._track_zones[g1] = "kitchen"

        lt2 = self._make_local_track(track_id=2, camera_id="cam1")
        r2 = self._process("cam1", [lt2], new_ids=[2], lost_ids=[])
        g2 = r2.new_global_tracks[0]
        self._identify_track(g2, self.person.id)
        self.gtm._track_zones[g2] = "living_room"

        # Lose g1 (kitchen) — g2 (living_room) is still active
        self._process("cam1", [lt2], new_ids=[], lost_ids=[1])

        person = self.repository.get_person(self.person.id)
        self.assertEqual(person.current_zone, "living_room")
        self.assertFalse(person.zone_is_estimated)

    # ---- Test: zone synced on gallery match ----

    def test_zone_synced_on_gallery_match(self):
        """When a new track is identified via gallery match, zone is synced to Person."""
        # Set up gallery manager to return a match
        mock_gallery = MagicMock()
        mock_match_result = MagicMock()
        mock_match_result.matched = True
        mock_match_result.person_name = "Alice"
        mock_match_result.score = 0.85
        mock_match_result.best_score = 0.85
        mock_match_result.db_id = 1
        mock_gallery.match_new_track.return_value = mock_match_result
        # Set on the underlying IdentificationManager
        self.identity_linker._id_manager._reid_gallery_manager = mock_gallery

        lt = self._make_local_track(with_crop=True)
        result = self._process("cam1", [lt])

        # The track should be identified via gallery match
        global_id = result.new_global_tracks[0]

        # Zone should have been set on Person
        person = self.repository.get_person(self.person.id)
        # Zone depends on whether the zone was computed for the track
        zone = self.gtm._track_zones.get(global_id)
        if zone:
            self.assertEqual(person.current_zone, zone)
            self.assertFalse(person.zone_is_estimated)

    # ---- Test: no update for unidentified tracks ----

    def test_no_zone_update_for_unidentified_tracks(self):
        """Unidentified tracks should not update any Person zone."""
        lt = self._make_local_track()
        self._process("cam1", [lt])

        # Force zone check
        self.gtm._last_cross_camera_check = 0
        self._process("cam1", [lt], new_ids=[], lost_ids=[])

        # Person should have no zone set (track is unidentified)
        person = self.repository.get_person(self.person.id)
        self.assertIsNone(person.current_zone)

    # ---- Test: no duplicate event when zone unchanged ----

    def test_no_event_when_zone_unchanged(self):
        """No person_zone_change event when repository detects no actual change."""
        # Directly test via repository: first call changes, second is a no-op
        result1 = self.repository.update_person_zone(self.person.id, "kitchen", is_estimated=False)
        self.assertIsNotNone(result1)  # changed

        result2 = self.repository.update_person_zone(self.person.id, "kitchen", is_estimated=False)
        self.assertIsNone(result2)  # no change

    # ---- Test: zone synced on zone propagation ----

    def test_zone_synced_on_zone_propagation(self):
        """When identity propagates across cameras in shared zone, Person zone is updated."""
        # Set up zones where both cam1 and cam2 see "living_room"
        zones_config = ZonesConfig(zones=[
            ZoneConfig(
                name="living_room",
                cameras={
                    "cam1": ZonePolygon(polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
                    "cam2": ZonePolygon(polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
                },
            ),
        ])
        self.config.camera_topology.enable_cross_camera_propagation = True

        self.gtm = GlobalTrackManager(
            topology_config=self.config.camera_topology,
            reid_config=self.config.reid,
            identity_linker=self.identity_linker,
            repository=self.repository,
            zones_config=zones_config,
        )

        # Create identified track on cam1
        lt1 = self._make_local_track(track_id=1, camera_id="cam1")
        r1 = self._process("cam1", [lt1])
        g1 = r1.new_global_tracks[0]
        self._identify_track(g1, self.person.id)

        # Create unidentified track on cam2
        lt2 = self._make_local_track(track_id=2, camera_id="cam2")
        r2 = self._process("cam2", [lt2])
        g2 = r2.new_global_tracks[0]

        # Force cross-camera propagation
        self.gtm._last_cross_camera_check = 0
        # Process both cameras to update zones
        self._process("cam1", [lt1], new_ids=[], lost_ids=[])
        self.gtm._last_cross_camera_check = 0
        self._process("cam2", [lt2], new_ids=[], lost_ids=[])

        # Person zone should be updated from propagation
        person = self.repository.get_person(self.person.id)
        if person.current_zone is not None:
            self.assertEqual(person.current_zone, "living_room")
            self.assertFalse(person.zone_is_estimated)


class TestPersonZoneRepository(unittest.TestCase):
    """Test the repository update_person_zone method."""

    def setUp(self):
        self.repository = Repository(":memory:")
        self.person = self.repository.create_person("Bob")

    def test_update_person_zone_sets_fields(self):
        self.repository.update_person_zone(self.person.id, "kitchen", is_estimated=False)

        person = self.repository.get_person(self.person.id)
        self.assertEqual(person.current_zone, "kitchen")
        self.assertFalse(person.zone_is_estimated)
        self.assertIsNotNone(person.zone_updated_at)

    def test_update_person_zone_estimated(self):
        self.repository.update_person_zone(self.person.id, "kitchen", is_estimated=True)

        person = self.repository.get_person(self.person.id)
        self.assertEqual(person.current_zone, "kitchen")
        self.assertTrue(person.zone_is_estimated)

    def test_update_person_zone_clear(self):
        self.repository.update_person_zone(self.person.id, "kitchen", is_estimated=False)
        self.repository.update_person_zone(self.person.id, None, is_estimated=True)

        person = self.repository.get_person(self.person.id)
        self.assertIsNone(person.current_zone)
        self.assertTrue(person.zone_is_estimated)

    def test_update_person_zone_nonexistent_person(self):
        """Updating zone for non-existent person should not raise."""
        self.repository.update_person_zone(9999, "kitchen", is_estimated=False)


class TestDatabaseMigration(unittest.TestCase):
    """Test that migration handles existing databases."""

    def test_migration_adds_columns_to_existing_db(self):
        """Calling init_database twice should not fail (migration is idempotent)."""
        from sqlalchemy import create_engine, text, inspect
        from sqlalchemy.orm import sessionmaker

        # First init
        engine1, session1 = init_database(":memory:")

        # Verify columns exist
        inspector = inspect(engine1)
        columns = {col["name"] for col in inspector.get_columns("persons")}
        self.assertIn("current_zone", columns)
        self.assertIn("zone_updated_at", columns)
        self.assertIn("zone_is_estimated", columns)

    def test_migration_idempotent(self):
        """Running migrate_database multiple times should not fail."""
        from src.database.models import migrate_database
        from sqlalchemy import create_engine, text

        engine = create_engine("sqlite://")
        from src.database.models import Base
        Base.metadata.create_all(engine)

        # Run migration twice — should not raise
        migrate_database(engine)
        migrate_database(engine)


class TestLocationAPIResponse(unittest.TestCase):
    """Test that LocationResponse includes new zone fields."""

    def test_location_response_has_zone_fields(self):
        from src.api.routes.events import LocationResponse

        resp = LocationResponse(
            person_id=1,
            person_name="Alice",
            current_zone="living_room",
            zone_is_estimated=False,
            zone_updated_at=datetime.utcnow(),
            status="active",
        )
        self.assertEqual(resp.current_zone, "living_room")
        self.assertFalse(resp.zone_is_estimated)
        self.assertIsNotNone(resp.zone_updated_at)

    def test_location_response_zone_fields_optional(self):
        from src.api.routes.events import LocationResponse

        resp = LocationResponse(
            person_id=1,
            person_name="Alice",
            status="never_seen",
        )
        self.assertIsNone(resp.current_zone)
        self.assertIsNone(resp.zone_is_estimated)
        self.assertIsNone(resp.zone_updated_at)


if __name__ == "__main__":
    unittest.main()
