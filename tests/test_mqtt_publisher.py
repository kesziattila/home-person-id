"""Tests for MQTT publisher."""

import json
import unittest
from unittest.mock import MagicMock, patch

from src.config import MQTTConfig
from src.events.mqtt_publisher import MQTTPublisher, sanitize_topic_name


class TestSanitizeTopicName(unittest.TestCase):
    def test_simple_name(self):
        self.assertEqual(sanitize_topic_name("Alice"), "alice")

    def test_name_with_spaces(self):
        self.assertEqual(sanitize_topic_name("John Doe"), "john_doe")

    def test_name_with_special_chars(self):
        self.assertEqual(sanitize_topic_name("José María"), "jos_mar_a")

    def test_name_with_multiple_spaces(self):
        self.assertEqual(sanitize_topic_name("  Bob   Smith  "), "bob_smith")

    def test_name_with_mixed_case(self):
        self.assertEqual(sanitize_topic_name("Alice-Bob"), "alice_bob")


class TestMQTTPublisherDisabled(unittest.TestCase):
    def test_disabled_config_does_not_connect(self):
        config = MQTTConfig(enabled=False)
        publisher = MQTTPublisher(config)
        self.assertIsNone(publisher._client)

    def test_publish_when_disabled_is_noop(self):
        config = MQTTConfig(enabled=False)
        publisher = MQTTPublisher(config)
        # Should not raise
        publisher.publish_person_location(
            person_name="Alice", person_id=1, zone="kitchen",
            is_estimated=False, prev_zone=None, camera_id="cam1", track_id="global_1",
        )


class TestMQTTPublisherPublish(unittest.TestCase):
    @patch("src.events.mqtt_publisher.MQTTPublisher.__init__", return_value=None)
    def setUp(self, mock_init):
        self.publisher = MQTTPublisher.__new__(MQTTPublisher)
        self.publisher._config = MQTTConfig(enabled=True, topic_prefix="home/person")
        self.publisher._client = MagicMock()
        self.publisher._connected = True

    def test_publish_topic_format(self):
        self.publisher.publish_person_location(
            person_name="Alice", person_id=1, zone="kitchen",
            is_estimated=False, prev_zone="living_room",
            camera_id="cam1", track_id="global_5",
        )

        self.publisher._client.publish.assert_called_once()
        args, kwargs = self.publisher._client.publish.call_args
        self.assertEqual(args[0], "home/person/alice/location")
        self.assertEqual(kwargs["qos"], 1)
        self.assertTrue(kwargs["retain"])

    def test_publish_payload_contains_all_fields(self):
        self.publisher.publish_person_location(
            person_name="John Doe", person_id=42, zone="bedroom",
            is_estimated=True, prev_zone="hallway",
            camera_id="cam2", track_id="global_10",
        )

        args, _ = self.publisher._client.publish.call_args
        payload = json.loads(args[1])

        self.assertEqual(payload["person_name"], "John Doe")
        self.assertEqual(payload["person_id"], 42)
        self.assertEqual(payload["zone"], "bedroom")
        self.assertTrue(payload["is_estimated"])
        self.assertEqual(payload["previous_zone"], "hallway")
        self.assertEqual(payload["camera_id"], "cam2")
        self.assertEqual(payload["track_id"], "global_10")
        self.assertIn("timestamp", payload)

    def test_publish_with_none_zone(self):
        self.publisher.publish_person_location(
            person_name="Alice", person_id=1, zone=None,
            is_estimated=True, prev_zone="kitchen",
            camera_id="cam1", track_id=None,
        )

        args, _ = self.publisher._client.publish.call_args
        payload = json.loads(args[1])
        self.assertIsNone(payload["zone"])
        self.assertIsNone(payload["track_id"])

    def test_no_publish_when_client_is_none(self):
        self.publisher._client = None
        # Should not raise
        self.publisher.publish_person_location(
            person_name="Alice", person_id=1, zone="kitchen",
            is_estimated=False, prev_zone=None, camera_id="cam1", track_id="global_1",
        )

    def test_topic_with_special_name(self):
        self.publisher.publish_person_location(
            person_name="José María", person_id=3, zone="kitchen",
            is_estimated=False, prev_zone=None, camera_id="cam1", track_id="global_1",
        )

        args, _ = self.publisher._client.publish.call_args
        self.assertEqual(args[0], "home/person/jos_mar_a/location")


class TestMQTTIntegrationWithGlobalTracker(unittest.TestCase):
    """Test that GlobalTrackManager publishes MQTT on zone changes."""

    def test_mqtt_published_on_zone_change(self):
        """When _update_person_zone detects a change, MQTT is published."""
        from src.config import (
            CameraTopologyConfig, Config, FaceRecognitionConfig, ReIDConfig,
        )
        from src.database.repository import Repository
        from src.recognition.identity_linker import IdentityLinker
        from src.tracking.global_tracker import GlobalTrackManager

        repository = Repository(":memory:")
        person = repository.create_person("Alice")

        config = Config()
        config.reid = ReIDConfig(enabled=True)
        config.face_recognition = FaceRecognitionConfig(enabled=False)

        mock_reid = MagicMock()
        mock_reid.extract.return_value = (None, 0.0)

        identity_linker = IdentityLinker(
            face_config=config.face_recognition,
            reid_config=config.reid,
            repository=repository,
            face_recognizer=MagicMock(),
            reid_extractor=mock_reid,
        )

        mock_mqtt = MagicMock()

        gtm = GlobalTrackManager(
            topology_config=CameraTopologyConfig(overlaps=[]),
            reid_config=config.reid,
            identity_linker=identity_linker,
            repository=repository,
            mqtt_publisher=mock_mqtt,
        )

        # Call _update_person_zone directly
        gtm._update_person_zone(
            person.id, "kitchen", is_estimated=False,
            camera_id="cam1", track_id="global_1",
        )

        mock_mqtt.publish_person_location.assert_called_once_with(
            person_name="Alice",
            person_id=person.id,
            zone="kitchen",
            is_estimated=False,
            prev_zone=None,
            camera_id="cam1",
            track_id="global_1",
        )

    def test_mqtt_not_published_when_no_change(self):
        """No MQTT publish when zone didn't actually change."""
        from src.config import (
            CameraTopologyConfig, Config, FaceRecognitionConfig, ReIDConfig,
        )
        from src.database.repository import Repository
        from src.recognition.identity_linker import IdentityLinker
        from src.tracking.global_tracker import GlobalTrackManager

        repository = Repository(":memory:")
        person = repository.create_person("Bob")

        config = Config()
        config.reid = ReIDConfig(enabled=True)
        config.face_recognition = FaceRecognitionConfig(enabled=False)

        identity_linker = IdentityLinker(
            face_config=config.face_recognition,
            reid_config=config.reid,
            repository=repository,
            face_recognizer=MagicMock(),
            reid_extractor=MagicMock(),
        )

        mock_mqtt = MagicMock()

        gtm = GlobalTrackManager(
            topology_config=CameraTopologyConfig(overlaps=[]),
            reid_config=config.reid,
            identity_linker=identity_linker,
            repository=repository,
            mqtt_publisher=mock_mqtt,
        )

        # Set initial zone
        gtm._update_person_zone(person.id, "kitchen", is_estimated=False, camera_id="cam1")
        mock_mqtt.publish_person_location.assert_called_once()
        mock_mqtt.reset_mock()

        # Same zone again — should NOT publish
        gtm._update_person_zone(person.id, "kitchen", is_estimated=False, camera_id="cam1")
        mock_mqtt.publish_person_location.assert_not_called()


if __name__ == "__main__":
    unittest.main()
