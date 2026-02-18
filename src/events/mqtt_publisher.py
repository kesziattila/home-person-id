"""MQTT publisher for Home Assistant integration."""

import json
import logging
import re
import threading
from datetime import datetime
from typing import Optional

from src.config import MQTTConfig

logger = logging.getLogger(__name__)


def sanitize_topic_name(name: str) -> str:
    """Convert a person name to a lowercase MQTT-safe topic segment.

    Replaces spaces and special characters with underscores,
    strips leading/trailing underscores, and lowercases.
    """
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


class MQTTPublisher:
    """Publishes person events to an MQTT broker.

    Thread-safe: uses paho-mqtt's loop_start() for async publishing.
    """

    def __init__(self, config: MQTTConfig):
        self._config = config
        self._client = None
        self._connected = False

        if not config.enabled:
            logger.info("MQTT publishing disabled")
            return

        try:
            import paho.mqtt.client as mqtt

            self._client = mqtt.Client()
            if config.username:
                self._client.username_pw_set(config.username, config.password)

            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect

            self._client.connect_async(config.broker, config.port)
            self._client.loop_start()
            logger.info(f"MQTT publisher connecting to {config.broker}:{config.port}")
        except ImportError:
            logger.warning("paho-mqtt not installed, MQTT publishing disabled")
            self._client = None
        except Exception as e:
            logger.error(f"Failed to initialize MQTT: {e}")
            self._client = None

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            logger.info("MQTT connected")
        else:
            logger.warning(f"MQTT connection failed with code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        if rc != 0:
            logger.warning(f"MQTT disconnected unexpectedly (rc={rc})")

    def publish_person_location(
        self,
        person_name: str,
        person_id: int,
        zone: Optional[str],
        is_estimated: bool,
        prev_zone: Optional[str],
        camera_id: str,
        track_id: Optional[str],
    ) -> None:
        """Publish a person location change to MQTT.

        Topic: {topic_prefix}/{sanitized_name}_location
        Payload: JSON with all location data + original person name.
        """
        if not self._client:
            return

        topic_name = sanitize_topic_name(person_name)
        topic = f"{self._config.topic_prefix}/{topic_name}/location"

        payload = {
            "person_name": person_name,
            "person_id": person_id,
            "zone": zone,
            "previous_zone": prev_zone,
            "is_estimated": is_estimated,
            "camera_id": camera_id,
            "track_id": track_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        try:
            self._client.publish(topic, json.dumps(payload), qos=1, retain=True)
            logger.debug(f"MQTT published to {topic}: zone={zone} estimated={is_estimated}")
        except Exception as e:
            logger.warning(f"MQTT publish failed: {e}")

    def stop(self):
        """Disconnect and stop the MQTT client loop."""
        if self._client:
            try:
                self._client.loop_stop()
                self._client.disconnect()
                logger.info("MQTT publisher stopped")
            except Exception as e:
                logger.debug(f"MQTT stop error: {e}")
