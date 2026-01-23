"""Main entry point for Home Person ID system."""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

from src.config import load_config
from src.database.repository import Repository
from src.detection.motion_detector import MotionDetectorManager
from src.detection.person_detector import PersonDetector
from src.recognition.identity_linker import IdentityLinker
from src.stream.manager import StreamManager
from src.tracking.byte_tracker import ByteTrackerManager
from src.tracking.global_tracker import GlobalTrackManager
from src.tracking.handover import HandoverManager

logger = logging.getLogger(__name__)


class PersonIDSystem:
    """Main system class that orchestrates all components.

    Pipeline:
    1. Stream Manager reads frames from RTSP cameras
    2. Motion Detector gates processing (skip if no motion)
    3. Person Detector (YOLO) detects persons
    4. ByteTracker tracks persons within each camera
    5. Global Tracker manages cross-camera tracking
    6. Identity Linker handles face recognition + Re-ID
    """

    def __init__(self, config_path: str):
        """Initialize the system.

        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self._setup_logging()

        logger.info("Initializing Person ID System")
        logger.info(f"Cameras: {len(self.config.cameras)}")
        logger.info(f"Face recognition: {'enabled' if self.config.face_recognition.enabled else 'disabled'}")

        # Initialize database
        self.repository = Repository(self.config.database.path)

        # Initialize stream management
        self.stream_manager = StreamManager(self.config)

        # Initialize detection
        self.motion_manager = MotionDetectorManager(self.config.motion)
        self.person_detector = PersonDetector(
            model_path=self.config.detection.model,
            confidence_threshold=self.config.detection.confidence_threshold,
            nms_iou_threshold=self.config.detection.nms_iou_threshold,
        )

        # Initialize tracking
        self.tracker_manager = ByteTrackerManager(self.config.tracking)
        self.handover_manager = HandoverManager(self.config.camera_topology)

        # Initialize recognition
        self.identity_linker = IdentityLinker(
            self.config.face_recognition,
            self.config.reid,
            self.repository,
        )

        # Initialize global tracking
        self.global_tracker = GlobalTrackManager(
            self.config.camera_topology,
            self.config.reid,
            self.identity_linker,
            self.repository,
            zones_config=self.config.zones,
        )

        # State
        self._running = False
        self._frame_count = 0
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes

    def _setup_logging(self):
        """Setup logging based on config."""
        log_level = getattr(logging, self.config.logging.level.upper(), logging.INFO)

        handlers = [logging.StreamHandler()]

        if self.config.logging.file:
            log_path = Path(self.config.logging.file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_path))

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=handlers,
            force=True,
        )

    def start(self):
        """Start the system."""
        logger.info("Starting Person ID System")

        # Warmup models
        logger.info("Warming up person detector...")
        self.person_detector.warmup()

        if self.config.face_recognition.enabled:
            logger.info("Warming up face recognizer...")
            # Face recognizer is lazy-loaded in identity_linker

        logger.info("Warming up Re-ID extractor...")
        self.identity_linker.reid_extractor.warmup()

        # Start camera streams
        self.stream_manager.start()

        self._running = True
        logger.info("System started successfully")

    def stop(self):
        """Stop the system."""
        logger.info("Stopping Person ID System")
        self._running = False
        self.stream_manager.stop()
        logger.info("System stopped")

    def run(self):
        """Main processing loop."""
        self.start()

        try:
            while self._running:
                self._process_frames()
                self._periodic_cleanup()
                time.sleep(0.001)  # Small sleep to prevent busy loop

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()

    def _process_frames(self):
        """Process frames from all cameras."""
        for frame in self.stream_manager.get_frames(timeout=0.1):
            self._frame_count += 1
            camera_id = frame.camera_id

            # Motion detection gate
            motion_result = self.motion_manager.detect(camera_id, frame.image)

            # Check if we have active tracks for this camera
            has_active_tracks = self.global_tracker.has_active_tracks(camera_id)

            # Continue processing if motion detected OR we have active tracks
            if not motion_result.has_motion and not has_active_tracks:
                continue

            # Person detection
            detections = self.person_detector.detect(frame.image)

            # If no detections but have active tracks, update tracker anyway
            # to maintain track state (will mark as lost after track_buffer frames)
            if detections.count == 0 and not has_active_tracks:
                continue

            # Single-camera tracking
            track_result = self.tracker_manager.update(
                camera_id, detections, frame.image
            )

            # Global tracking (cross-camera)
            global_result = self.global_tracker.process_local_tracks(
                camera_id=camera_id,
                local_tracks=track_result.tracks,
                frame=frame.image,
                new_track_ids=track_result.new_track_ids,
                lost_track_ids=track_result.lost_track_ids,
                has_motion=motion_result.has_motion,
            )

            # Log events
            self._log_events(camera_id, global_result)

    def _log_events(self, camera_id: str, result):
        """Log tracking events."""
        # New tracks
        for track_id in result.new_global_tracks:
            logger.info(f"[{camera_id}] New person detected: {track_id}")

        # Handovers
        for track_id, from_cam, to_cam in result.handovers_completed:
            state = self.identity_linker.get_track_state(track_id)
            person_name = "Unknown"
            if state and state.is_identified:
                person = self.repository.get_person(state.person_id)
                if person:
                    person_name = person.name

            logger.info(
                f"[{to_cam}] Person moved from {from_cam}: {person_name} ({track_id})"
            )

            # Create event
            self.repository.create_event(
                camera_id=to_cam,
                event_type="track_handover",
                track_id=track_id,
                extra_data={"from_camera": from_cam},
            )

        # Lost tracks
        for track_id in result.tracks_lost:
            state = self.identity_linker.get_track_state(track_id)
            if state and state.is_identified:
                person = self.repository.get_person(state.person_id)
                if person:
                    logger.info(f"[{camera_id}] Person left: {person.name}")

    def _periodic_cleanup(self):
        """Run periodic cleanup tasks."""
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = current_time

        # Cleanup old tracks
        self.global_tracker.cleanup_old_tracks(
            max_age_hours=self.config.database.archive_tracks_after_hours
        )
        self.identity_linker.cleanup_old_tracks(max_age_seconds=3600)

        # Cleanup old events
        self.repository.cleanup_old_events(days=self.config.database.event_retention_days)

        # Archive old tracks in database
        self.repository.archive_old_tracks(
            hours=self.config.database.archive_tracks_after_hours
        )

        # Log status
        occupancy = self.global_tracker.get_occupancy()
        logger.info(
            f"Status: {occupancy['total_tracks']} tracks, "
            f"identified: {occupancy['persons']}, unknown: {occupancy['unknown_count']}"
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Home Person ID System")
    parser.add_argument(
        "--config",
        "-c",
        default="config/config.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    # Handle signals
    system = None

    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        if system:
            system.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run system
    system = PersonIDSystem(args.config)
    system.run()


if __name__ == "__main__":
    main()
