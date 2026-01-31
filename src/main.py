"""Main entry point for Home Person ID system."""

import argparse
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any, Dict, List

import numpy as np

from src.config import load_config
from src.database.repository import Repository
from src.detection.motion_detector import MotionDetectorManager
from src.detection.person_detector import PersonDetector, DetectionResult
from src.recognition.identity_linker import IdentityLinker
from src.recognition.unidentified_face_manager import UnidentifiedFaceManager
from src.stream.manager import StreamManager
from src.tracking.byte_tracker import ByteTrackerManager
from src.tracking.global_tracker import GlobalTrackManager, GlobalTrackingResult
from src.tracking.stationary_tracker import StationaryTracker
from src.tracking.zone_manager import ZoneManager
from src.tracking.handover import HandoverManager
from src.visualization.preview import PreviewBuffer
from src.api.server import APIServer
from src.utils.profiler import profiler, memory_profiler, create_data_structure_tracker

from queue import Queue, Empty
from threading import Thread

logger = logging.getLogger(__name__)


@dataclass
class InferenceTask:
    """Task for inference worker."""

    camera_id: str
    frame: np.ndarray
    motion_result: Any
    has_active_tracks: bool
    timestamp: float


@dataclass
class InferenceResult:
    """Result from inference worker."""

    camera_id: str
    frame: np.ndarray
    motion_result: Any
    detections: Any
    timestamp: float
    reid_embeddings: Dict[int, tuple[np.ndarray, float]] = field(default_factory=dict)


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

    def __init__(self, config_path: str, debug: bool = False, mem_profile: bool = False):
        """Initialize the system.

        Args:
            config_path: Path to configuration file
            debug: Enable debug output to stdout
            mem_profile: Enable memory profiling
        """
        self.config = load_config(config_path)
        self._debug = debug
        self._mem_profile = mem_profile

        # Start memory profiling early to capture initialization
        if self._mem_profile:
            memory_profiler.enable_tracemalloc()
            memory_profiler.take_snapshot("before_init")

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
        self.person_detector = None
        if self.config.detection.enabled:
            self.person_detector = PersonDetector(
                model_path=self.config.detection.model,
                confidence_threshold=self.config.detection.confidence_threshold,
                nms_iou_threshold=self.config.detection.nms_iou_threshold,
            )

        # Initialize tracking
        self.tracker_manager = ByteTrackerManager(self.config.tracking)
        self.stationary_tracker = StationaryTracker(self.config.tracking.stationary_timeout)
        self.zone_manager = ZoneManager(self.config.zones)
        self.handover_manager = HandoverManager(self.config.camera_topology)

        # Initialize unidentified face manager (optional)
        self.unidentified_face_manager: Optional[UnidentifiedFaceManager] = None
        if self.config.unidentified_faces.enabled:
            self.unidentified_face_manager = UnidentifiedFaceManager(
                config=self.config.unidentified_faces,
                repository=self.repository,
            )

        # Initialize recognition
        self.identity_linker = IdentityLinker(
            self.config.face_recognition,
            self.config.reid,
            self.repository,
            unidentified_face_manager=self.unidentified_face_manager,
        )

        # Initialize global tracking
        self.global_tracker = GlobalTrackManager(
            self.config.camera_topology,
            self.config.reid,
            self.identity_linker,
            self.repository,
            zones_config=self.config.zones,
        )

        # Parallel processing queues
        self._inference_queue = Queue(maxsize=len(self.config.cameras) * 2)
        self._result_queue = Queue(maxsize=len(self.config.cameras) * 2)
        self._inference_thread = None

        # State
        self._running = False
        self._frame_count = 0
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes

        # Initialize Web UI / API
        self.preview_buffer = PreviewBuffer()
        self.api_server = APIServer(
            self.preview_buffer,
            repository=self.repository,
            face_config=self.config.face_recognition,
            unidentified_faces_config=self.config.unidentified_faces,
            host=self.config.api.host,
            port=self.config.api.port,
            use_nvjpeg=self.config.detection.use_nvjpeg,
            zone_manager=self.zone_manager
        )

    def _setup_logging(self):
        """Setup logging based on config."""
        # Use DEBUG level if --debug flag is set, otherwise use config value
        if self._debug:
            log_level = logging.DEBUG
        else:
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

        if self._debug:
            print("=" * 60)
            print("[DEBUG MODE ENABLED]")
            print("=" * 60)
            print(f"Cameras: {[c.id for c in self.config.cameras]}")
            print(f"Motion detection: {'enabled' if self.config.motion.enabled else 'disabled'}")
            print(f"Person detection: {'enabled' if self.config.detection.enabled else 'disabled'}")
            print(f"Face recognition: {'enabled' if self.config.face_recognition.enabled else 'disabled'}")
            print(f"Re-ID: {'enabled' if self.config.reid.enabled else 'disabled'}")
            print(f"Database: {self.config.database.path}")
            print("=" * 60)

        # Warmup models
        logger.info("Warming up models...")
        if self.person_detector:
            self.person_detector.warmup()
        self.identity_linker.warmup()

        # Initialize memory profiling if enabled
        if self._mem_profile:
            logger.info("Memory profiling enabled")

            # Take post-init snapshot and print initialization memory footprint
            memory_profiler.take_snapshot("after_init")
            print(memory_profiler.get_leak_report("before_init", "after_init"))

            # Register data structure tracking
            tracker_callback = create_data_structure_tracker(
                global_tracker=self.global_tracker,
                identity_linker=self.identity_linker,
                id_manager=self.identity_linker._id_manager,
            )
            memory_profiler.register_data_structure(tracker_callback)

            # Set baseline for ongoing leak detection
            memory_profiler._baseline_name = "after_init"
            memory_profiler._tracking_enabled = True
            memory_profiler._last_check_time = time.time()

        # Start camera streams
        self.stream_manager.start()

        # Start unidentified face manager
        if self.unidentified_face_manager:
            self.unidentified_face_manager.start()

        # Start API server
        self.api_server.start()

        # Start inference worker thread
        self._inference_thread = Thread(target=self._inference_worker, name="InferenceWorker", daemon=True)
        self._inference_thread.start()

        self._running = True
        logger.info("System started successfully")

        if self._debug:
            print("[DEBUG] System started. Waiting for frames...")

    def stop(self):
        """Stop the system."""
        logger.info("Stopping Person ID System")
        self._running = False
        self.api_server.stop()
        if self.unidentified_face_manager:
            self.unidentified_face_manager.stop()
        self.stream_manager.stop()
        logger.info("System stopped")

    def _inference_worker(self):
        """Background thread for running person detection."""
        logger.info("Inference worker started")
        while self._running:
            try:
                task: InferenceTask = self._inference_queue.get(timeout=0.1)

                try:
                    # Person detection (expensive)
                    detections = DetectionResult(detections=[], frame_shape=task.frame.shape)
                    if self.person_detector:
                        with profiler.measure("PersonDetector.detect"):
                            detections = self.person_detector.detect(task.frame)

                    # Extract Re-ID for all detections to avoid doing it in main thread
                    reid_embeddings = {}
                    if detections.detections:
                        crops = []
                        det_indices = []
                        for i, det in enumerate(detections.detections):
                            # Use a more robust crop to ensure Re-ID quality matches what's expected
                            # ReIDExtractor uses the whole crop, but we need to ensure it's not empty
                            x1, y1, x2, y2 = map(int, det.bbox)
                            # Pad a bit if possible to avoid edge artifacts
                            h, w = task.frame.shape[:2]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)

                            crop = task.frame[y1:y2, x1:x2]
                            if crop.size > 0:
                                crops.append(crop)
                                det_indices.append(i)

                        if crops:
                            with profiler.measure("ReIDExtractor.extract_batch"):
                                # Lazy load or use existing extractor
                                reid_extractor = self.identity_linker._id_manager.reid_extractor
                                if reid_extractor:
                                    results = reid_extractor.extract_batch(crops)
                                    for idx, res in zip(det_indices, results):
                                        reid_embeddings[idx] = res

                    # Push to results
                    result = InferenceResult(
                        camera_id=task.camera_id,
                        frame=task.frame,
                        motion_result=task.motion_result,
                        detections=detections,
                        timestamp=task.timestamp,
                        reid_embeddings=reid_embeddings
                    )

                    # If result queue is full, drop oldest
                    if self._result_queue.full():
                        try:
                            self._result_queue.get_nowait()
                        except Empty:
                            pass

                    self._result_queue.put(result)

                except Exception as e:
                    logger.error(f"Error processing inference task: {e}", exc_info=True)
                finally:
                    self._inference_queue.task_done()

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in inference worker: {e}", exc_info=True)
                time.sleep(0.1)

    def run(self):
        """Main processing loop."""
        self.start()

        last_frame_report = time.time()
        last_frame_count = 0

        try:
            while self._running:
                loop_start = time.perf_counter()
                
                # 1. Collect frames from cameras and push to inference queue
                frames = self.stream_manager.get_frames(timeout=0.01)
                for frame in frames:
                    self._frame_count += 1
                    camera_id = frame.camera_id

                    # Motion detection gate
                    with profiler.measure("MotionDetector.detect"):
                        motion_result = self.motion_manager.detect(camera_id, frame.image)

                    # Check if we have active tracks for this camera
                    has_active_tracks = self.global_tracker.has_active_tracks(camera_id)

                    if motion_result.has_motion or has_active_tracks:
                        # Push to inference queue
                        task = InferenceTask(
                            camera_id=camera_id,
                            frame=frame.image,
                            motion_result=motion_result,
                            has_active_tracks=has_active_tracks,
                            timestamp=frame.timestamp
                        )
                        
                        # If queue is full, drop oldest
                        if self._inference_queue.full():
                            try:
                                self._inference_queue.get_nowait()
                            except Empty:
                                pass
                        
                        self._inference_queue.put(task)
                    else:
                        # No motion/tracks, still update preview buffer for live view
                        with profiler.measure("PreviewBuffer.update"):
                            self.preview_buffer.update(
                                camera_id=camera_id,
                                image=frame.image,
                                metadata={
                                    "camera_id": camera_id,
                                    "tracks": [],
                                    "identities": {},
                                    "global_tracks": [
                                        t for t in self.global_tracker.get_active_tracks()
                                        if t.current_camera_id == camera_id
                                    ]
                                }
                            )

                # Process results from inference worker
                try:
                    while True:
                        result = self._result_queue.get_nowait()
                        if self._debug:
                            logger.debug(f"Processing inference result for {result.camera_id}")
                        self._process_inference_result(result)
                        self._result_queue.task_done()
                except Empty:
                    pass

                # 3. Periodic tasks
                self._periodic_cleanup()

                # 4. Stats
                now = time.time()
                loop_time = time.perf_counter() - loop_start
                profiler.end_measure("MainLoop", loop_start)
                
                if now - last_frame_report >= 10.0:
                    frames_processed = self._frame_count - last_frame_count
                    fps = frames_processed / (now - last_frame_report)
                    
                    extra_stats = {
                        "FPS": f"{fps:.1f}",
                        "Total Frames": self._frame_count,
                        "Inference Queue": self._inference_queue.qsize(),
                        "Result Queue": self._result_queue.qsize(),
                        "Main Thread Blocked": "Yes" if loop_time > 0.5 else "No",
                        "Loop Time (ms)": f"{loop_time*1000:.1f}"
                    }
                    
                    profiler.log_stats(extra_info=extra_stats)
                    
                    if frames_processed == 0:
                        logger.warning("No frames received in the last 10s! Check camera connections.")
                    
                    last_frame_report = now
                    last_frame_count = self._frame_count

                # Tiny sleep to prevent high CPU when idle
                time.sleep(0.01)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()

    def _process_inference_result(self, result: InferenceResult):
        """Process results from the inference worker."""
        camera_id = result.camera_id
        frame_image = result.frame
        motion_result = result.motion_result
        detections = result.detections

        # 1. Update tracking
        with profiler.measure("ByteTracker.update"):
            track_result = self.tracker_manager.update(
                camera_id, detections, frame_image, result.reid_embeddings
            )

        # 2. Global tracking (cross-camera)
        with profiler.measure("GlobalTracker.process"):
            global_result = self.global_tracker.process_local_tracks(
                camera_id=camera_id,
                local_tracks=track_result.tracks,
                frame=frame_image,
                new_track_ids=track_result.new_track_ids,
                lost_track_ids=track_result.lost_track_ids,
                has_motion=motion_result.has_motion,
            )

        # 3. Update stationary state
        current_time = result.timestamp
        track_ids = [t.track_id for t in track_result.tracks]
        self.stationary_tracker.update(camera_id, track_ids, motion_result.has_motion, current_time)

        # 4. Log events
        self._log_events(camera_id, global_result)

        # 5. Update preview buffer
        identities = {}
        stationary_times = {}
        with profiler.measure("IdentityLinker.get_identity"):
            for track in track_result.tracks:
                global_track = self.global_tracker.get_global_track_for_local(camera_id, track.track_id)
                if global_track:
                    identity = self.identity_linker.get_identity(global_track.track_id)
                    if identity:
                        identities[track.track_id] = identity
                
                # Get stationary time
                s_time = self.stationary_tracker.get_stationary_time(camera_id, track.track_id, current_time)
                if s_time is not None:
                    stationary_times[track.track_id] = s_time

        with profiler.measure("PreviewBuffer.update"):
            self.preview_buffer.update(
                camera_id=camera_id,
                image=frame_image,
                metadata={
                    "camera_id": camera_id,
                    "tracks": track_result.tracks,
                    "identities": identities,
                    "stationary_times": stationary_times,
                    "global_tracks": global_result.active_tracks
                }
            )

    def _process_frames(self):
        """Deprecated: use run() with decoupled pipeline."""
        pass

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
        # Clean up track states more aggressively (10 minutes instead of 1 hour)
        # to prevent unbounded growth of _track_states dictionary
        self.identity_linker.cleanup_old_tracks(max_age_seconds=600)

        # Cleanup old events
        self.repository.cleanup_old_events(days=self.config.database.event_retention_days)

        # Archive old tracks in database
        self.repository.archive_old_tracks(
            hours=self.config.database.archive_tracks_after_hours
        )

        # Cleanup old unidentified faces
        if self.config.unidentified_faces.enabled:
            self.repository.cleanup_old_unidentified_faces(
                days=self.config.unidentified_faces.retention_days
            )

        # Check for memory leaks if profiling enabled
        if self._mem_profile:
            report = memory_profiler.check_for_leaks(force=True)
            if report:
                print(report)

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
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable debug output (print detections to stdout)",
    )
    parser.add_argument(
        "--perf-report",
        action="store_true",
        help="Print performance report every 10 seconds",
    )
    parser.add_argument(
        "--mem-profile",
        action="store_true",
        help="Enable memory profiling (tracks Python heap and CUDA memory)",
    )
    args = parser.parse_args()

    # Configure profiler
    profiler.enabled = args.perf_report

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
    system = PersonIDSystem(args.config, debug=args.debug, mem_profile=args.mem_profile)
    system.run()


if __name__ == "__main__":
    main()
