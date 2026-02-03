"""Background manager for capturing unidentified faces for manual review."""

import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Optional

import cv2
import numpy as np

from src.config import UnidentifiedFacesConfig
from src.database.repository import Repository
from src.utils.profiler import profiler

logger = logging.getLogger(__name__)


@dataclass
class UnidentifiedFaceTask:
    """Task for background processing of unidentified face."""

    camera_id: str
    track_id: Optional[str]
    face_crop: np.ndarray
    embedding: np.ndarray
    best_match_person_id: Optional[int]
    best_match_score: Optional[float]
    face_bbox: tuple[int, int, int, int]  # x1, y1, x2, y2


class UnidentifiedFaceManager:
    """Thread-safe manager for capturing and storing unidentified faces.

    Features:
    - Non-blocking submit() for main thread
    - Background worker for quality scoring and storage
    - Diversity check to avoid duplicates
    - Per-camera limit enforcement
    - Automatic cleanup of old entries
    """

    def __init__(
        self,
        config: UnidentifiedFacesConfig,
        repository: Repository,
    ):
        """Initialize the manager.

        Args:
            config: Unidentified faces configuration
            repository: Database repository
        """
        self.config = config
        self.repository = repository

        # Processing queue
        self._queue: Queue[UnidentifiedFaceTask] = Queue(maxsize=config.queue_size)
        self._stop_event = Event()
        self._worker_thread: Optional[Thread] = None

        # Ensure images directory exists
        self._images_path = Path(config.images_path)
        self._images_path.mkdir(parents=True, exist_ok=True)

        # Last cleanup time
        self._last_cleanup = 0.0
        self._cleanup_interval = 3600  # 1 hour

        logger.info(
            f"UnidentifiedFaceManager initialized "
            f"(images_path={config.images_path}, max_per_camera={config.max_per_camera})"
        )

    def start(self):
        """Start the background worker thread."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        self._stop_event.clear()
        self._worker_thread = Thread(
            target=self._worker_loop,
            name="UnidentifiedFaceWorker",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info("UnidentifiedFaceManager worker started")

    def stop(self):
        """Stop the background worker thread."""
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5.0)
            self._worker_thread = None
        logger.info("UnidentifiedFaceManager worker stopped")

    def submit(
        self,
        camera_id: str,
        face_crop: np.ndarray,
        embedding: np.ndarray,
        face_bbox: tuple[int, int, int, int],
        track_id: Optional[str] = None,
        best_match_person_id: Optional[int] = None,
        best_match_score: Optional[float] = None,
    ) -> bool:
        """Submit an unidentified face for background processing.

        Non-blocking - returns immediately.

        Args:
            camera_id: Camera ID
            face_crop: Face crop image
            embedding: Face embedding (512-dim)
            face_bbox: Face bounding box (x1, y1, x2, y2)
            track_id: Optional track ID
            best_match_person_id: Best matching person ID (if any)
            best_match_score: Similarity score to best match

        Returns:
            True if submitted, False if queue is full
        """
        if not self.config.enabled:
            return False

        # Quick validation
        if face_crop.size == 0 or embedding.size != 512:
            return False

        # Check minimum match score for best match
        if best_match_score is not None:
            if best_match_score < self.config.min_match_score:
                best_match_person_id = None
                best_match_score = None

        task = UnidentifiedFaceTask(
            camera_id=camera_id,
            track_id=track_id,
            face_crop=face_crop.copy(),  # Copy to avoid race conditions
            embedding=embedding.copy(),
            best_match_person_id=best_match_person_id,
            best_match_score=best_match_score,
            face_bbox=face_bbox,
        )

        try:
            self._queue.put_nowait(task)
            return True
        except Exception:
            # Queue full
            logger.debug("Unidentified face queue full, dropping submission")
            return False

    def _worker_loop(self):
        """Background worker loop."""
        while not self._stop_event.is_set():
            try:
                task = self._queue.get(timeout=0.5)
                self._process_task(task)
                self._queue.task_done()
            except Empty:
                # Check for periodic cleanup
                self._maybe_cleanup()
                continue
            except Exception as e:
                logger.error(f"Error processing unidentified face: {e}")

    def _process_task(self, task: UnidentifiedFaceTask):
        """Process a single unidentified face task.

        1. Compute quality score
        2. Check diversity against existing embeddings
        3. Save image to disk
        4. Store in database
        5. Enforce per-camera limit
        """
        try:
            # 1. Compute quality score
            quality_score, sharpness_score = self._compute_quality(
                task.face_crop, task.face_bbox
            )

            # Check minimum quality
            if quality_score < self.config.min_quality_score:
                logger.debug(
                    f"Unidentified face quality too low: {quality_score:.2f} < {self.config.min_quality_score}"
                )
                return

            # Check minimum sharpness (rejects motion blur and focus blur)
            if sharpness_score < self.config.min_sharpness_score:
                logger.debug(
                    f"Unidentified face too blurry: sharpness {sharpness_score:.2f} < {self.config.min_sharpness_score}"
                )
                return

            # Check minimum face size
            face_width = task.face_bbox[2] - task.face_bbox[0]
            if face_width < self.config.min_face_size:
                logger.debug(
                    f"Unidentified face too small: {face_width}px < {self.config.min_face_size}px"
                )
                return

            # 2. Diversity check
            if not self._is_diverse(task.camera_id, task.embedding):
                logger.debug(
                    f"Unidentified face too similar to existing, skipping"
                )
                return

            # 3. Save image to disk
            image_path = self._save_image(task.camera_id, task.face_crop)
            if not image_path:
                logger.error("Failed to save unidentified face image")
                return

            # 4. Store in database
            # Note: blur_score column stores sharpness score (0-1, higher = sharper)
            face = self.repository.add_unidentified_face(
                camera_id=task.camera_id,
                embedding=task.embedding,
                image_path=image_path,
                quality_score=float(quality_score),
                track_id=task.track_id,
                best_match_person_id=task.best_match_person_id,
                best_match_score=float(task.best_match_score) if task.best_match_score is not None else None,
                blur_score=float(sharpness_score),
                face_size=int(face_width),
            )

            # Emit event
            self.repository.create_event(
                camera_id=task.camera_id,
                event_type="unidentified_face_saved",
                track_id=task.track_id,
                person_id=task.best_match_person_id,
                confidence=float(task.best_match_score) if task.best_match_score is not None else None,
                snapshot_path=image_path,
                extra_data={
                    "quality_score": float(quality_score),
                    "face_id": int(face.id),
                    "best_match_person_id": task.best_match_person_id,
                    "best_match_score": float(task.best_match_score) if task.best_match_score is not None else None,
                }
            )

            logger.info(
                f"Captured unidentified face {face.id} from {task.camera_id} "
                f"(quality={quality_score:.2f}, match_score={task.best_match_score})"
            )

            # 5. Enforce per-camera limit
            self.repository.cleanup_unidentified_faces_for_camera(
                task.camera_id, self.config.max_per_camera
            )

        except Exception as e:
            logger.error(f"Error processing unidentified face task: {e}")

    def _compute_quality(
        self, face_crop: np.ndarray, face_bbox: tuple[int, int, int, int]
    ) -> tuple[float, float]:
        """Compute quality score for a face crop.

        Quality is based on:
        - Face size (larger is better)
        - Sharpness (FFT-based, catches both focus and motion blur)
        - Brightness (not too dark, not too bright)

        Performance: ~8-10ms per call (FFT is the bottleneck).
        Runs in background thread so doesn't block main processing.

        Args:
            face_crop: Face crop image
            face_bbox: Face bounding box

        Returns:
            Tuple of (quality_score, sharpness_score)
        """
        if face_crop.size == 0:
            return 0.0, 0.0

        # Face size score (0-1)
        face_width = face_bbox[2] - face_bbox[0]
        size_score = min(1.0, face_width / 200.0)  # 200px = perfect

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop

        # Sharpness score using FFT high-frequency ratio
        # This catches both focus blur AND motion blur
        with profiler.measure("QualityCheck.sharpness_fft"):
            sharpness_score = self._compute_sharpness_fft(gray)

        # Brightness score (0-1, penalize extremes)
        mean_brightness = np.mean(gray)
        # Ideal brightness around 128, penalize dark (<50) and bright (>200)
        if mean_brightness < 50:
            brightness_score = mean_brightness / 50.0
        elif mean_brightness > 200:
            brightness_score = (255 - mean_brightness) / 55.0
        else:
            brightness_score = 1.0

        # Combined quality score (weighted average)
        quality_score = (
            0.4 * size_score +
            0.4 * sharpness_score +
            0.2 * brightness_score
        )

        return quality_score, sharpness_score

    def _compute_sharpness_fft(self, gray: np.ndarray) -> float:
        """Compute sharpness using FFT high-frequency content ratio.

        This method detects both focus blur and motion blur by analyzing
        the frequency domain. Blurry images (from any cause) have less
        high-frequency content.

        Args:
            gray: Grayscale image

        Returns:
            Sharpness score 0-1 (higher = sharper)
        """
        # Compute 2D FFT and shift zero frequency to center
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        h, w = gray.shape
        center_y, center_x = h // 2, w // 2

        # Create distance mask from center
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        # High frequency = outer region (beyond 25% of image size)
        radius_threshold = min(h, w) * 0.25
        high_freq_mask = dist > radius_threshold

        # Compute ratio of high frequency to total energy
        total_energy = np.sum(magnitude)
        if total_energy < 1e-6:
            return 0.0

        high_freq_energy = np.sum(magnitude[high_freq_mask])
        high_freq_ratio = high_freq_energy / total_energy

        # Normalize: sharp images have ratio ~0.7+, blurry ~0.3-0.5
        # Map to 0-1 score: 0.3 -> 0, 0.7 -> 1
        sharpness_score = (high_freq_ratio - 0.3) / 0.4
        sharpness_score = max(0.0, min(1.0, sharpness_score))

        return sharpness_score

    def _is_diverse(self, camera_id: str, embedding: np.ndarray) -> bool:
        """Check if the embedding is diverse enough from existing ones.

        Args:
            camera_id: Camera ID
            embedding: New embedding to check

        Returns:
            True if diverse enough, False if too similar
        """
        existing_embeddings = self.repository.get_recent_unidentified_embeddings(
            camera_id, limit=self.config.max_per_camera
        )

        if not existing_embeddings:
            return True

        # Normalize embedding
        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-8)

        for existing in existing_embeddings:
            existing_norm = existing / (np.linalg.norm(existing) + 1e-8)
            similarity = np.dot(embedding_norm, existing_norm)

            if similarity > self.config.embedding_similarity_threshold:
                return False

        return True

    def _save_image(self, camera_id: str, face_crop: np.ndarray) -> Optional[str]:
        """Save face crop to disk.

        Args:
            camera_id: Camera ID (used for subdirectory)
            face_crop: Face image to save

        Returns:
            Path to saved image, or None on failure
        """
        try:
            # Create camera subdirectory
            camera_dir = self._images_path / camera_id
            camera_dir.mkdir(parents=True, exist_ok=True)

            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = uuid.uuid4().hex[:8]
            filename = f"{timestamp}_{unique_id}.jpg"
            image_path = camera_dir / filename

            # Save image
            cv2.imwrite(str(image_path), face_crop)

            return str(image_path)
        except Exception as e:
            logger.error(f"Failed to save unidentified face image: {e}")
            return None

    def _maybe_cleanup(self):
        """Run periodic cleanup if needed."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = now

        try:
            # Cleanup old faces
            deleted = self.repository.cleanup_old_unidentified_faces(
                days=self.config.retention_days
            )
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old unidentified faces")

            # Cleanup orphaned image files
            self._cleanup_orphaned_images()
        except Exception as e:
            logger.error(f"Error during unidentified faces cleanup: {e}")

    def _cleanup_orphaned_images(self):
        """Remove image files that are no longer in the database."""
        try:
            # Get all image paths from database
            faces = self.repository.get_unidentified_faces(limit=10000)
            db_paths = {f.image_path for f in faces}

            # Walk the images directory
            for camera_dir in self._images_path.iterdir():
                if not camera_dir.is_dir():
                    continue

                for image_file in camera_dir.iterdir():
                    if not image_file.is_file():
                        continue

                    image_path = str(image_file)
                    if image_path not in db_paths:
                        try:
                            os.remove(image_path)
                            logger.debug(f"Removed orphaned image: {image_path}")
                        except Exception as e:
                            logger.warning(f"Failed to remove orphaned image: {e}")

        except Exception as e:
            logger.error(f"Error cleaning up orphaned images: {e}")
