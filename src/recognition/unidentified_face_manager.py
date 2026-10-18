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
from src.recognition.face_quality import FaceQualityAssessor
from src.utils.embedding_utils import cosine_similarity_batch

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
    face_bbox: tuple[int, int, int, int]  # detected face box x1, y1, x2, y2 (not the margin crop)
    best_match_face_id: Optional[int] = None
    # Face detection confidence (stored for analysis)
    det_confidence: Optional[float] = None
    # ArcFace-aligned 112x112 face (`align_face` on the landmarks) — what the
    # quality model scores; None without landmarks (never saved)
    aligned_face: Optional[np.ndarray] = None
    # Raw ArcFace output norm (logged for quality analysis, not gated on yet)
    embedding_norm: Optional[float] = None


class UnidentifiedFaceManager:
    """Thread-safe manager for capturing and storing unidentified faces.

    Features:
    - Non-blocking submit() for main thread
    - Background worker for quality scoring (`FaceQualityAssessor`) and storage
    - One suggestion per track: its best crop so far; a better one archives
      the previous (row and image kept, `superseded_by_id`)
    - Diversity check to avoid duplicates across tracks
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

        self._quality_assessor: Optional[FaceQualityAssessor] = None

        # Last cleanup time
        self._last_cleanup = 0.0
        self._cleanup_interval = 3600  # 1 hour

        logger.info(
            f"UnidentifiedFaceManager initialized "
            f"(images_path={config.images_path}, max_per_camera={config.max_per_camera})"
        )

    @property
    def quality_assessor(self) -> FaceQualityAssessor:
        if self._quality_assessor is None:
            self._quality_assessor = FaceQualityAssessor(self.config.quality_model)
        return self._quality_assessor

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
        best_match_face_id: Optional[int] = None,
        det_confidence: Optional[float] = None,
        aligned_face: Optional[np.ndarray] = None,
        embedding_norm: Optional[float] = None,
    ) -> bool:
        """Submit an unidentified face for background processing.

        Non-blocking - returns immediately.

        Args:
            camera_id: Camera ID
            face_crop: Face crop image
            embedding: Face embedding (512-dim)
            face_bbox: Detected face box (x1, y1, x2, y2), not the margin crop
            track_id: Optional track ID
            best_match_person_id: Best matching person ID (if any)
            best_match_score: Similarity score to best match
            best_match_face_id: ID of the best matching face embedding
            det_confidence: Face detection confidence (stored for analysis)
            aligned_face: ArcFace-aligned face the quality model scores
            embedding_norm: Raw ArcFace output norm (stored in the event for analysis)

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
                best_match_face_id = None

        task = UnidentifiedFaceTask(
            camera_id=camera_id,
            track_id=track_id,
            face_crop=face_crop.copy(),  # Copy to avoid race conditions
            embedding=embedding.copy(),
            best_match_person_id=best_match_person_id,
            best_match_score=best_match_score,
            face_bbox=face_bbox,
            best_match_face_id=best_match_face_id,
            det_confidence=det_confidence,
            aligned_face=aligned_face.copy() if aligned_face is not None else None,
            embedding_norm=embedding_norm,
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

        1. Face-size gate
        2. Quality gate (`FaceQualityAssessor`, one learned score)
        3. One suggestion per track: keep only a crop better than the track's
           current suggestion (live 2026-10-18: one 4-min track of a face
           half-hidden by a phone produced 13 suggestions 10 s apart and
           evicted another person's crops from the per-camera cap; same-person
           crops in dim light embed only 0.4-0.6 alike, so the embedding
           diversity check never caught them)
        4. Check diversity against other tracks' suggestions
        5. Save image to disk, store in database, archive the replaced one
        6. Enforce per-camera limit
        """
        try:
            # 1. Face size (cheap, before scoring): the quality model resizes every
            # face to 112x112 and cannot see how much a small one was upscaled
            # (a 44 px face scored 0.42, like a clean 220 px grimace)
            x1, y1, x2, y2 = task.face_bbox
            face_size = min(x2 - x1, y2 - y1)
            if face_size < self.config.min_face_size:
                logger.debug(
                    f"Unidentified face too small: {face_size}px < {self.config.min_face_size}px"
                )
                return

            # 2. Quality: how usable the face is for recognition (one learned
            # score, see face_quality.py). No aligned face = no landmarks:
            # cannot be scored, never suggested.
            if task.aligned_face is None:
                logger.debug("Unidentified face without landmarks, skipping")
                return
            quality_score = self.quality_assessor.score(task.aligned_face)
            if quality_score < self.config.min_quality_score:
                logger.debug(
                    f"Unidentified face quality too low: {quality_score:.2f} < {self.config.min_quality_score}"
                )
                return

            # 3. One suggestion per track: only a better crop replaces it
            current = self.repository.get_track_suggestion(task.track_id) if task.track_id else None
            if current is not None and quality_score <= current.quality_score:
                logger.debug(
                    f"Unidentified face not better than track {task.track_id}'s suggestion "
                    f"{current.id}: {quality_score:.2f} <= {current.quality_score:.2f}"
                )
                return

            # 4. Diversity check (a track's first suggestion only; a better crop
            # of the same track replaces its own suggestion)
            if current is None and not self._is_diverse(task.camera_id, task.embedding):
                logger.debug(
                    f"Unidentified face too similar to existing, skipping"
                )
                return

            # 5. Save image to disk
            image_path = self._save_image(task.camera_id, task.face_crop)
            if not image_path:
                logger.error("Failed to save unidentified face image")
                return

            # Store in database
            face = self.repository.add_unidentified_face(
                camera_id=task.camera_id,
                embedding=task.embedding,
                image_path=image_path,
                quality_score=float(quality_score),
                track_id=task.track_id,
                best_match_person_id=task.best_match_person_id,
                best_match_score=float(task.best_match_score) if task.best_match_score is not None else None,
                face_size=int(face_size),
                det_confidence=float(task.det_confidence) if task.det_confidence is not None else None,
            )
            if current is not None:
                self.repository.archive_unidentified_face(current.id, superseded_by_id=face.id)

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
                    "embedding_norm": task.embedding_norm,
                    "unidentified_face_id": int(face.id),
                    "supersedes_face_id": int(current.id) if current is not None else None,
                    "best_match_face_id": task.best_match_face_id,
                    "best_match_person_id": task.best_match_person_id,
                    "best_match_score": float(task.best_match_score) if task.best_match_score is not None else None,
                }
            )

            logger.info(
                f"Captured unidentified face {face.id} from {task.camera_id} "
                f"(quality={quality_score:.2f}, match_score={task.best_match_score})"
            )

            # 6. Enforce per-camera limit (archived faces don't count)
            self.repository.cleanup_unidentified_faces_for_camera(
                task.camera_id, self.config.max_per_camera
            )

        except Exception as e:
            logger.error(f"Error processing unidentified face task: {e}")

    def _is_diverse(self, camera_id: str, embedding: np.ndarray) -> bool:
        """Check if the embedding is diverse enough from existing ones.

        Args:
            camera_id: Camera ID
            embedding: New embedding to check

        Returns:
            True if diverse enough, False if too similar
        """
        existing_embeddings = list(self.repository.get_recent_unidentified_embeddings(
            camera_id, limit=self.config.max_per_camera
        ))

        if not existing_embeddings:
            return True

        similarities = cosine_similarity_batch(embedding, np.stack(existing_embeddings))
        return not bool(np.any(similarities > self.config.embedding_similarity_threshold))

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
