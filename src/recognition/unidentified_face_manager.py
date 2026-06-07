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
    best_match_face_id: Optional[int] = None
    # InsightFace detection confidence — best free proxy for face usability
    det_confidence: Optional[float] = None
    # 5-point landmarks (left_eye, right_eye, nose, mouth_l, mouth_r) in any
    # coordinate frame — frontalness uses only relative geometry
    landmarks: Optional[np.ndarray] = None


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

        # Burst suppression: last save time per track (worker thread only)
        self._last_saved_per_track: dict[str, float] = {}

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
        best_match_face_id: Optional[int] = None,
        det_confidence: Optional[float] = None,
        landmarks: Optional[np.ndarray] = None,
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
            best_match_face_id: ID of the best matching face embedding
            det_confidence: Face detection confidence (factored into quality)
            landmarks: 5-point facial landmarks (used for the frontalness gate)

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
            landmarks=landmarks.copy() if landmarks is not None else None,
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
            # 1. Head pose. A crop without landmarks cannot be pose-checked and
            # is untrustworthy for enrollment (live detections always carry
            # landmarks) — score it 0.0 so the gate below rejects it.
            frontalness = (
                self._compute_frontalness(task.landmarks)
                if task.landmarks is not None else 0.0
            )

            # Check frontalness (rejects profile/looking-down poses and
            # landmark-less crops — they produce weak embeddings unsuitable
            # for enrollment)
            if self.config.min_frontalness > 0 and frontalness < self.config.min_frontalness:
                logger.debug(
                    f"Unidentified face not frontal enough: "
                    f"{frontalness:.2f} < {self.config.min_frontalness}"
                )
                return

            # 2. Compute quality score
            quality_score, sharpness_score = self._compute_quality(
                task.face_crop, task.face_bbox, task.det_confidence, frontalness
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

            # 3. Burst suppression: one sitting episode can produce a dozen
            # near-duplicate crops within seconds, churning the review queue
            # and evicting diverse older candidates (per-camera cap).
            if task.track_id and self.config.burst_min_interval_sec > 0:
                now = time.time()
                last_saved = self._last_saved_per_track.get(task.track_id, 0.0)
                if now - last_saved < self.config.burst_min_interval_sec:
                    logger.debug(
                        f"Unidentified face burst-suppressed for track {task.track_id} "
                        f"({now - last_saved:.1f}s < {self.config.burst_min_interval_sec}s)"
                    )
                    return

            # 4. Diversity check
            if not self._is_diverse(task.camera_id, task.embedding):
                logger.debug(
                    f"Unidentified face too similar to existing, skipping"
                )
                return

            # 5. Save image to disk
            image_path = self._save_image(task.camera_id, task.face_crop)
            if not image_path:
                logger.error("Failed to save unidentified face image")
                return
            if task.track_id:
                self._last_saved_per_track[task.track_id] = time.time()

            # 6. Store in database
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
                frontalness=float(frontalness),
                det_confidence=float(task.det_confidence) if task.det_confidence is not None else None,
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
                    "frontalness": float(frontalness),
                    "unidentified_face_id": int(face.id),
                    "best_match_face_id": task.best_match_face_id,
                    "best_match_person_id": task.best_match_person_id,
                    "best_match_score": float(task.best_match_score) if task.best_match_score is not None else None,
                }
            )

            logger.info(
                f"Captured unidentified face {face.id} from {task.camera_id} "
                f"(quality={quality_score:.2f}, match_score={task.best_match_score})"
            )

            # 7. Enforce per-camera limit
            self.repository.cleanup_unidentified_faces_for_camera(
                task.camera_id, self.config.max_per_camera
            )

        except Exception as e:
            logger.error(f"Error processing unidentified face task: {e}")

    def _compute_quality(
        self,
        face_crop: np.ndarray,
        face_bbox: tuple[int, int, int, int],
        det_confidence: Optional[float] = None,
        frontalness: float = 1.0,
    ) -> tuple[float, float]:
        """Compute quality score for a face crop.

        Quality is based on:
        - Face size (larger is better)
        - Sharpness (min(focus, motion, detail) at a fixed 112x112 input)
        - Brightness (not too dark, not too bright)
        - Detection confidence (when provided — best free proxy for usability)
        - Frontalness (head pose; profile/look-down crops embed poorly)

        Performance: ~2-3ms per call (small convolutions on the fixed 112x112 input).
        Runs in background thread so doesn't block main processing.

        Args:
            face_crop: Face crop image
            face_bbox: Face bounding box
            det_confidence: Face detection confidence (0-1), optional
            frontalness: Head pose score (0-1, 1 = frontal)

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
        with profiler.measure("QualityCheck.sharpness"):
            sharpness_score = self._compute_sharpness(gray)

        # Brightness score (0-1, penalize extremes)
        mean_brightness = np.mean(gray)
        # Ideal brightness around 128, penalize dark (<50) and bright (>200)
        if mean_brightness < 50:
            brightness_score = mean_brightness / 50.0
        elif mean_brightness > 200:
            brightness_score = (255 - mean_brightness) / 55.0
        else:
            brightness_score = 1.0

        # Combined quality score (weighted average). Detection confidence is
        # the best free proxy for face usability (it tracked embedding utility
        # better than size/sharpness in live-data analysis); frontalness added
        # 2026-06-07 — pose junk (profiles, top-of-head) scored high on pure
        # photographic metrics because hair/background texture reads as detail.
        if det_confidence is not None:
            quality_score = (
                0.25 * size_score +
                0.15 * sharpness_score +
                0.15 * brightness_score +
                0.25 * min(1.0, det_confidence) +
                0.2 * frontalness
            )
        else:
            quality_score = (
                0.3 * size_score +
                0.3 * sharpness_score +
                0.15 * brightness_score +
                0.25 * frontalness
            )

        return quality_score, sharpness_score

    # Fixed analysis size: blur measures depend on image size and JPEG noise,
    # so scores at native size are not comparable across face sizes. 112x112
    # matches the aligned-face input size of the recognition model, i.e. we
    # measure sharpness at the scale where the embedding is actually computed.
    _SHARPNESS_INPUT_SIZE = 112

    # Immerkaer noise-estimation kernel, reused as a fine-grain detail probe
    _DETAIL_KERNEL = np.array(
        [[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32
    )

    def _compute_sharpness(self, gray: np.ndarray) -> float:
        """Compute sharpness as min(focus, motion, detail) at a fixed 112x112.

        The previous FFT high-frequency-ratio metric measured sensor/JPEG noise
        as much as detail: on labeled live data it ranked a heavily
        motion-blurred face ABOVE every visually usable crop. Replaced by three
        complementary measures on a fixed 112x112 grayscale (focus and motion
        on a lightly denoised copy so noise doesn't masquerade as detail):

        - Focus: Laplacian variance — catches gaussian/defocus blur.
          Calibration (live 1080p home-camera crops, 2026-06-06): usable
          crops 58-170, gaussian-blurred copies 4-32 -> map 10->0, 60->1.
        - Motion: gradient-direction anisotropy min(E_gx,E_gy)/max(E_gx,E_gy)
          — motion blur kills gradients along the smear direction even when
          strong edges keep the Laplacian high. Calibration: usable crops
          0.72-0.93, motion-blurred sample 0.62, heavy gaussian 0.56
          -> map 0.55->0, 0.75->1.
        - Detail: fine-grain energy (Immerkaer noise-sigma estimator on the
          raw, non-denoised input). Low-light camera temporal NR smears away
          texture while leaving strong edges, so such crops pass the Laplacian
          (which saturates at 60) yet look smeared/blurry. Calibration
          (evening captures 2026-06-06 19:13 local): smeared crops 0.64-1.00
          vs usable daylight crops 1.55-3.52 -> map 0.8->0, 1.7->1.

        With the default min_sharpness_score gate of 0.4: usable live crops
        score 0.83-1.0, the motion-blurred sample 0.36, NR-smeared evening
        crops <=0.22, gaussian-blurred copies <=0.03 (all rejected).

        Performance: ~1ms per call (fixed-size resize + small convolutions).

        Args:
            gray: Grayscale image

        Returns:
            Sharpness score 0-1 (higher = sharper)
        """
        size = self._SHARPNESS_INPUT_SIZE
        gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)

        # Focus blur: Laplacian variance (10 -> 0, 60 -> 1)
        lap_var = float(cv2.Laplacian(denoised, cv2.CV_32F).var())
        focus_score = max(0.0, min(1.0, (lap_var - 10.0) / 50.0))

        # Motion blur: gradient-direction anisotropy (0.55 -> 0, 0.75 -> 1)
        gx = cv2.Sobel(denoised, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(denoised, cv2.CV_32F, 0, 1)
        ex = float(np.sum(gx ** 2))
        ey = float(np.sum(gy ** 2))
        anisotropy = min(ex, ey) / max(ex, ey) if max(ex, ey) > 0 else 0.0
        motion_score = max(0.0, min(1.0, (anisotropy - 0.55) / 0.20))

        # NR over-smoothing: fine-grain energy on the RAW input (0.8 -> 0, 1.7 -> 1)
        conv = cv2.filter2D(gray.astype(np.float32), -1, self._DETAIL_KERNEL)
        hf_energy = float(
            np.sqrt(np.pi / 2.0) * np.sum(np.abs(conv)) / (6.0 * (size - 2) * (size - 2))
        )
        detail_score = max(0.0, min(1.0, (hf_energy - 0.8) / 0.9))

        return min(focus_score, motion_score, detail_score)

    @staticmethod
    def _compute_frontalness(landmarks: np.ndarray) -> float:
        """Score head pose from 5-point landmarks (1 = frontal, 0 = profile/down).

        Profile and looking-down poses produce weak ArcFace embeddings that are
        unsuitable for enrollment. Two scale/translation-invariant proxies
        (calibrated on labeled live crops, 2026-06-06):

        - Yaw: |nose offset along the eye axis from the face center| / eye_dist.
          Frontal faces 0.01-0.44, profiles 0.79-3.01 -> 1.0 below 0.35, 0 at 0.8.
        - Pose ratio: eye-to-mouth distance / inter-eye distance. Frontal
          1.03-1.40; yaw collapses eye_dist (profiles 2.09-6.88), pitch-down
          collapses face height (look-down sample 0.66)
          -> 0 at 0.6, 1 at 0.85; 1 until 1.5, 0 at 2.2.

        Args:
            landmarks: 5x2 array (left_eye, right_eye, nose, mouth_l, mouth_r)

        Returns:
            Frontalness score 0-1 (frontal live crops >=0.79, profile and
            look-down crops <=0.23)
        """
        if landmarks is None or landmarks.shape[0] < 5:
            return 1.0  # No landmarks — don't gate

        le, re, nose, ml, mr = np.asarray(landmarks, dtype=np.float64)[:5]
        mid_eye = (le + re) / 2.0
        mid_mouth = (ml + mr) / 2.0
        eye_vec = re - le
        eye_dist = float(np.linalg.norm(eye_vec))
        face_height = float(np.linalg.norm(mid_mouth - mid_eye))
        if eye_dist < 1e-6 or face_height < 1e-6:
            return 0.0

        # Yaw: nose offset along the eye axis from the face center
        axis = eye_vec / eye_dist
        center = (mid_eye + mid_mouth) / 2.0
        yaw_offset = abs(float(np.dot(nose - center, axis))) / eye_dist
        yaw_score = max(0.0, min(1.0, 1.0 - (yaw_offset - 0.35) / 0.45))

        # Pose ratio: face height vs eye distance
        ratio = face_height / eye_dist
        if ratio < 1.0:
            ratio_score = max(0.0, min(1.0, (ratio - 0.6) / 0.25))
        else:
            ratio_score = max(0.0, min(1.0, 1.0 - (ratio - 1.5) / 0.7))

        return min(yaw_score, ratio_score)

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

        # Prune stale burst-suppression entries (track ids grow forever)
        cutoff = now - max(self.config.burst_min_interval_sec, 60.0)
        self._last_saved_per_track = {
            t: ts for t, ts in self._last_saved_per_track.items() if ts > cutoff
        }

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
