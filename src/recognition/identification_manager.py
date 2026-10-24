"""Unified identification manager for face recognition and Re-ID.

This module provides a single class that handles:
1. Face recognition to identify known persons
2. Re-ID for cross-camera tracking and re-appearance detection
3. Identity state management for tracks
4. Optional database integration for production use

Can be used in both preview mode (no database) and production mode (with database).
"""

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Callable

import numpy as np

from src.config import Config, FaceRecognitionConfig, ReIDConfig
from src.recognition.face_recognizer import Face, FaceRecognizer
from src.recognition.id_method import IdMethod
from src.recognition.reid_extractor import ReIDExtractor, is_grayscale_image
from src.recognition.reid_gallery import ReIDGalleryManager, DebugImageSaver
from src.recognition.tensorrt_face import align_face
from src.tracking.attribution import owner_face, skeleton_face
from src.utils.embedding_utils import cosine_similarity_batch
from src.utils.image_utils import crop_bbox, crop_face_region, crop_with_margin
from src.utils.profiler import profiler

if TYPE_CHECKING:
    from src.database.repository import Repository
    from src.recognition.face_score_logger import FaceScoreLogger
    from src.recognition.unidentified_face_manager import UnidentifiedFaceManager

logger = logging.getLogger(__name__)


@dataclass
class IdentificationResult:
    """Result of an identification attempt."""
    identified: bool
    person_name: Optional[str] = None
    confidence: float = 0.0


@dataclass
class FaceGalleryEntry:
    """Entry in the face recognition gallery."""
    person_id: int
    person_name: str
    embedding: np.ndarray
    embedding_id: Optional[int] = None


@dataclass
class FaceMatchScores:
    """Similarity of one face embedding against the gallery, aggregated per person."""

    best_entry: Optional[FaceGalleryEntry] = None  # gallery entry with the highest similarity
    top1: float = 0.0  # best similarity overall (person of best_entry)
    top2: float = 0.0  # best similarity of any OTHER person (0.0 if none)
    per_person: dict[int, float] = field(default_factory=dict)  # person_id -> max similarity

    @property
    def margin(self) -> float:
        return self.top1 - self.top2

    def accepts(self, threshold: float, min_margin: float = 0.0) -> bool:
        """Match policy: top1 above threshold AND clear lead over the runner-up person."""
        return self.best_entry is not None and self.top1 >= threshold and self.margin >= min_margin


def score_against_gallery(
    embedding: np.ndarray,
    gallery: list[FaceGalleryEntry],
    similarity: Callable[[np.ndarray, np.ndarray], np.ndarray] = cosine_similarity_batch,
) -> FaceMatchScores:
    """Score a face embedding against gallery entries, per person (~0.1 ms for ~100 entries).

    The one scoring rule: live recognition (`IdentificationManager.score_face`, with the
    recognizer's batch comparison) and the review UI (unidentified face suggestions, API)
    both use it.
    """
    if not gallery:
        return FaceMatchScores()
    similarities = similarity(embedding, np.array([entry.embedding for entry in gallery]))
    best_idx = int(np.argmax(similarities))
    best_entry = gallery[best_idx]

    per_person: dict[int, float] = {}
    for entry, sim in zip(gallery, similarities):
        if sim > per_person.get(entry.person_id, -1.0):
            per_person[entry.person_id] = float(sim)
    others = [s for pid, s in per_person.items() if pid != best_entry.person_id]

    return FaceMatchScores(
        best_entry=best_entry,
        top1=float(similarities[best_idx]),
        top2=max(others, default=0.0),
        per_person=per_person,
    )


# Face confirmation: the same person must be accepted this many attempts in a row,
# unless one attempt is strong (face_recognition.strong_threshold / strong_min_margin)
FACE_CONSECUTIVE_MATCHES = 2


@dataclass
class FaceAttempt:
    """Outcome of one face recognition attempt on a person box (`attempt_face`)."""

    face: Face  # the face the box owns (attribution.owner_face)
    embedding: np.ndarray
    scores: FaceMatchScores
    region: np.ndarray  # head region the face bbox refers to
    num_faces: int  # faces of at least min_face_size in the region
    ambiguous: bool  # no single face owns the box: logged only, never used
    accepted: bool  # this attempt passes the match policy
    confirmed: bool  # accepted FACE_CONSECUTIVE_MATCHES times in a row for the same person, or strong
    strong: bool = False  # this attempt alone confirms (strong_threshold / strong_min_margin)


@dataclass
class TrackIdentity:
    """Identity information for a track."""

    track_id: str = ""
    person_id: Optional[int] = None
    person_name: Optional[str] = None
    confidence: float = 0.0

    method: IdMethod = IdMethod.NONE

    # Scores for display
    reid_score: float = -1.0
    reid_info: Optional[tuple[str, float]] = None  # (closest_name, confidence)
    face_info: Optional[tuple[str, float]] = None  # (closest_name, confidence)

    # Multi-face detection flag
    has_multiple_faces: bool = False

    # Stationary time for display
    stationary_time: Optional[int] = None

    # Timestamps
    first_seen: float = field(default_factory=time.time)
    last_face_check: float = 0.0
    identified_at: Optional[float] = None

    # Face confirmation streak (FACE_CONSECUTIVE_MATCHES)
    candidate_person_id: Optional[int] = None
    consecutive_matches: int = 0

    @property
    def is_identified(self) -> bool:
        """Check if track has been identified."""
        return self.person_name is not None

    @property
    def is_face_identified(self) -> bool:
        return self.method == IdMethod.FACE

    @property
    def is_reid_identified(self) -> bool:
        return self.method.is_appearance


class IdentificationManager:
    """Unified manager for face recognition and Re-ID.

    Works in two modes:
    1. Preview mode: Pass face_gallery directly, no database
    2. Production mode: Pass repository, gallery loaded from database with TTL

    Example (preview mode):
        gallery = [FaceGalleryEntry(person_id=1, person_name="Alice", embedding=emb)]
        manager = IdentificationManager(config, face_gallery=gallery)

    Example (production mode):
        manager = IdentificationManager(config, repository=repo)
    """

    def __init__(
        self,
        config: Config,
        face_gallery: Optional[list[FaceGalleryEntry]] = None,
        repository: Optional["Repository"] = None,
        enable_debug_images: bool = False,
        face_recognizer: Optional[FaceRecognizer] = None,
        reid_extractor: Optional[ReIDExtractor] = None,
        unidentified_face_manager: Optional["UnidentifiedFaceManager"] = None,
        face_score_logger: Optional["FaceScoreLogger"] = None,
    ):
        """Initialize identification manager.

        Args:
            config: Application configuration
            face_gallery: Pre-loaded face gallery - for preview.
            repository: Database repository - for production (loads gallery automatically)
            enable_debug_images: Whether to save debug images
            face_recognizer: Optional pre-initialized face recognizer (for testing/shared use)
            unidentified_face_manager: Optional manager for capturing unidentified faces
            reid_extractor: Optional pre-initialized Re-ID extractor (for testing/shared use)
            face_score_logger: Optional per-attempt face score log (offline tuning)
        """
        self.config = config
        self.face_config = config.face_recognition
        self.reid_config = config.reid
        self.repository = repository

        # Unidentified face manager
        self._unidentified_face_manager = unidentified_face_manager
        self._face_score_logger = face_score_logger

        # Face gallery - either pre-loaded or from database
        self._face_gallery: list[FaceGalleryEntry] = face_gallery or []
        self._gallery_loaded_at = 0.0
        self._gallery_ttl = 60.0  # Reload from DB every 60 seconds

        # Lazy-loaded recognizers
        self._face_recognizer = face_recognizer
        self._reid_extractor = reid_extractor
        self._reid_gallery_manager: Optional[ReIDGalleryManager] = None

        # Debug image saver
        self._debug_saver: Optional[DebugImageSaver] = None
        if enable_debug_images:
            self._debug_saver = DebugImageSaver()

        # Track identity states
        self._identities: dict[str, TrackIdentity] = {}

        # Configuration shortcuts
        self.face_threshold = config.face_recognition.similarity_threshold
        self.face_min_margin = config.face_recognition.min_margin
        self.face_strong_threshold = config.face_recognition.strong_threshold
        self.face_strong_min_margin = config.face_recognition.strong_min_margin
        self.face_check_interval = config.face_recognition.detection_interval
        self.reid_confirm_interval = config.face_recognition.reid_confirmation_interval
        self.reid_embedding_interval = 30
        self.min_consecutive_matches = config.reid.min_consecutive_matches

    @property
    def face_recognizer(self) -> Optional[FaceRecognizer]:
        """Get face recognizer (lazy loading)."""
        if self._face_recognizer is None and self.face_config.enabled:
            self._face_recognizer = FaceRecognizer(self.face_config)
        return self._face_recognizer

    @property
    def reid_extractor(self) -> Optional[ReIDExtractor]:
        """Get Re-ID extractor (lazy loading)."""
        if self._reid_extractor is None and self.reid_config.enabled:
            self._reid_extractor = ReIDExtractor(self.reid_config)
        return self._reid_extractor

    @property
    def reid_gallery_manager(self) -> Optional[ReIDGalleryManager]:
        """Get Re-ID gallery manager (lazy loading)."""
        if self._reid_gallery_manager is None and self.reid_config.enabled:
            extractor = self.reid_extractor
            if extractor:
                gallery_snapshot_dir = None
                if self.config.snapshots.enabled:
                    from pathlib import Path
                    gallery_snapshot_dir = str(
                        Path(self.config.snapshots.path) / "reid_gallery"
                    )
                self._reid_gallery_manager = ReIDGalleryManager(
                    reid_extractor=extractor,
                    similarity_threshold=self.reid_config.similarity_threshold,
                    max_reappear_time_sec=self.reid_config.max_reappear_time_sec,
                    max_embeddings_per_person=self.reid_config.gallery_size,
                    crop_cache_path=self.reid_config.crop_cache_path,
                    gallery_snapshot_dir=gallery_snapshot_dir,
                    face_recognizer=self.face_recognizer,
                    repository=self.repository,
                    debug_saver=self._debug_saver,
                    update_interval_sec=self.reid_config.gallery_update_interval_sec,
                )
        return self._reid_gallery_manager

    @property
    def reid_enabled(self) -> bool:
        """Check if Re-ID is enabled."""
        return self.reid_config.enabled

    @property
    def face_gallery(self) -> list[FaceGalleryEntry]:
        """Get face gallery, loading from database if needed.
        
        Returns:
            List of FaceGalleryEntry
        """
        if self.repository:
            self._load_face_gallery_from_db()
        return self._face_gallery

    def score_face(self, embedding: np.ndarray) -> FaceMatchScores:
        """Score a face embedding against the gallery, per person (~0.1ms for ~100 entries)."""
        if self.face_recognizer is None:
            return FaceMatchScores()
        return score_against_gallery(embedding, self.face_gallery, self.face_recognizer.compare_embeddings_batch)

    @property
    def gallery_size(self) -> int:
        """Number of persons in Re-ID gallery."""
        if self.reid_gallery_manager:
            return self.reid_gallery_manager.gallery_size
        return 0

    def _load_face_gallery_from_db(self):
        """Load face gallery from database with TTL caching."""
        current_time = time.time()
        if current_time - self._gallery_loaded_at < self._gallery_ttl:
            return  # Still fresh

        self._face_gallery = []
        embeddings = self.repository.get_all_face_embeddings()

        for person_id, emb_id, embedding in embeddings:
            person = self.repository.get_person(person_id)
            if person:
                self._face_gallery.append(
                    FaceGalleryEntry(
                        person_id=person_id,
                        person_name=person.name,
                        embedding=embedding,
                        embedding_id=emb_id
                    )
                )

        self._gallery_loaded_at = current_time
        logger.debug(f"Loaded {len(self._face_gallery)} face embeddings")

    def _save_snapshot(self, frame: np.ndarray, event_type: str, person_name: Optional[str] = None) -> Optional[str]:
        """Save a snapshot to disk.

        Returns:
            Path to saved snapshot or None if disabled
        """
        if not self.config.snapshots or not self.config.snapshots.enabled:
            return None

        # Determine if we should save based on event type
        should_save = False
        if event_type == "face_match" and self.config.snapshots.save_on_identification:
            should_save = True
        elif event_type == "reid_match" and self.config.snapshots.save_on_identification:
            should_save = True
        
        if not should_save:
            return None

        try:
            from src.utils.image_utils import write_jpeg
            import uuid
            from pathlib import Path

            snapshot_dir = Path(self.config.snapshots.path)
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            unique_id = uuid.uuid4().hex[:8]
            name_part = f"_{person_name}" if person_name else ""
            filename = f"{timestamp}_{event_type}{name_part}_{unique_id}.jpg"
            filepath = snapshot_dir / filename

            write_jpeg(str(filepath), frame)
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
            return None

    # ==================== Identity State Management ====================

    def get_identity(self, track_id: str) -> TrackIdentity:
        """Get or create identity state for a track."""
        if track_id not in self._identities:
            self._identities[track_id] = TrackIdentity(track_id=track_id)
        return self._identities[track_id]

    def is_identified(self, track_id: str) -> bool:
        """Check if a track is identified."""
        identity = self._identities.get(track_id)
        return identity is not None and identity.is_identified

    def on_track_lost(self, track_id: str, local_track_id: int):
        """Handle track being lost - store Re-ID embeddings for later matching."""
        if not self.reid_gallery_manager:
            return
        identity = self._identities.get(track_id)
        was_face_identified = identity.is_face_identified if identity else False
        self.reid_gallery_manager.on_track_lost(local_track_id, was_face_identified)

    def on_track_removed(self, track_id: str):
        """Clean up identity state when track is removed."""
        self._identities.pop(track_id, None)

    def cleanup_expired(self):
        """Clean up expired Re-ID gallery entries."""
        if self.reid_gallery_manager:
            self.reid_gallery_manager.cleanup_expired()

    def cleanup_old_tracks(self, max_age_seconds: float = 3600):
        """Clean up old track identity states."""
        current_time = time.time()
        expired = [
            tid for tid, state in self._identities.items()
            if current_time - state.first_seen > max_age_seconds
        ]
        for tid in expired:
            del self._identities[tid]

    # ==================== Re-ID Matching ====================

    def process_track(
        self,
        track_id: str,
        crop: np.ndarray,
        isolated: bool,
        camera_id: Optional[str] = None,
        frame: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Try to match a track against Re-ID gallery.

        Args:
            track_id: Track identifier
            crop: Person crop image
            isolated: Box overlaps no other track (src/tracking/attribution.py)
            camera_id: Optional camera ID for snapshot saving
            frame: Optional full frame for snapshot saving

        Returns:
            Gallery crop if matched, None otherwise
        """
        if not self.reid_gallery_manager or crop.size == 0 or not isolated:
            return None

        # Skip grayscale/IR images
        if is_grayscale_image(crop):
            return None

        match_result = self.reid_gallery_manager.match_new_track(crop)
        identity = self.get_identity(track_id)

        if match_result.best_person_name:
            identity.reid_info = (match_result.best_person_name, match_result.best_score)

        if match_result.score >= 0:
            identity.reid_score = match_result.score

        if match_result.matched:
            identity.person_name = match_result.person_name
            identity.confidence = match_result.score
            identity.method = IdMethod.REID_GALLERY
            identity.identified_at = time.time()

            # Save snapshot if frame provided
            snapshot_path = None
            if frame is not None:
                snapshot_path = self._save_snapshot(frame, "reid_match", match_result.person_name)

            # Emit reid_match event
            if self.repository:
                person = self.repository.get_person_by_name(match_result.person_name)
                person_id = person.id if person else None
                self.repository.create_event(
                    camera_id=camera_id or "unknown",
                    event_type="reid_match",
                    track_id=track_id,
                    person_id=person_id,
                    confidence=match_result.score,
                    reid_embedding_id=match_result.db_id,
                    snapshot_path=snapshot_path,
                    extra_data={
                        "match_policy": "threshold",
                        "threshold": self.reid_gallery_manager.similarity_threshold,
                        "top1_score": match_result.best_score,
                        "top1_person_id": person_id
                    }
                )

            return match_result.gallery_crop

        return None

    # ==================== Face Recognition ====================

    def attempt_face(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
        track_id: str,
        camera_id: Optional[str] = None,
        allow_strong: bool = True,
        keypoints: Optional[np.ndarray] = None,
    ) -> Optional[FaceAttempt]:
        """One face recognition attempt for a person box — the only face path (~15-25ms on Jetson TRT).

        Shared by preview (`try_face_recognition`) and production
        (`IdentityLinker._try_face_identification`), so both see the same
        evidence: detect in the head region (upper half of the box), take the face
        the box owns (attribution.skeleton_face from the box's pose `keypoints`, else
        attribution.owner_face), score it per person, count the
        confirmation streak, log the attempt (FaceScoreLogger) and keep an
        unaccepted, unambiguous face as an unidentified-face candidate. Ambiguous
        attempts are logged only. Closest-match display info goes to the track's
        TrackIdentity. The caller decides what a confirmed attempt means.

        allow_strong=False (the box's appearance just changed, attribution.appearance_settled):
        a single strong face does not confirm, only the FACE_CONSECUTIVE_MATCHES streak.

        Returns:
            FaceAttempt, or None when no usable face was found
        """
        recognizer = self.face_recognizer
        if recognizer is None:
            return None
        region = crop_face_region(frame, bbox, height_ratio=0.5)
        if region.size == 0:
            return None

        with profiler.measure("FaceRecognizer.detect"):
            faces = recognizer.detect_faces(region).faces
        identity = self.get_identity(track_id)
        identity.has_multiple_faces = len(faces) > 1
        if not faces:
            return None

        origin = (max(0, int(bbox[0])), max(0, int(bbox[1])))   # crop_face_region's top-left
        owned = skeleton_face(faces, keypoints, origin)
        face, ambiguous = owned if owned is not None else owner_face(faces, region.shape[1])
        embedding = face.embedding
        if embedding is None:
            with profiler.measure("FaceRecognizer.extract"):
                embedding = recognizer.extract_embedding(region, face)
            if embedding is None:
                return None

        with profiler.measure("FaceRecognizer.compare"):
            scores = self.score_face(embedding)
        best = scores.best_entry
        accepted = not ambiguous and scores.accepts(self.face_threshold, self.face_min_margin)
        # allow_strong=False: the box's appearance just changed (attribution.appearance_settled)
        strong = (accepted and allow_strong and self.face_strong_threshold > 0
                  and scores.accepts(self.face_strong_threshold, self.face_strong_min_margin))

        confirmed = False
        if accepted:
            if identity.candidate_person_id == best.person_id:
                identity.consecutive_matches += 1
            else:
                identity.candidate_person_id = best.person_id
                identity.consecutive_matches = 1
            confirmed = strong or identity.consecutive_matches >= FACE_CONSECUTIVE_MATCHES
            if confirmed:
                identity.candidate_person_id = None
                identity.consecutive_matches = 0
        if best:
            identity.face_info = (best.person_name, scores.top1)

        attempt = FaceAttempt(
            face=face, embedding=embedding, scores=scores, region=region,
            num_faces=len(faces), ambiguous=ambiguous, accepted=accepted, confirmed=confirmed,
            strong=strong,
        )
        if self._face_score_logger and best:
            self._face_score_logger.log(
                camera_id=camera_id,
                track_id=track_id,
                per_person=scores.per_person,
                top1_person_id=best.person_id,
                top1=scores.top1,
                top2=scores.top2,
                accepted=accepted,
                strong=strong,
                appearance_settled=allow_strong,
                face_height=face.height,
                face_width=face.width,
                det_confidence=face.confidence,
                threshold=self.face_threshold,
                min_margin=self.face_min_margin,
                num_faces=len(faces),
                ambiguous=ambiguous,
                embedding_norm=face.embedding_norm,
            )
        if not accepted and not ambiguous:
            self._submit_unidentified(attempt, track_id, camera_id)
        return attempt

    def _submit_unidentified(self, attempt: FaceAttempt, track_id: str, camera_id: Optional[str]) -> None:
        """Offer a face no gallery person accepted to the unidentified-face manager."""
        if not self._unidentified_face_manager or not camera_id:
            return
        face_crop, _ = crop_with_margin(attempt.region, attempt.face.bbox, margin_ratio=0.3)
        if face_crop.size == 0:
            return
        best = attempt.scores.best_entry
        self._unidentified_face_manager.submit(
            camera_id=camera_id,
            face_crop=face_crop,
            embedding=attempt.embedding,
            face_bbox=tuple(int(v) for v in attempt.face.bbox),   # the face, not the margin crop
            track_id=track_id,
            best_match_person_id=best.person_id if best else None,
            best_match_score=attempt.scores.top1 if attempt.scores.top1 > 0 else None,
            best_match_face_id=best.embedding_id if best else None,
            det_confidence=attempt.face.confidence,
            aligned_face=(align_face(attempt.region, attempt.face.landmarks)
                          if attempt.face.landmarks is not None else None),
            embedding_norm=attempt.face.embedding_norm,
        )

    def try_face_recognition(
        self,
        track_id: str,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
        local_track_id: int,
        isolated: bool,
        camera_id: Optional[str] = None,
        keypoints: Optional[np.ndarray] = None,
    ) -> bool:
        """Preview face recognition on a track (`attempt_face` + identity update).

        Args:
            track_id: Track identifier
            frame: Full frame
            bbox: Person box in the frame
            local_track_id: Local tracker ID (for Re-ID gallery)
            isolated: Box overlaps no other track (gallery is only fed then)
            camera_id: Camera ID for unidentified face capture

        Returns:
            True if the track just became face-identified
        """
        identity = self.get_identity(track_id)
        if identity.is_face_identified:
            return False

        attempt = self.attempt_face(frame, bbox, track_id, camera_id, keypoints=keypoints)
        if attempt is None or not attempt.confirmed:
            return False

        best = attempt.scores.best_entry
        previous_person_id, previous_method = identity.person_id, identity.method
        identity.person_id = best.person_id
        identity.person_name = best.person_name
        identity.confidence = attempt.scores.top1
        identity.method = IdMethod.FACE
        identity.face_info = None
        identity.identified_at = time.time()

        snapshot_path = self._save_snapshot(frame, "face_match", best.person_name)
        if self.repository:
            self.repository.create_event(
                camera_id=camera_id or "unknown",
                event_type="face_match",
                track_id=track_id,
                person_id=best.person_id,
                confidence=attempt.scores.top1,
                face_embedding_id=best.embedding_id,  # link to the matched gallery embedding
                snapshot_path=snapshot_path,
                extra_data={"threshold": self.face_threshold, "face_id": best.embedding_id},
            )
            if previous_method.is_appearance and previous_person_id is not None:
                self.repository.create_event(
                    camera_id=camera_id or "unknown",
                    event_type="id_upgraded",
                    track_id=track_id,
                    person_id=best.person_id,
                    extra_data={
                        "from": str(previous_method),
                        "to": str(IdMethod.FACE),
                        "previous_person_id": previous_person_id,
                        "new_person_id": best.person_id,
                    },
                )

        # Update Re-ID gallery with face-confirmed identity
        if self.reid_gallery_manager and isolated:
            self.reid_gallery_manager.update_track_embedding(
                local_track_id, crop_bbox(frame, bbox), best.person_name
            )
        return True
    # ==================== Re-ID Embedding Updates ====================

    def update_reid_embedding(
        self,
        track_id: str,
        local_track_id: int,
        crop: np.ndarray,
        isolated: bool,
    ):
        """Update Re-ID embedding for a face-identified, isolated track.

        Note: Grayscale/IR and multi-face checks are handled by ReIDGalleryManager.

        Args:
            track_id: Track identifier
            local_track_id: Local tracker ID
            crop: Person crop image
            isolated: Box overlaps no other track (attribution.py)
        """
        if not self.reid_gallery_manager or not isolated:
            return

        identity = self._identities.get(track_id)
        if identity and identity.is_face_identified and crop.size > 0:
            self.reid_gallery_manager.update_track_embedding(
                local_track_id, crop, identity.person_name
            )

    # ==================== Utility ====================

    def warmup(self):
        """Warm up ML models."""
        if self.face_recognizer:
            try:
                self.face_recognizer.warmup()
            except Exception as e:
                logger.error(f"Error during face recognizer warmup: {e}")

        if self.reid_extractor:
            try:
                self.reid_extractor.warmup()
            except Exception as e:
                logger.error(f"Error during Re-ID extractor warmup: {e}")

    def get_person_name(self, person_id: int) -> Optional[str]:
        """Get person name by ID (cached to avoid repeated DB queries)."""
        # Check cache first
        if not hasattr(self, '_person_name_cache'):
            self._person_name_cache: dict[int, str] = {}

        if person_id in self._person_name_cache:
            return self._person_name_cache[person_id]

        # Query database
        if self.repository:
            person = self.repository.get_person(person_id)
            if person:
                self._person_name_cache[person_id] = person.name
                return person.name
            return None

        # Search in gallery
        for entry in self._face_gallery:
            if entry.person_id == person_id:
                self._person_name_cache[person_id] = entry.person_name
                return entry.person_name
        return None
