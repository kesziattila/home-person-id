"""Identity linker - production wrapper around IdentificationManager.

This module provides additional production-specific functionality:
1. Track lifecycle management (register/unregister)
2. Workflow orchestration with timing controls
3. Identity transfer for cross-camera handover
4. Database integration for persistence

Uses IdentificationManager for core face recognition and Re-ID logic.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.config import Config, FaceRecognitionConfig, ReIDConfig
from src.database.repository import Repository
from src.recognition.face_recognizer import FaceRecognizer
from src.recognition.identification_manager import IdentificationManager
from src.recognition.reid_extractor import ReIDExtractor, EmbeddingGallery
from src.utils.profiler import profiler

logger = logging.getLogger(__name__)


@dataclass
class IdentificationResult:
    """Result of person identification."""

    person_id: Optional[int] = None
    person_name: Optional[str] = None
    confidence: float = 0.0
    method: str = "none"  # 'face', 'reid', 'none'
    is_confirmed: bool = False

    # For visualization compatibility with TrackIdentity
    reid_score: float = -1.0
    face_info: Optional[tuple[str, float]] = None
    is_reid_identified: bool = False


@dataclass
class TrackIdentityState:
    """Identity state for a global track.

    Extended state for production use with consecutive match tracking.
    """

    global_track_id: str
    person_id: Optional[int] = None
    identification_confidence: float = 0.0
    identified_by: str = "none"  # 'face', 'reid', 'transfer'

    # Face embedding if captured
    face_embedding: Optional[np.ndarray] = None

    # Re-ID gallery for stable matching (per-track)
    reid_gallery: EmbeddingGallery = field(default_factory=EmbeddingGallery)

    # Consecutive match tracking for stability
    candidate_person_id: Optional[int] = None
    consecutive_matches: int = 0

    # Timestamps
    first_seen: float = field(default_factory=time.time)
    last_face_check: float = 0.0
    identified_at: Optional[float] = None

    @property
    def is_identified(self) -> bool:
        """Check if track has been linked to a known person."""
        return self.person_id is not None

    def update_reid_gallery(
        self, embedding: np.ndarray, quality: float, max_size: int = 10
    ):
        """Update Re-ID embedding gallery."""
        self.reid_gallery.max_size = max_size
        self.reid_gallery.add(embedding, quality, time.time())

    def confirm_identity(self, person_id: int, confidence: float, method: str):
        """Confirm identity assignment."""
        self.person_id = person_id
        self.identification_confidence = confidence
        self.identified_by = method
        self.identified_at = time.time()
        self.candidate_person_id = None
        self.consecutive_matches = 0
        logger.info(
            f"Track {self.global_track_id} identified as person {person_id} "
            f"via {method} (conf={confidence:.2f})"
        )


class IdentityLinker:
    """Production identity linker with workflow orchestration.

    Wraps IdentificationManager and adds:
    - Track lifecycle management
    - Timing-based face check intervals
    - Consecutive match stability tracking
    - Database persistence
    - Identity transfer for handovers
    """

    def __init__(
        self,
        face_config: FaceRecognitionConfig,
        reid_config: ReIDConfig,
        repository: Repository,
        enable_debug_images: bool = True,
        face_recognizer: Optional[FaceRecognizer] = None,
        reid_extractor: Optional[ReIDExtractor] = None,
    ):
        """Initialize identity linker.

        Args:
            face_config: Face recognition configuration
            reid_config: Re-ID configuration
            repository: Database repository
            enable_debug_images: Whether to save debug images
            face_recognizer: Optional pre-initialized face recognizer (for testing)
            reid_extractor: Optional pre-initialized Re-ID extractor (for testing)
        """
        self.face_config = face_config
        self.reid_config = reid_config
        self.repository = repository

        # Create a minimal config for IdentificationManager
        self._config = self._create_config(face_config, reid_config)

        # Core identification manager (handles face recognition and Re-ID)
        self._id_manager = IdentificationManager(
            config=self._config,
            repository=repository,
            enable_debug_images=enable_debug_images,
            face_recognizer=face_recognizer,
            reid_extractor=reid_extractor,
        )

        # Track identity states (production-specific, with consecutive match tracking)
        self._track_states: dict[str, TrackIdentityState] = {}

    def _create_config(self, face_config: FaceRecognitionConfig, reid_config: ReIDConfig) -> Config:
        """Create a Config object from individual configs."""
        # Import here to avoid circular imports
        from src.config import Config
        config = Config()
        config.face_recognition = face_config
        config.reid = reid_config
        return config

    # ==================== Delegate to IdentificationManager ====================

    @property
    def face_recognizer(self):
        """Get face recognizer."""
        return self._id_manager.face_recognizer

    @property
    def reid_extractor(self):
        """Get Re-ID extractor."""
        return self._id_manager.reid_extractor

    @property
    def reid_gallery_manager(self):
        """Get shared Re-ID gallery manager."""
        return self._id_manager.reid_gallery_manager

    # ==================== Track Lifecycle ====================

    def register_track(self, global_track_id: str) -> TrackIdentityState:
        """Register a new track for identity tracking.

        Args:
            global_track_id: Global track ID

        Returns:
            TrackIdentityState for the track
        """
        if global_track_id not in self._track_states:
            self._track_states[global_track_id] = TrackIdentityState(
                global_track_id=global_track_id
            )
            logger.debug(f"Registered track {global_track_id} for identity tracking")

        return self._track_states[global_track_id]

    def unregister_track(self, global_track_id: str, was_face_identified: bool = False):
        """Unregister a track from identity tracking.

        If the track was face-identified, its embeddings are stored in the
        shared gallery for future cross-camera Re-ID.

        Args:
            global_track_id: Track ID to unregister
            was_face_identified: Whether track was identified by face recognition
        """
        if global_track_id in self._track_states:
            state = self._track_states[global_track_id]

            # Store embeddings in shared gallery if face-identified
            if was_face_identified and state.is_identified and self.reid_gallery_manager:
                track_id_num = hash(global_track_id) % (10**9)
                self.reid_gallery_manager.on_track_lost(track_id_num, was_face_identified=True)

            del self._track_states[global_track_id]
            logger.debug(f"Unregistered track {global_track_id}")

    def get_track_state(self, global_track_id: str) -> Optional[TrackIdentityState]:
        """Get identity state for a track."""
        return self._track_states.get(global_track_id)

    def get_identity(self, global_track_id: str) -> Optional[IdentificationResult]:
        """Get current identification status for a track.

        This method is used by the system to get identity information for
        visualization and logging.

        Args:
            global_track_id: Global track ID

        Returns:
            IdentificationResult if track exists, None otherwise
        """
        state = self._track_states.get(global_track_id)
        if not state:
            return None

        name = self._get_person_name(state.person_id) if state.person_id else None
        
        # Get reid score from shared gallery if not identified or if identified via reid
        reid_score = -1.0
        # For preview we might want to see the best match score even if not confirmed
        # But we don't have the crop here. 
        # TrackIdentityState doesn't store the latest score, but TrackIdentity does.
        # However, for now we just want to fix the AttributeError.
        
        return IdentificationResult(
            person_id=state.person_id,
            person_name=name,
            confidence=state.identification_confidence,
            method=state.identified_by,
            is_confirmed=state.is_identified,
            is_reid_identified=state.identified_by == "reid" or "transfer" in state.identified_by,
            face_info=(name, state.identification_confidence) if "face" in state.identified_by else None
        )

    # ==================== Main Processing Workflow ====================

    def process_track(
        self,
        global_track_id: str,
        frame: np.ndarray,
        person_crop: np.ndarray,
        bbox: tuple[float, float, float, float],
        force_face_check: bool = False,
        num_persons_in_frame: int = 1,
        precomputed_reid: Optional[tuple[np.ndarray, float]] = None,
    ) -> IdentificationResult:
        """Process a track for identification.

        Orchestrates face recognition and Re-ID with timing controls.

        Args:
            global_track_id: Global track ID
            frame: Full frame image
            person_crop: Cropped person image
            bbox: Bounding box of person in frame
            force_face_check: Force face recognition even if recently checked
            num_persons_in_frame: Number of persons detected in frame
            precomputed_reid: Optional pre-computed Re-ID (embedding, quality)

        Returns:
            IdentificationResult with identification status
        """
        state = self.register_track(global_track_id)
        current_time = time.time()

        # If already identified, just update Re-ID gallery
        if state.is_identified:
            self._update_reid_gallery(state, person_crop, num_persons_in_frame, precomputed_reid)
            name = self._get_person_name(state.person_id)
            return IdentificationResult(
                person_id=state.person_id,
                person_name=name,
                confidence=state.identification_confidence,
                method=state.identified_by,
                is_confirmed=True,
                is_reid_identified=state.identified_by == "reid",
                face_info=(name, state.identification_confidence) if state.identified_by == "face" else None
            )

        # Check if we should run face recognition
        time_since_face_check = current_time - state.last_face_check
        
        # Determine check interval (longer if already identified to save CPU)
        check_interval = self.face_config.detection_interval
        if state.is_identified:
            check_interval = self.face_config.reid_confirmation_interval
        
        should_check_face = (
            force_face_check
            or time_since_face_check > (check_interval / 2.0)
        )

        result = IdentificationResult()

        if should_check_face and self.face_config.enabled:
            state.last_face_check = current_time
            face_result = self._try_face_identification(state, frame, bbox, person_crop, num_persons_in_frame)

            if face_result.is_confirmed:
                return face_result
            elif face_result.person_id is not None:
                result = face_result

        # Always update Re-ID gallery for cross-camera matching
        self._update_reid_gallery(state, person_crop, num_persons_in_frame, precomputed_reid)

        return result

    def _get_person_name(self, person_id: int) -> Optional[str]:
        """Get person name from ID."""
        return self._id_manager.get_person_name(person_id)

    def _try_face_identification(
        self,
        state: TrackIdentityState,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
        person_crop: np.ndarray,
        num_persons_in_frame: int,
    ) -> IdentificationResult:
        """Try to identify track using face recognition.

        Uses IdentificationManager for core logic, adds consecutive match tracking.
        """
        gallery = self._id_manager.face_gallery
        if not gallery or not self._id_manager.face_recognizer:
            return IdentificationResult()

        # Crop face region from person bbox (upper portion)
        x1, y1, x2, y2 = map(int, bbox)
        h = y2 - y1
        face_region = frame[y1 : y1 + int(h * 0.5), x1:x2]

        if face_region.size == 0:
            return IdentificationResult()

        # Detect faces
        face_recognizer = self._id_manager.face_recognizer
        with profiler.measure("FaceRecognizer.detect"):
            detection_result = face_recognizer.detect_faces(face_region)

        if not detection_result.faces:
            return IdentificationResult()

        # Get best face
        best_face = max(detection_result.faces, key=lambda f: f.area * f.confidence)

        if best_face.height < self.face_config.min_face_size:
            return IdentificationResult()

        # Get embedding
        if best_face.embedding is None:
            with profiler.measure("FaceRecognizer.extract"):
                embedding = face_recognizer.extract_embedding(face_region, best_face)
            if embedding is None:
                return IdentificationResult()
        else:
            embedding = best_face.embedding

        state.face_embedding = embedding

        # Match against gallery
        best_match_id = None
        best_match_name = None
        best_score = 0.0

        with profiler.measure("FaceRecognizer.compare"):
            for person_id, name, gallery_embedding in gallery:
                similarity = face_recognizer.compare_embeddings(embedding, gallery_embedding)
                if similarity > self.face_config.similarity_threshold and similarity > best_score:
                    best_match_id = person_id
                    best_match_name = name
                    best_score = similarity

        if best_match_id is None:
            return IdentificationResult()

        # Consecutive match tracking for stability
        if state.candidate_person_id == best_match_id:
            state.consecutive_matches += 1
        else:
            state.candidate_person_id = best_match_id
            state.consecutive_matches = 1

        min_consecutive = 2
        if state.consecutive_matches >= min_consecutive:
            state.confirm_identity(best_match_id, best_score, "face")

            # Update database
            self.repository.update_track(
                state.global_track_id,
                person_id=best_match_id,
                face_embedding=embedding,
            )

            # Update shared Re-ID gallery
            if self.reid_gallery_manager and best_match_name:
                track_id_num = hash(state.global_track_id) % (10**9)
                self.reid_gallery_manager.update_track_embedding(
                    track_id_num, person_crop, best_match_name, num_persons_in_frame
                )

            return IdentificationResult(
                person_id=best_match_id,
                person_name=best_match_name,
                confidence=best_score,
                method="face",
                is_confirmed=True,
                face_info=(best_match_name, best_score)
            )

        return IdentificationResult(
            person_id=best_match_id,
            person_name=best_match_name,
            confidence=best_score,
            method="face",
            is_confirmed=False,
            face_info=(best_match_name, best_score)
        )

    def _update_reid_gallery(
        self,
        state: TrackIdentityState,
        crop: np.ndarray,
        num_persons_in_frame: int = 1,
        precomputed_reid: Optional[tuple[np.ndarray, float]] = None,
    ):
        """Update Re-ID embedding gallery for a track.

        Updates both the per-track gallery and shared gallery manager.
        """
        if num_persons_in_frame > 1:
            return

        if precomputed_reid:
            embedding, quality = precomputed_reid
        else:
            reid_extractor = self._id_manager.reid_extractor
            if not reid_extractor:
                return

            with profiler.measure("ReIDExtractor.extract"):
                embedding, quality = reid_extractor.extract(crop, return_quality=True)

        if quality >= self.reid_config.min_visibility:
            # Update per-track gallery
            state.update_reid_gallery(
                embedding, quality, max_size=self.reid_config.gallery_size
            )

            # Update shared gallery if identified
            if state.is_identified and self.reid_gallery_manager:
                person_name = self._get_person_name(state.person_id)
                if person_name:
                    track_id_num = hash(state.global_track_id) % (10**9)
                    self.reid_gallery_manager.update_track_embedding(
                        track_id_num, crop, person_name, num_persons_in_frame
                    )

    # ==================== Cross-Camera Re-ID ====================

    def match_reid_cross_camera(
        self,
        new_track_id: str,
        new_crop: np.ndarray,
        candidate_track_ids: list[str],
        num_persons_in_frame: int = 1,
        precomputed_reid: Optional[tuple[np.ndarray, float]] = None,
    ) -> Optional[tuple[str, float, Optional[str]]]:
        """Match a new track against existing tracks using Re-ID.

        Args:
            new_track_id: ID of the new track
            new_crop: Person crop from new track
            candidate_track_ids: IDs of tracks to match against
            num_persons_in_frame: Number of persons in frame
            precomputed_reid: Optional pre-computed Re-ID (embedding, quality)

        Returns:
            Tuple of (matched_track_id, similarity, person_name) or None
        """
        if num_persons_in_frame > 1:
            return None

        if precomputed_reid:
            new_embedding, quality = precomputed_reid
        else:
            reid_extractor = self._id_manager.reid_extractor
            if not reid_extractor or new_crop.size == 0:
                return None

            with profiler.measure("ReIDExtractor.extract"):
                new_embedding, quality = reid_extractor.extract(new_crop, return_quality=True)

        if quality < self.reid_config.min_visibility:
            return None

        best_match_id = None
        best_score = 0.0
        best_person_name = None

        # Match against active track states
        with profiler.measure("ReIDExtractor.compare"):
            for track_id in candidate_track_ids:
                state = self._track_states.get(track_id)
                if state is None or len(state.reid_gallery) == 0:
                    continue

                similarity = state.reid_gallery.match(new_embedding)
                if similarity > self.reid_config.similarity_threshold and similarity > best_score:
                    best_match_id = track_id
                    best_score = similarity
                    if state.is_identified:
                        best_person_name = self._get_person_name(state.person_id)

        # Also try shared gallery
        if self.reid_gallery_manager:
            with profiler.measure("ReIDGallery.match"):
                match_result = self.reid_gallery_manager.match_new_track(new_crop, num_persons_in_frame)
            if match_result.matched and match_result.score > best_score:
                logger.info(
                    f"Re-ID gallery match: {new_track_id} -> {match_result.person_name} "
                    f"(sim={match_result.score:.2f})"
                )
                return (None, match_result.score, match_result.person_name)

        if best_match_id is not None:
            logger.debug(f"Re-ID match: {new_track_id} -> {best_match_id} (sim={best_score:.2f})")
            return (best_match_id, best_score, best_person_name)

        return None

    # ==================== Identity Transfer ====================

    def transfer_identity(self, from_track_id: str, to_track_id: str):
        """Transfer identity from one track to another.

        Used when tracks are matched across cameras.
        """
        if from_track_id == to_track_id:
            return

        from_state = self._track_states.get(from_track_id)
        to_state = self.register_track(to_track_id)

        if from_state is None:
            return

        if from_state.is_identified:
            to_state.person_id = from_state.person_id
            to_state.identification_confidence = from_state.identification_confidence
            to_state.identified_by = f"transfer_from_{from_track_id}"
            to_state.identified_at = time.time()

        if from_state.face_embedding is not None:
            to_state.face_embedding = from_state.face_embedding

        for emb in from_state.reid_gallery.embeddings:
            to_state.reid_gallery.embeddings.append(emb)

        logger.info(
            f"Transferred identity from {from_track_id} to {to_track_id} "
            f"(person_id={to_state.person_id})"
        )

    # ==================== Utility ====================

    def get_identified_tracks(self) -> list[tuple[str, int, str]]:
        """Get all identified tracks."""
        return [
            (track_id, state.person_id, state.identified_by)
            for track_id, state in self._track_states.items()
            if state.is_identified
        ]

    def cleanup_old_tracks(self, max_age_seconds: float = 3600):
        """Remove old track states to prevent memory growth."""
        current_time = time.time()
        to_remove = [
            track_id for track_id, state in self._track_states.items()
            if current_time - state.first_seen > max_age_seconds
        ]

        for track_id in to_remove:
            del self._track_states[track_id]

        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} old track states")

    def warmup(self):
        """Warm up models."""
        self._id_manager.warmup()
