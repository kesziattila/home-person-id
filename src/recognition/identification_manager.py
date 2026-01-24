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
from typing import Optional, Callable

import numpy as np

from src.config import Config, FaceRecognitionConfig, ReIDConfig
from src.recognition.face_recognizer import FaceRecognizer
from src.recognition.reid_extractor import ReIDExtractor, is_grayscale_image
from src.recognition.reid_gallery import ReIDGalleryManager, DebugImageSaver

logger = logging.getLogger(__name__)


@dataclass
class TrackIdentity:
    """Identity information for a track."""

    track_id: str = ""
    person_id: Optional[int] = None
    person_name: Optional[str] = None
    confidence: float = 0.0

    # Identification method
    is_face_identified: bool = False
    is_reid_identified: bool = False

    # Scores for display
    reid_score: float = -1.0
    face_info: Optional[tuple[str, float]] = None  # (closest_name, confidence)

    # Multi-face detection flag
    has_multiple_faces: bool = False

    # Timestamps
    first_seen: float = field(default_factory=time.time)
    last_face_check: float = 0.0
    identified_at: Optional[float] = None

    # Stability tracking (for consecutive match requirement)
    candidate_person_id: Optional[int] = None
    candidate_person_name: Optional[str] = None
    consecutive_matches: int = 0

    @property
    def is_identified(self) -> bool:
        """Check if track has been identified."""
        return self.person_name is not None


class IdentificationManager:
    """Unified manager for face recognition and Re-ID.

    Works in two modes:
    1. Preview mode: Pass face_gallery directly, no database
    2. Production mode: Pass repository, gallery loaded from database with TTL

    Example (preview mode):
        manager = IdentificationManager(config, face_gallery=gallery_list)

    Example (production mode):
        manager = IdentificationManager(config, repository=repo)
    """

    def __init__(
        self,
        config: Config,
        face_gallery: Optional[list[tuple[int, str, np.ndarray]]] = None,
        repository: Optional[object] = None,  # Repository type
        enable_debug_images: bool = False,
    ):
        """Initialize identification manager.

        Args:
            config: Application configuration
            face_gallery: Pre-loaded face gallery (person_id, name, embedding) - for preview
            repository: Database repository - for production (loads gallery automatically)
            enable_debug_images: Whether to save debug images
        """
        self.config = config
        self.face_config = config.face_recognition
        self.reid_config = config.reid
        self.repository = repository

        # Face gallery - either pre-loaded or from database
        self._face_gallery = face_gallery or []
        self._gallery_loaded_at = 0.0
        self._gallery_ttl = 60.0  # Reload from DB every 60 seconds

        # Lazy-loaded recognizers
        self._face_recognizer: Optional[FaceRecognizer] = None
        self._reid_extractor: Optional[ReIDExtractor] = None
        self._reid_gallery_manager: Optional[ReIDGalleryManager] = None

        # Debug image saver
        self._debug_saver: Optional[DebugImageSaver] = None
        if enable_debug_images:
            self._debug_saver = DebugImageSaver()

        # Track identity states
        self._identities: dict[str, TrackIdentity] = {}

        # Configuration shortcuts
        self.face_threshold = config.face_recognition.similarity_threshold
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
                self._reid_gallery_manager = ReIDGalleryManager(
                    reid_extractor=extractor,
                    similarity_threshold=self.reid_config.similarity_threshold,
                    max_reappear_time_sec=self.reid_config.max_reappear_time_sec,
                    max_embeddings_per_person=self.reid_config.gallery_size,
                    crop_cache_path=self.reid_config.crop_cache_path,
                    face_recognizer=self.face_recognizer,
                    debug_saver=self._debug_saver,
                )
        return self._reid_gallery_manager

    @property
    def reid_enabled(self) -> bool:
        """Check if Re-ID is enabled."""
        return self.reid_config.enabled

    @property
    def face_gallery(self) -> list[tuple[int, str, np.ndarray]]:
        """Get face gallery, loading from database if needed."""
        if self.repository:
            self._load_face_gallery_from_db()
        return self._face_gallery

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
                self._face_gallery.append((person_id, person.name, embedding))

        self._gallery_loaded_at = current_time
        logger.debug(f"Loaded {len(self._face_gallery)} face embeddings")

    def warmup(self):
        """Warm up models for faster first inference."""
        if self.reid_extractor:
            self.reid_extractor.warmup()

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

    def try_reid_match(
        self,
        track_id: str,
        crop: np.ndarray,
        num_persons: int
    ) -> Optional[np.ndarray]:
        """Try to match a track against Re-ID gallery.

        Args:
            track_id: Track identifier
            crop: Person crop image
            num_persons: Number of persons in frame

        Returns:
            Gallery crop if matched, None otherwise
        """
        if not self.reid_gallery_manager or crop.size == 0:
            return None

        # Skip grayscale/IR images
        if is_grayscale_image(crop):
            return None

        match_result = self.reid_gallery_manager.match_new_track(crop, num_persons)
        identity = self.get_identity(track_id)

        if match_result.score >= 0:
            identity.reid_score = match_result.score

        if match_result.matched:
            identity.person_name = match_result.person_name
            identity.confidence = match_result.score
            identity.is_reid_identified = True
            identity.is_face_identified = False
            identity.identified_at = time.time()
            return match_result.gallery_crop

        return None

    def match_reid_cross_camera(
        self,
        new_track_id: str,
        new_crop: np.ndarray,
        candidate_track_ids: list[str],
        num_persons_in_frame: int = 1,
    ) -> Optional[tuple[str, float, Optional[str]]]:
        """Match a new track against existing tracks using Re-ID.

        Used for cross-camera matching when a person appears on a new camera.

        Args:
            new_track_id: ID of the new track
            new_crop: Person crop from new track
            candidate_track_ids: IDs of tracks to match against
            num_persons_in_frame: Number of persons in frame

        Returns:
            Tuple of (matched_track_id, similarity, person_name) or None
        """
        if num_persons_in_frame > 1:
            return None

        if not self.reid_extractor or new_crop.size == 0:
            return None

        # Skip grayscale/IR images
        if is_grayscale_image(new_crop):
            return None

        new_embedding, quality = self.reid_extractor.extract(new_crop, return_quality=True)
        if quality < self.reid_config.min_visibility:
            return None

        best_match = None
        best_score = self.reid_config.similarity_threshold

        # Match against active track states
        for track_id in candidate_track_ids:
            if track_id == new_track_id:
                continue

            state = self._identities.get(track_id)
            if not state:
                continue

            # Try matching against shared gallery if identified
            if state.is_identified and self.reid_gallery_manager:
                # Get embeddings from gallery manager
                pass  # Gallery manager handles this internally

        # Also try shared gallery for lost face-identified persons
        if self.reid_gallery_manager:
            match_result = self.reid_gallery_manager.match_new_track(new_crop, num_persons_in_frame)
            if match_result.matched and match_result.score > best_score:
                return (new_track_id, match_result.score, match_result.person_name)

        return best_match

    # ==================== Face Recognition ====================

    def try_face_recognition(
        self,
        track_id: str,
        person_crop: np.ndarray,
        local_track_id: int,
        num_persons: int,
    ) -> bool:
        """Try face recognition on a track.

        Args:
            track_id: Track identifier
            person_crop: Person crop image
            local_track_id: Local tracker ID (for Re-ID gallery)
            num_persons: Number of persons in frame

        Returns:
            True if face identified
        """
        if not self.face_recognizer:
            return False

        gallery = self.face_gallery
        if not gallery:
            return False

        if person_crop.size == 0:
            return False

        identity = self.get_identity(track_id)

        # Skip if already face-identified
        if identity.is_face_identified:
            return False

        face_result = self.face_recognizer.detect_faces(person_crop)
        if not face_result.faces:
            identity.has_multiple_faces = False
            return False

        # Check for multiple faces (indicates 2+ people in crop)
        if len(face_result.faces) > 1:
            identity.has_multiple_faces = True
            return False

        identity.has_multiple_faces = False
        face = face_result.faces[0]
        if face.embedding is None:
            return False

        # Find best match
        best_score = 0.0
        best_name = None
        best_person_id = None

        for pid, name, emb in gallery:
            score = self.face_recognizer.compare_embeddings(face.embedding, emb)
            if score > best_score:
                best_score = score
                best_name = name
                best_person_id = pid

        # Store face info even if below threshold
        if best_name:
            identity.face_info = (best_name, best_score)

        # Check if above threshold
        if best_score >= self.face_threshold:
            identity.person_id = best_person_id
            identity.person_name = best_name
            identity.confidence = best_score
            identity.is_face_identified = True
            identity.is_reid_identified = False
            identity.face_info = None
            identity.identified_at = time.time()

            # Update Re-ID gallery with face-confirmed identity
            if self.reid_gallery_manager:
                self.reid_gallery_manager.update_track_embedding(
                    local_track_id, person_crop, best_name, num_persons
                )
            return True

        return False

    # ==================== Re-ID Embedding Updates ====================

    def update_reid_embedding(
        self,
        track_id: str,
        local_track_id: int,
        crop: np.ndarray,
        num_persons: int,
    ):
        """Update Re-ID embedding for a face-identified track.

        Note: Grayscale/IR and multi-face checks are handled by ReIDGalleryManager.

        Args:
            track_id: Track identifier
            local_track_id: Local tracker ID
            crop: Person crop image
            num_persons: Number of persons in frame
        """
        if not self.reid_gallery_manager:
            return

        identity = self._identities.get(track_id)
        if identity and identity.is_face_identified and crop.size > 0:
            self.reid_gallery_manager.update_track_embedding(
                local_track_id, crop, identity.person_name, num_persons
            )

    # ==================== Utility ====================

    def get_person_name(self, person_id: int) -> Optional[str]:
        """Get person name by ID."""
        if self.repository:
            person = self.repository.get_person(person_id)
            return person.name if person else None

        # Search in gallery
        for pid, name, _ in self._face_gallery:
            if pid == person_id:
                return name
        return None
