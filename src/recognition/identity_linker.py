"""Identity linker - connects face recognition and Re-ID to tracks.

This module handles:
1. Linking face recognition results to active tracks
2. Using Re-ID to maintain identity when face isn't visible
3. Stable matching with gallery-based comparison and temporal smoothing
4. Cross-camera Re-ID gallery management (shared with CLI preview)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.config import FaceRecognitionConfig, ReIDConfig
from src.database.repository import Repository
from src.recognition.face_recognizer import Face, FaceRecognizer
from src.recognition.reid_extractor import EmbeddingGallery, ReIDExtractor, cosine_similarity
from src.recognition.reid_gallery import ReIDGalleryManager, DebugImageSaver
from src.tracking.track import GlobalTrack, LocalTrack

logger = logging.getLogger(__name__)


@dataclass
class IdentificationResult:
    """Result of person identification."""

    person_id: Optional[int] = None
    person_name: Optional[str] = None
    confidence: float = 0.0
    method: str = "none"  # 'face', 'reid', 'none'
    is_confirmed: bool = False


@dataclass
class TrackIdentityState:
    """Identity state for a global track.

    Tracks identification status and manages Re-ID gallery for stable matching.
    """

    global_track_id: str
    person_id: Optional[int] = None
    identification_confidence: float = 0.0
    identified_by: str = "none"  # 'face', 'reid'

    # Face embedding if captured
    face_embedding: Optional[np.ndarray] = None

    # Re-ID gallery for stable matching
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
    """Links face recognition and Re-ID results to tracks.

    Workflow:
    1. When a track is created, it has no identity
    2. Face recognition runs periodically on tracks
    3. If face matches known person -> identity is assigned
    4. Re-ID maintains that identity when face isn't visible
    5. For cross-camera matching, Re-ID is used to transfer identity

    Uses shared ReIDGalleryManager for cross-camera Re-ID when available.
    """

    def __init__(
        self,
        face_config: FaceRecognitionConfig,
        reid_config: ReIDConfig,
        repository: Repository,
        enable_debug_images: bool = True,
    ):
        """Initialize identity linker.

        Args:
            face_config: Face recognition configuration
            reid_config: Re-ID configuration
            repository: Database repository
            enable_debug_images: Whether to save debug images
        """
        self.face_config = face_config
        self.reid_config = reid_config
        self.repository = repository

        # Lazy-loaded recognizers
        self._face_recognizer: Optional[FaceRecognizer] = None
        self._reid_extractor: Optional[ReIDExtractor] = None

        # Debug image saver (shared with CLI preview)
        self._debug_saver: Optional[DebugImageSaver] = None
        if enable_debug_images:
            self._debug_saver = DebugImageSaver()

        # Shared Re-ID gallery manager (created after reid_extractor is initialized)
        self._reid_gallery_manager: Optional[ReIDGalleryManager] = None

        # Track identity states
        self._track_states: dict[str, TrackIdentityState] = {}

        # Cache of known person embeddings
        self._face_gallery: list[tuple[int, str, np.ndarray]] = []  # (person_id, name, embedding)
        self._gallery_loaded_at: float = 0.0
        self._gallery_ttl: float = 60.0  # Reload every 60 seconds

    @property
    def face_recognizer(self) -> FaceRecognizer:
        """Get face recognizer (lazy loading)."""
        if self._face_recognizer is None:
            self._face_recognizer = FaceRecognizer(self.face_config)
        return self._face_recognizer

    @property
    def reid_extractor(self) -> ReIDExtractor:
        """Get Re-ID extractor (lazy loading)."""
        if self._reid_extractor is None:
            self._reid_extractor = ReIDExtractor(self.reid_config)
            # Initialize shared gallery manager
            self._reid_gallery_manager = ReIDGalleryManager(
                reid_extractor=self._reid_extractor,
                similarity_threshold=self.reid_config.similarity_threshold,
                max_reappear_time_sec=self.reid_config.max_reappear_time_sec,
                max_embeddings_per_person=self.reid_config.gallery_size,
                debug_saver=self._debug_saver,
            )
        return self._reid_extractor

    @property
    def reid_gallery_manager(self) -> Optional[ReIDGalleryManager]:
        """Get shared Re-ID gallery manager."""
        # Ensure reid_extractor is initialized first
        _ = self.reid_extractor
        return self._reid_gallery_manager

    def _load_face_gallery(self):
        """Load known person face embeddings from database."""
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
        logger.debug(f"Loaded {len(self._face_gallery)} face embeddings for {len(set(p[0] for p in self._face_gallery))} persons")

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
            if was_face_identified and state.is_identified and self._reid_gallery_manager:
                track_id_num = hash(global_track_id) % (10**9)
                self._reid_gallery_manager.on_track_lost(track_id_num, was_face_identified=True)

            del self._track_states[global_track_id]
            logger.debug(f"Unregistered track {global_track_id}")

    def get_track_state(self, global_track_id: str) -> Optional[TrackIdentityState]:
        """Get identity state for a track."""
        return self._track_states.get(global_track_id)

    def process_track(
        self,
        global_track_id: str,
        frame: np.ndarray,
        person_crop: np.ndarray,
        bbox: tuple[float, float, float, float],
        force_face_check: bool = False,
        num_persons_in_frame: int = 1,
    ) -> IdentificationResult:
        """Process a track for identification.

        Args:
            global_track_id: Global track ID
            frame: Full frame image
            person_crop: Cropped person image
            bbox: Bounding box of person in frame
            force_face_check: Force face recognition even if recently checked
            num_persons_in_frame: Number of persons detected in frame (for Re-ID safety)

        Returns:
            IdentificationResult with identification status
        """
        state = self.register_track(global_track_id)
        current_time = time.time()

        # If already identified, just update Re-ID gallery (skip if multiple persons)
        if state.is_identified:
            self._update_reid_gallery(state, person_crop, num_persons_in_frame)
            return IdentificationResult(
                person_id=state.person_id,
                person_name=self._get_person_name(state.person_id),
                confidence=state.identification_confidence,
                method=state.identified_by,
                is_confirmed=True,
            )

        # Check if we should run face recognition
        time_since_face_check = current_time - state.last_face_check
        should_check_face = (
            force_face_check
            or time_since_face_check > (self.face_config.detection_interval / 5.0)  # ~2 sec at 5 FPS
        )

        result = IdentificationResult()

        if should_check_face and self.face_config.enabled:
            state.last_face_check = current_time
            face_result = self._try_face_identification(state, frame, bbox)

            if face_result.is_confirmed:
                return face_result
            elif face_result.person_id is not None:
                # Tentative match - track for stability
                result = face_result

        # Always update Re-ID gallery for cross-camera matching (skip if multiple persons)
        self._update_reid_gallery(state, person_crop, num_persons_in_frame)

        return result

    def _get_person_name(self, person_id: int) -> Optional[str]:
        """Get person name from ID."""
        self._load_face_gallery()
        for pid, name, _ in self._face_gallery:
            if pid == person_id:
                return name
        return None

    def _try_face_identification(
        self,
        state: TrackIdentityState,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> IdentificationResult:
        """Try to identify track using face recognition.

        Args:
            state: Track identity state
            frame: Full frame image
            bbox: Person bounding box

        Returns:
            IdentificationResult
        """
        self._load_face_gallery()

        if not self._face_gallery:
            return IdentificationResult()

        # Crop face region from person bbox (upper portion)
        x1, y1, x2, y2 = map(int, bbox)
        h = y2 - y1
        face_region = frame[y1 : y1 + int(h * 0.5), x1:x2]  # Upper half

        if face_region.size == 0:
            return IdentificationResult()

        # Detect faces
        detection_result = self.face_recognizer.detect_faces(face_region)

        if not detection_result.faces:
            return IdentificationResult()

        # Get best face (largest, highest confidence)
        best_face = max(detection_result.faces, key=lambda f: f.area * f.confidence)

        # Check face quality
        if best_face.height < self.face_config.min_face_size:
            return IdentificationResult()

        # Get face embedding
        if best_face.embedding is None:
            embedding = self.face_recognizer.extract_embedding(face_region, best_face)
            if embedding is None:
                return IdentificationResult()
        else:
            embedding = best_face.embedding

        # Save face embedding to state
        state.face_embedding = embedding

        # Match against gallery
        best_match_id = None
        best_match_name = None
        best_score = 0.0

        for person_id, name, gallery_embedding in self._face_gallery:
            similarity = self.face_recognizer.compare_embeddings(embedding, gallery_embedding)

            if similarity > self.face_config.similarity_threshold and similarity > best_score:
                best_match_id = person_id
                best_match_name = name
                best_score = similarity

        if best_match_id is None:
            return IdentificationResult()

        # Check for stable match (consecutive matches)
        if state.candidate_person_id == best_match_id:
            state.consecutive_matches += 1
        else:
            state.candidate_person_id = best_match_id
            state.consecutive_matches = 1

        # Require multiple consecutive matches for confirmation
        min_consecutive = 2  # Lower for face since it's more reliable
        if state.consecutive_matches >= min_consecutive:
            state.confirm_identity(best_match_id, best_score, "face")

            # Update database
            self.repository.update_track(
                state.global_track_id,
                person_id=best_match_id,
                face_embedding=embedding,
            )

            return IdentificationResult(
                person_id=best_match_id,
                person_name=best_match_name,
                confidence=best_score,
                method="face",
                is_confirmed=True,
            )

        return IdentificationResult(
            person_id=best_match_id,
            person_name=best_match_name,
            confidence=best_score,
            method="face",
            is_confirmed=False,  # Still tentative
        )

    def _update_reid_gallery(
        self,
        state: TrackIdentityState,
        crop: np.ndarray,
        num_persons_in_frame: int = 1,
    ):
        """Update Re-ID embedding gallery for a track.

        Args:
            state: Track identity state
            crop: Person crop image
            num_persons_in_frame: Number of persons in frame (skip if >1)
        """
        # Skip if multiple persons to avoid confusion
        if num_persons_in_frame > 1:
            return

        embedding, quality = self.reid_extractor.extract(crop, return_quality=True)

        if quality >= self.reid_config.min_visibility:
            state.update_reid_gallery(
                embedding, quality, max_size=self.reid_config.gallery_size
            )

            # Also update the shared gallery manager if track is identified
            if state.is_identified and self._reid_gallery_manager:
                person_name = self._get_person_name(state.person_id)
                if person_name:
                    # Use hash of global_track_id as numeric track_id for gallery manager
                    track_id_num = hash(state.global_track_id) % (10**9)
                    self._reid_gallery_manager.update_track_embedding(
                        track_id_num, crop, person_name, num_persons_in_frame
                    )

    def match_reid_cross_camera(
        self,
        new_track_id: str,
        new_crop: np.ndarray,
        candidate_track_ids: list[str],
        num_persons_in_frame: int = 1,
    ) -> Optional[tuple[str, float, Optional[str]]]:
        """Match a new track against existing tracks using Re-ID.

        This is used for cross-camera matching when a person appears
        on a new camera and we need to find if they were seen before.

        First tries matching against active track states, then against
        the shared gallery of lost face-identified persons.

        Args:
            new_track_id: ID of the new track
            new_crop: Person crop from new track
            candidate_track_ids: IDs of tracks to match against
            num_persons_in_frame: Number of persons in frame (skip if >1)

        Returns:
            Tuple of (matched_track_id, similarity, person_name) or None
        """
        # Skip if multiple persons to avoid confusion
        if num_persons_in_frame > 1:
            return None

        # Extract embedding from new track
        new_embedding, quality = self.reid_extractor.extract(new_crop, return_quality=True)

        if quality < self.reid_config.min_visibility:
            return None

        best_match_id = None
        best_score = 0.0
        best_person_name = None

        # First, try matching against active track states
        for track_id in candidate_track_ids:
            state = self._track_states.get(track_id)
            if state is None or len(state.reid_gallery) == 0:
                continue

            # Use gallery-based matching for stability
            similarity = state.reid_gallery.match(new_embedding)

            if (
                similarity > self.reid_config.similarity_threshold
                and similarity > best_score
            ):
                best_match_id = track_id
                best_score = similarity
                if state.is_identified:
                    best_person_name = self._get_person_name(state.person_id)

        # Also try matching against the shared gallery of lost persons
        if self._reid_gallery_manager:
            match_result = self._reid_gallery_manager.match_new_track(
                new_crop, num_persons_in_frame
            )
            if match_result.matched and match_result.score > best_score:
                # Gallery match is better - return person name instead of track ID
                logger.info(
                    f"Re-ID gallery match: {new_track_id} -> {match_result.person_name} "
                    f"(sim={match_result.score:.2f})"
                )
                return (None, match_result.score, match_result.person_name)

        if best_match_id is not None:
            logger.debug(
                f"Re-ID match: {new_track_id} -> {best_match_id} (sim={best_score:.2f})"
            )
            return (best_match_id, best_score, best_person_name)

        return None

    def transfer_identity(self, from_track_id: str, to_track_id: str):
        """Transfer identity from one track to another.

        Used when tracks are matched across cameras.

        Args:
            from_track_id: Source track ID
            to_track_id: Target track ID
        """
        from_state = self._track_states.get(from_track_id)
        to_state = self.register_track(to_track_id)

        if from_state is None:
            return

        # Transfer identity if source is identified
        if from_state.is_identified:
            to_state.person_id = from_state.person_id
            to_state.identification_confidence = from_state.identification_confidence
            to_state.identified_by = f"transfer_from_{from_track_id}"
            to_state.identified_at = time.time()

        # Transfer face embedding if available
        if from_state.face_embedding is not None:
            to_state.face_embedding = from_state.face_embedding

        # Transfer Re-ID gallery
        for emb in from_state.reid_gallery.embeddings:
            to_state.reid_gallery.embeddings.append(emb)

        logger.info(
            f"Transferred identity from {from_track_id} to {to_track_id} "
            f"(person_id={to_state.person_id})"
        )

    def get_identified_tracks(self) -> list[tuple[str, int, str]]:
        """Get all identified tracks.

        Returns:
            List of (track_id, person_id, method) tuples
        """
        result = []
        for track_id, state in self._track_states.items():
            if state.is_identified:
                result.append((track_id, state.person_id, state.identified_by))
        return result

    def cleanup_old_tracks(self, max_age_seconds: float = 3600):
        """Remove old track states to prevent memory growth.

        Args:
            max_age_seconds: Maximum age of tracks to keep
        """
        current_time = time.time()
        to_remove = []

        for track_id, state in self._track_states.items():
            age = current_time - state.first_seen
            if age > max_age_seconds:
                to_remove.append(track_id)

        for track_id in to_remove:
            del self._track_states[track_id]

        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} old track states")
