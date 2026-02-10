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

from typing import TYPE_CHECKING

from src.config import Config, FaceRecognitionConfig, ReIDConfig
from src.database.repository import Repository
from src.recognition.face_recognizer import FaceRecognizer
from src.recognition.identification_manager import IdentificationManager, CrossCameraMatch
from src.recognition.reid_extractor import ReIDExtractor, EmbeddingGallery
from src.utils.profiler import profiler
from src.utils.image_utils import crop_face_region, crop_with_margin

if TYPE_CHECKING:
    from src.recognition.unidentified_face_manager import UnidentifiedFaceManager

logger = logging.getLogger(__name__)

# Identification method precedence (weakest → strongest).
# A track can only be upgraded, never downgraded.
IDENTIFICATION_PRECEDENCE = {"none": 0, "handover": 1, "reid": 2, "reid_gallery": 2, "face": 3}


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
    reid_info: Optional[tuple[str, float]] = None  # (closest_name, confidence)
    face_info: Optional[tuple[str, float]] = None
    is_reid_identified: bool = False
    has_multiple_faces: bool = False  # Added for compatibility
    stationary_time: Optional[int] = None


@dataclass
class TrackIdentityState:
    """Identity state for a global track.

    Extended state for production use with consecutive match tracking.
    """

    global_track_id: str
    person_id: Optional[int] = None
    identification_confidence: float = 0.0
    identified_by: str = "none"  # 'face', 'reid', 'reid_gallery', 'handover', 'none'

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
    last_gallery_check: float = 0.0
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
        """Confirm identity assignment.

        Respects identification precedence — a weaker method cannot
        overwrite a stronger one (e.g. reid cannot overwrite face).
        """
        current_rank = IDENTIFICATION_PRECEDENCE.get(self.identified_by, 0)
        new_rank = IDENTIFICATION_PRECEDENCE.get(method, 0)
        if new_rank < current_rank:
            logger.debug(
                f"Track {self.global_track_id}: ignoring {method} (rank {new_rank}), "
                f"already identified by {self.identified_by} (rank {current_rank})"
            )
            return
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
        unidentified_face_manager: Optional["UnidentifiedFaceManager"] = None,
        snapshot_config: Optional[Config] = None, # Using full config for easier access if needed
    ):
        """Initialize identity linker.

        Args:
            face_config: Face recognition configuration
            reid_config: Re-ID configuration
            repository: Database repository
            enable_debug_images: Whether to save debug images
            face_recognizer: Optional pre-initialized face recognizer (for testing)
            reid_extractor: Optional pre-initialized Re-ID extractor (for testing)
            unidentified_face_manager: Optional manager for capturing unidentified faces
            snapshot_config: Optional snapshot configuration
        """
        self.face_config = face_config
        self.reid_config = reid_config
        self.repository = repository
        self._unidentified_face_manager = unidentified_face_manager
        self.snapshot_config = snapshot_config.snapshots if snapshot_config else None

        # Create a minimal config for IdentificationManager
        self._config = self._create_config(face_config, reid_config)
        if snapshot_config:
            self._config.snapshots = snapshot_config.snapshots

        # Core identification manager (handles face recognition and Re-ID)
        self._id_manager = IdentificationManager(
            config=self._config,
            repository=repository,
            enable_debug_images=enable_debug_images,
            face_recognizer=face_recognizer,
            reid_extractor=reid_extractor,
            unidentified_face_manager=unidentified_face_manager,
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
        
        # Get info from underlying IdentificationManager
        identity = self._id_manager.get_identity(global_track_id)
        
        return IdentificationResult(
            person_id=state.person_id,
            person_name=name,
            confidence=state.identification_confidence,
            method=state.identified_by,
            is_confirmed=state.is_identified,
            is_reid_identified=state.identified_by in ("reid", "reid_gallery", "handover"),
            face_info=identity.face_info if identity else None,
            reid_info=identity.reid_info if identity else None,
            reid_score=identity.reid_score if identity else -1.0,
            has_multiple_faces=identity.has_multiple_faces if identity else False
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
        camera_id: Optional[str] = None,
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
            camera_id: Camera ID for unidentified face capture

        Returns:
            IdentificationResult with identification status
        """
        state = self.register_track(global_track_id)
        current_time = time.time()

        # If already identified, just update Re-ID gallery
        if state.is_identified:
            self._update_reid_gallery(state, person_crop, num_persons_in_frame, precomputed_reid)
            name = self._get_person_name(state.person_id)
            
            # Also get latest reid_score/info for display
            identity = self._id_manager.get_identity(global_track_id)
            
            return IdentificationResult(
                person_id=state.person_id,
                person_name=name,
                confidence=state.identification_confidence,
                method=state.identified_by,
                is_confirmed=True,
                is_reid_identified=state.identified_by == "reid",
                face_info=(name, state.identification_confidence) if state.identified_by == "face" else None,
                reid_info=identity.reid_info if identity else None,
                reid_score=identity.reid_score if identity else -1.0
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
        
        # Initialize result with reid_info=None to avoid AttributeError in renderer
        result = IdentificationResult(reid_info=None)

        if should_check_face and self.face_config.enabled:
            state.last_face_check = current_time
            face_result = self._try_face_identification(state, frame, bbox, person_crop, num_persons_in_frame, camera_id)

            if face_result.is_confirmed:
                # Update with reid_info for rendering before returning
                identity = self._id_manager.get_identity(global_track_id)
                if identity:
                    face_result.reid_info = identity.reid_info
                    face_result.reid_score = identity.reid_score
                return face_result
            elif face_result.person_id is not None:
                result = face_result

        # Always update Re-ID gallery for cross-camera matching
        self._update_reid_gallery(state, person_crop, num_persons_in_frame, precomputed_reid)

        # For unidentified tracks, try periodic gallery recheck
        if not state.is_identified:
            if self._try_gallery_identification(state, person_crop, num_persons_in_frame, camera_id):
                name = self._get_person_name(state.person_id)
                return IdentificationResult(
                    person_id=state.person_id,
                    person_name=name,
                    confidence=state.identification_confidence,
                    method="reid_gallery",
                    is_confirmed=True,
                    is_reid_identified=True,
                )

        # Get best-match info from underlying IdentificationManager for unidentified tracks
        if not result.is_confirmed:
            identity = self._id_manager.get_identity(global_track_id)
            if identity:
                result.reid_info = identity.reid_info
                result.face_info = identity.face_info
                result.reid_score = identity.reid_score
                result.has_multiple_faces = identity.has_multiple_faces

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
        camera_id: Optional[str] = None,
    ) -> IdentificationResult:
        """Try to identify track using face recognition.

        Uses IdentificationManager for core logic, adds consecutive match tracking.
        """
        if not self._id_manager.face_recognizer:
            return IdentificationResult()

        gallery = self._id_manager.face_gallery

        # Crop face region from person bbox (upper portion)
        face_region = crop_face_region(frame, bbox, height_ratio=0.5)
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

        # Match against gallery (track best match regardless of threshold)
        best_match_id = None
        best_match_name = None
        best_score = 0.0
        best_face_id = None

        if gallery:
            with profiler.measure("FaceRecognizer.compare"):
                # Vectorized comparison for better performance
                gallery_embeddings = np.array([entry.embedding for entry in gallery])
                similarities = face_recognizer.compare_embeddings_batch(
                    embedding, gallery_embeddings
                )
                best_idx = int(np.argmax(similarities))
                best_score = float(similarities[best_idx])
                
                best_entry = gallery[best_idx]
                best_match_id = best_entry.person_id
                best_match_name = best_entry.person_name
                best_face_id = best_entry.embedding_id

        # Check if match is above threshold
        if best_score < self.face_config.similarity_threshold:
            # Submit to unidentified face manager
            if self._unidentified_face_manager and camera_id:
                face_crop, face_bbox = crop_with_margin(
                    face_region, best_face.bbox, margin_ratio=0.3
                )
                if face_crop.size > 0:
                    self._unidentified_face_manager.submit(
                        camera_id=camera_id,
                        face_crop=face_crop,
                        embedding=embedding,
                        face_bbox=face_bbox,
                        track_id=state.global_track_id,
                        best_match_person_id=best_match_id,
                        best_match_score=best_score if best_score > 0 else None,
                        best_match_face_id=best_face_id if best_match_id else None,
                    )

            return IdentificationResult(
                face_info=(best_match_name, best_score) if best_match_name else None
            )

        # Consecutive match tracking for stability
        if state.candidate_person_id == best_match_id:
            state.consecutive_matches += 1
        else:
            state.candidate_person_id = best_match_id
            state.consecutive_matches = 1

        min_consecutive = 2
        if state.consecutive_matches >= min_consecutive:
            previous_person_id = state.person_id
            was_reid_identified = (state.identified_by == "reid")
            
            state.confirm_identity(best_match_id, best_score, "face")

            # Save snapshot
            snapshot_path = self._save_snapshot(face_region, best_face.bbox, "face_match", best_match_name)

            # Update database
            self.repository.update_track(
                state.global_track_id,
                person_id=best_match_id,
                face_embedding=embedding,
            )
            
            # Emit events
            self.repository.create_event(
                camera_id=camera_id or "unknown",
                event_type="face_match",
                track_id=state.global_track_id,
                person_id=best_match_id,
                confidence=best_score,
                face_embedding_id=best_face_id,
                snapshot_path=snapshot_path,
                extra_data={
                    "threshold": self.face_config.similarity_threshold,
                    "face_id": best_face_id
                }
            )

            if was_reid_identified and previous_person_id is not None:
                self.repository.create_event(
                    camera_id=camera_id or "unknown",
                    event_type="id_upgraded",
                    track_id=state.global_track_id,
                    person_id=best_match_id,
                    extra_data={
                        "from": "reid",
                        "to": "face",
                        "previous_person_id": previous_person_id,
                        "new_person_id": best_match_id
                    }
                )

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
        camera_id: Optional[str] = None,
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

    # ==================== Gallery Recheck ====================

    def _try_gallery_identification(
        self,
        state: TrackIdentityState,
        person_crop: np.ndarray,
        num_persons_in_frame: int = 1,
        camera_id: Optional[str] = None,
    ) -> bool:
        """Try to identify an unidentified track via the shared Re-ID gallery.

        Time-gated by reid_config.gallery_recheck_interval.

        Args:
            state: Track identity state
            person_crop: Person crop image
            num_persons_in_frame: Number of persons in frame
            camera_id: Camera ID for event logging

        Returns:
            True if identification succeeded
        """
        current_time = time.time()
        if current_time - state.last_gallery_check < self.reid_config.gallery_recheck_interval:
            return False

        state.last_gallery_check = current_time

        if not self.reid_gallery_manager:
            return False

        with profiler.measure("ReIDGallery.recheck"):
            match_result = self.reid_gallery_manager.match_new_track(
                person_crop, num_persons_in_frame
            )

        if not match_result.matched:
            return False

        # Resolve person name to person_id
        person = self.repository.get_person_by_name(match_result.person_name)
        if not person:
            return False

        state.confirm_identity(person.id, match_result.score, "reid_gallery")

        # Update database
        self.repository.update_track(
            state.global_track_id, person_id=person.id
        )

        # Save current person crop as snapshot
        snapshot_path = self._save_snapshot_crop(
            person_crop, "reid_match", match_result.person_name
        )

        # Emit reid_match event
        self.repository.create_event(
            camera_id=camera_id or "unknown",
            event_type="reid_match",
            track_id=state.global_track_id,
            person_id=person.id,
            confidence=match_result.score,
            reid_embedding_id=match_result.db_id,
            snapshot_path=snapshot_path,
            extra_data={
                "match_policy": "gallery_recheck",
                "threshold": self.reid_config.similarity_threshold,
                "top1_score": match_result.best_score,
                "original_reid_embedding_id": match_result.db_id,
            }
        )

        logger.info(
            f"Gallery recheck: track {state.global_track_id} identified as "
            f"{match_result.person_name} (sim={match_result.score:.2f})"
        )

        return True

    # ==================== Cross-Camera Re-ID ====================

    def _find_track_for_person(self, person_name: str, candidate_track_ids: list[str]) -> Optional[str]:
        """Find a candidate track belonging to the given person.

        Used to resolve a gallery match (person name) to a concrete track ID.

        Args:
            person_name: Person name to look up
            candidate_track_ids: Track IDs to search among

        Returns:
            Matching track ID, or None if no candidate belongs to this person
        """
        for track_id in candidate_track_ids:
            state = self._track_states.get(track_id)
            if state and state.is_identified:
                if self._get_person_name(state.person_id) == person_name:
                    return track_id
        return None

    def match_reid_cross_camera(
        self,
        new_track_id: str,
        new_crop: np.ndarray,
        candidate_track_ids: list[str],
        num_persons_in_frame: int = 1,
        precomputed_reid: Optional[tuple[np.ndarray, float]] = None,
    ) -> Optional[CrossCameraMatch]:
        """Match a new track against existing tracks using Re-ID.

        Checks both per-track galleries and the shared gallery. When the
        shared gallery wins, we try to resolve the person name back to a
        candidate track so the caller can relink to an existing global
        track rather than creating a new one.

        Args:
            new_track_id: ID of the new track
            new_crop: Person crop from new track
            candidate_track_ids: IDs of tracks to match against
            num_persons_in_frame: Number of persons in frame
            precomputed_reid: Optional pre-computed Re-ID (embedding, quality)

        Returns:
            CrossCameraMatch object or None
        """
        if num_persons_in_frame > 1:
            logger.debug(f"Re-ID skip: {new_track_id} has {num_persons_in_frame} persons in frame")
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
            logger.debug(f"Re-ID skip: {new_track_id} quality {quality:.2f} < {self.reid_config.min_visibility}")
            return None

        logger.debug(
            f"Re-ID cross-camera: {new_track_id}, candidates={len(candidate_track_ids)}, "
            f"quality={quality:.2f}"
        )

        best_match_id = None
        best_score = 0.0
        best_person_name = None
        match_source = None  # "per_track" or "gallery" or "gallery_resolved"

        # 1. Match against per-track galleries
        with profiler.measure("ReIDExtractor.compare"):
            for track_id in candidate_track_ids:
                state = self._track_states.get(track_id)
                if state is None or len(state.reid_gallery) == 0:
                    continue

                similarity = state.reid_gallery.match(new_embedding)
                person_name = self._get_person_name(state.person_id) if state.is_identified else None
                logger.debug(
                    f"  candidate {track_id}: gallery_size={len(state.reid_gallery)}, "
                    f"sim={similarity:.3f}, person={person_name}"
                )
                if similarity > self.reid_config.similarity_threshold and similarity > best_score:
                    best_match_id = track_id
                    best_score = similarity
                    best_person_name = person_name
                    match_source = "per_track"

        # 2. Also try shared gallery — does NOT early-return so it
        #    cannot discard a valid per-track match
        if self.reid_gallery_manager:
            with profiler.measure("ReIDGallery.match"):
                match_result = self.reid_gallery_manager.match_new_track(new_crop, num_persons_in_frame)
            if match_result.matched and match_result.score > best_score:
                # Try to resolve person name to a candidate track
                resolved_track = self._find_track_for_person(
                    match_result.person_name, candidate_track_ids
                )
                if resolved_track:
                    best_match_id = resolved_track
                    match_source = "gallery_resolved"
                    logger.debug(
                        f"  gallery match resolved: {match_result.person_name} -> track {resolved_track}"
                    )
                else:
                    # Gallery-only match — no candidate track found, but we know the person
                    match_source = "gallery"
                    logger.debug(
                        f"  gallery match unresolved: {match_result.person_name} "
                        f"(no candidate track found)"
                    )
                best_score = match_result.score
                best_person_name = match_result.person_name

                # Emit reid_match event for gallery matches
                if self.repository:
                    person = self.repository.get_person_by_name(match_result.person_name)
                    person_id = person.id if person else None
                    snapshot_path = self._save_snapshot_crop(
                        new_crop, "reid_match", match_result.person_name
                    )
                    self.repository.create_event(
                        camera_id="unknown",
                        event_type="reid_match",
                        track_id=new_track_id,
                        person_id=person_id,
                        confidence=match_result.score,
                        reid_embedding_id=match_result.db_id,
                        snapshot_path=snapshot_path,
                        extra_data={
                            "match_policy": "gallery" if not resolved_track else "gallery_resolved",
                            "threshold": self.reid_config.similarity_threshold,
                            "top1_score": match_result.best_score,
                            "resolved_track": resolved_track,
                            "original_reid_embedding_id": match_result.db_id,
                        }
                    )

        # 3. Single return path
        if best_score > 0 and (best_match_id is not None or best_person_name is not None):
            logger.info(
                f"Re-ID match: {new_track_id} -> track={best_match_id}, "
                f"person={best_person_name}, sim={best_score:.2f}, source={match_source}"
            )

            # Emit reid_match event for per-track matches (gallery events emitted above)
            if match_source == "per_track" and self.repository:
                state = self._track_states.get(best_match_id)
                person_id = state.person_id if state else None
                self.repository.create_event(
                    camera_id="unknown",
                    event_type="reid_match",
                    track_id=new_track_id,
                    person_id=person_id,
                    confidence=best_score,
                    extra_data={
                        "match_policy": "cross_camera",
                        "matched_track_id": best_match_id,
                        "threshold": self.reid_config.similarity_threshold
                    }
                )

            return CrossCameraMatch(
                matched_track_id=best_match_id,
                similarity=best_score,
                person_name=best_person_name
            )

        logger.debug(f"Re-ID no match for {new_track_id}")
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
            to_state.identified_by = "handover"
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

    # ==================== Snapshot Saving ====================

    def _save_snapshot(self, image: np.ndarray, bbox: tuple, event_type: str, person_name: Optional[str] = None) -> Optional[str]:
        """Save a snapshot to disk.

        Returns:
            Path to saved snapshot or None if disabled
        """
        if not self.snapshot_config or not self.snapshot_config.enabled:
            return None

        # Determine if we should save based on event type
        should_save = False
        if event_type == "face_match" and self.snapshot_config.save_on_identification:
            should_save = True
        
        if not should_save:
            return None

        try:
            from src.utils.image_utils import write_jpeg, crop_with_margin
            import uuid
            from pathlib import Path

            snapshot_dir = Path(self.snapshot_config.path)
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            unique_id = uuid.uuid4().hex[:8]
            name_part = f"_{person_name}" if person_name else ""
            filename = f"{timestamp}_{event_type}{name_part}_{unique_id}.jpg"
            filepath = snapshot_dir / filename

            # Crop the face from the provided image (face_region)
            face_crop, _ = crop_with_margin(image, bbox, margin_ratio=0.3)
            if face_crop.size > 0:
                write_jpeg(str(filepath), face_crop)
                return str(filepath)
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
            return None

    def _save_snapshot_crop(self, crop: np.ndarray, event_type: str, label: str = "") -> Optional[str]:
        """Save a person crop directly as an event snapshot (no bbox cropping).

        Returns:
            Path to saved snapshot or None if disabled/failed.
        """
        if not self.snapshot_config or not self.snapshot_config.enabled:
            return None

        try:
            from src.utils.image_utils import write_jpeg
            import uuid
            from pathlib import Path

            snapshot_dir = Path(self.snapshot_config.path) / event_type
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            unique_id = uuid.uuid4().hex[:8]
            label_part = f"_{label}" if label else ""
            filename = f"{timestamp}{label_part}_{unique_id}.jpg"

            write_jpeg(str(snapshot_dir / filename), crop)
            return str(snapshot_dir / filename)
        except Exception as e:
            logger.debug(f"Failed to save {event_type} snapshot: {e}")
            return None
