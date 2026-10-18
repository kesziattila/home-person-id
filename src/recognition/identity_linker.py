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
from src.recognition.id_method import IdMethod
from src.recognition.identification_manager import IdentificationManager
from src.recognition.reid_extractor import ReIDExtractor
from src.tracking.attribution import PersonAppearance
from src.utils.profiler import profiler

if TYPE_CHECKING:
    from src.recognition.face_score_logger import FaceScoreLogger
    from src.recognition.unidentified_face_manager import UnidentifiedFaceManager

logger = logging.getLogger(__name__)


@dataclass
class IdentificationResult:
    """Result of person identification."""

    person_id: Optional[int] = None
    person_name: Optional[str] = None
    confidence: float = 0.0
    method: IdMethod = IdMethod.NONE
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
    identified_by: IdMethod = IdMethod.NONE

    # Face embedding if captured
    face_embedding: Optional[np.ndarray] = None

    # Latest Re-ID sample; a face confirmation reports it to the person's
    # face-confirmed appearance (attribution.PersonAppearance, shared by all tracks)
    last_appearance: Optional[np.ndarray] = None
    appearances: Optional[PersonAppearance] = None

    # Consecutive match tracking for the Re-ID gallery recheck path
    # (the face streak lives in IdentificationManager.attempt_face)
    gallery_candidate_person_id: Optional[int] = None
    gallery_consecutive_matches: int = 0

    # Timestamps
    first_seen: float = field(default_factory=time.time)
    last_face_check: float = 0.0
    last_gallery_check: float = 0.0
    identified_at: Optional[float] = None

    # Last processing time (idle-based eviction in cleanup_old_tracks)
    last_seen_time: float = field(default_factory=time.time)

    @property
    def is_identified(self) -> bool:
        """Check if track has been linked to a known person."""
        return self.person_id is not None

    def confirm_identity(self, person_id: int, confidence: float, method: IdMethod):
        """Confirm identity assignment.

        A weaker method cannot overwrite a stronger one (IdMethod.strength):
        Re-ID cannot overwrite face, face can correct Re-ID or handover.
        """
        if method.strength < self.identified_by.strength:
            logger.debug(
                f"Track {self.global_track_id}: ignoring {method}, "
                f"already identified by {self.identified_by}"
            )
            return
        self.person_id = person_id
        self.identification_confidence = confidence
        self.identified_by = method
        self.identified_at = time.time()
        if method == IdMethod.FACE and self.appearances is not None:
            self.appearances.face_confirmed(person_id, self.global_track_id, self.last_appearance)
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
        face_score_logger: Optional["FaceScoreLogger"] = None,
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
            face_score_logger: Optional per-attempt face score log (offline tuning)
        """
        self.face_config = face_config
        self.reid_config = reid_config
        self.repository = repository
        self._unidentified_face_manager = unidentified_face_manager
        self._face_score_logger = face_score_logger
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
            face_score_logger=face_score_logger,
        )

        # Track identity states (production-specific, with consecutive match tracking)
        self._track_states: dict[str, TrackIdentityState] = {}
        # Face-confirmed appearance per person: gates the gallery feed (in memory,
        # like the gallery itself)
        self._appearances = PersonAppearance(reid_config.anchor_similarity)

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
                global_track_id=global_track_id,
                appearances=self._appearances,
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
                self.reid_gallery_manager.on_track_lost(
                    self.gallery_track_key(global_track_id), was_face_identified=True,
                    global_track_id=global_track_id,
                )

            self._forget(global_track_id)
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
            is_reid_identified=state.identified_by.is_appearance,
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
        isolated: bool = True,
        precomputed_reid: Optional[tuple[np.ndarray, float]] = None,
        camera_id: Optional[str] = None,
        persons_present: frozenset[int] = frozenset(),
    ) -> IdentificationResult:
        """Process a track for identification.

        Orchestrates face recognition and Re-ID with timing controls.

        Args:
            global_track_id: Global track ID
            frame: Full frame image
            person_crop: Cropped person image
            bbox: Bounding box of person in frame
            force_face_check: Force face recognition even if recently checked
            isolated: Box overlaps no other track (src/tracking/attribution.py); appearance
                (Re-ID) evidence is only used and stored for isolated tracks
            precomputed_reid: Optional pre-computed Re-ID (embedding, quality)
            camera_id: Camera ID for unidentified face capture
            persons_present: Persons held by OTHER live tracks on this camera
                (GlobalTrackManager.persons_tracked_on_camera); a gallery match
                proposing one of them is rejected (one person, one live track per camera)

        Returns:
            IdentificationResult with identification status
        """
        state = self.register_track(global_track_id)
        current_time = time.time()

        state.last_seen_time = current_time

        if precomputed_reid is not None:
            state.last_appearance = precomputed_reid[0]

        # Face until face-confirmed; an appearance-based label is re-checked less
        # often, so face can still correct or upgrade it (IdMethod.strength)
        if self.face_config.enabled and state.identified_by != IdMethod.FACE:
            interval = (self.face_config.reid_confirmation_interval if state.is_identified
                        else self.face_config.detection_interval)
            if force_face_check or current_time - state.last_face_check > interval / 2.0:
                state.last_face_check = current_time
                self._try_face_identification(state, frame, bbox, person_crop, isolated, camera_id)

        # Appearance evidence only for isolated tracks (attribution.py)
        if isolated:
            self._update_reid_gallery(state, person_crop, precomputed_reid, camera_id)
            if not state.is_identified:
                self._try_gallery_identification(state, person_crop, camera_id, persons_present)

        return self.get_identity(global_track_id)

    def revoke_identity(self, global_track_id: str, camera_id: Optional[str], reason: str) -> None:
        """Unname a track whose identity was contradicted (attribution.py: exclusivity)."""
        state = self._track_states.get(global_track_id)
        if state is None or not state.is_identified:
            return
        logger.info(f"Track {global_track_id}: dropping name person {state.person_id} "
                    f"({state.identified_by}) — {reason}")
        self.repository.create_event(
            camera_id=camera_id or "unknown",
            event_type="id_revoked",
            track_id=global_track_id,
            person_id=state.person_id,
            extra_data={"method": str(state.identified_by), "reason": reason},
        )
        state.person_id = None
        state.identification_confidence = 0.0
        state.identified_by = IdMethod.NONE
        state.identified_at = None
        state.last_face_check = 0.0
        state.gallery_candidate_person_id = None
        state.gallery_consecutive_matches = 0

    def _get_person_name(self, person_id: int) -> Optional[str]:
        """Get person name from ID."""
        return self._id_manager.get_person_name(person_id)

    def _try_face_identification(
        self,
        state: TrackIdentityState,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
        person_crop: np.ndarray,
        isolated: bool,
        camera_id: Optional[str] = None,
    ) -> bool:
        """Production face step: `IdentificationManager.attempt_face` + DB persistence.

        Returns:
            True if the track just became face-identified
        """
        attempt = self._id_manager.attempt_face(frame, bbox, state.global_track_id, camera_id)
        if attempt is None or attempt.ambiguous:
            return False
        state.face_embedding = attempt.embedding
        if not attempt.confirmed:
            return False

        best = attempt.scores.best_entry
        score = attempt.scores.top1
        previous_person_id, previous_method = state.person_id, state.identified_by
        state.confirm_identity(best.person_id, score, IdMethod.FACE)

        snapshot_path = self._save_snapshot(attempt.region, attempt.face.bbox, "face_match", best.person_name)
        self.repository.update_track(
            state.global_track_id,
            person_id=best.person_id,
            face_embedding=attempt.embedding,
        )
        self.repository.create_event(
            camera_id=camera_id or "unknown",
            event_type="face_match",
            track_id=state.global_track_id,
            person_id=best.person_id,
            confidence=score,
            face_embedding_id=best.embedding_id,
            snapshot_path=snapshot_path,
            extra_data={
                "threshold": self.face_config.similarity_threshold,
                "face_id": best.embedding_id,
            }
        )
        if previous_method.is_appearance and previous_person_id is not None:
            self.repository.create_event(
                camera_id=camera_id or "unknown",
                event_type="id_upgraded",
                track_id=state.global_track_id,
                person_id=best.person_id,
                extra_data={
                    "from": str(previous_method),
                    "to": str(IdMethod.FACE),
                    "previous_person_id": previous_person_id,
                    "new_person_id": best.person_id,
                }
            )

        if self.reid_gallery_manager and isolated:
            self.reid_gallery_manager.update_track_embedding(
                self.gallery_track_key(state.global_track_id), person_crop, best.person_name
            )
        return True

    def _update_reid_gallery(
        self,
        state: TrackIdentityState,
        crop: np.ndarray,
        precomputed_reid: Optional[tuple[np.ndarray, float]] = None,
        camera_id: Optional[str] = None,
    ):
        """Feed an identified, isolated track's appearance (if it looks like the person) to the gallery.

        The gallery buffers it per track and keeps it only if the track ends
        face-identified (`ReIDGalleryManager.on_track_lost`).
        """
        if not state.is_identified or not self.reid_gallery_manager:
            return
        person_name = self._get_person_name(state.person_id)
        if not person_name:
            return
        if precomputed_reid:
            embedding, quality = precomputed_reid
        else:
            reid_extractor = self._id_manager.reid_extractor
            if not reid_extractor:
                return

            with profiler.measure("ReIDExtractor.extract"):
                embedding, quality = reid_extractor.extract(crop, return_quality=True)

        # Only the named person's face-confirmed appearance (attribution.PersonAppearance)
        state.last_appearance = embedding
        if not self._appearances.consistent(state.person_id, embedding):
            return
        if quality >= self.reid_config.min_visibility:
            self.reid_gallery_manager.update_track_embedding(
                self.gallery_track_key(state.global_track_id), crop, person_name,
                embedding=embedding, quality=quality,
            )

    # ==================== Gallery Recheck ====================

    def _try_gallery_identification(
        self,
        state: TrackIdentityState,
        person_crop: np.ndarray,
        camera_id: Optional[str] = None,
        persons_present: frozenset[int] = frozenset(),
    ) -> bool:
        """Try to identify an unidentified (isolated) track via the shared Re-ID gallery.

        Time-gated by reid_config.gallery_recheck_interval.

        Args:
            state: Track identity state
            person_crop: Person crop image
            camera_id: Camera ID for event logging
            persons_present: Persons held by other live tracks on this camera

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
            match_result = self.reid_gallery_manager.match_new_track(person_crop)

        if not match_result.matched:
            # A miss breaks the consecutive-match streak
            state.gallery_candidate_person_id = None
            state.gallery_consecutive_matches = 0
            return False

        # Resolve person name to person_id
        person = self.repository.get_person_by_name(match_result.person_name)
        if not person:
            return False

        # One person, one live track per camera (duplicate-person guard). A rejection
        # breaks the streak like a miss: live 2026-10-18 15:55 UTC Virag's track
        # collected Ati 1/3, 2/3, was rejected twice, then confirmed at 3/3 the
        # moment Ati's track was briefly lost
        if person.id in persons_present:
            state.gallery_candidate_person_id = None
            state.gallery_consecutive_matches = 0
            logger.info(
                f"Gallery recheck rejected for {state.global_track_id}: "
                f"{match_result.person_name} (sim={match_result.score:.2f}) is already "
                f"active on {camera_id} (duplicate-person guard)"
            )
            return False

        # Enforce reid.min_consecutive_matches: a single noisy frame (e.g. a
        # lying-down pose OSNet embeds badly) must not flip a track's identity.
        # Previously this config option was parsed but never applied here.
        if state.gallery_candidate_person_id == person.id:
            state.gallery_consecutive_matches += 1
        else:
            state.gallery_candidate_person_id = person.id
            state.gallery_consecutive_matches = 1

        min_consecutive = max(1, self.reid_config.min_consecutive_matches)
        if state.gallery_consecutive_matches < min_consecutive:
            logger.info(
                f"Gallery recheck: {state.global_track_id} candidate "
                f"{match_result.person_name} (sim={match_result.score:.2f}), "
                f"{state.gallery_consecutive_matches}/{min_consecutive} consecutive — waiting"
            )
            return False

        state.confirm_identity(person.id, match_result.score, IdMethod.REID_GALLERY)

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
            to_state.confirm_identity(
                from_state.person_id, from_state.identification_confidence, IdMethod.HANDOVER
            )

        if from_state.face_embedding is not None:
            to_state.face_embedding = from_state.face_embedding

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
        """Remove track states not seen for max_age_seconds (memory bound).

        Eviction is keyed on last_seen_time (idle time), NOT first_seen (track
        age): an actively-processed track must keep its identity indefinitely.
        Evicting by age wiped face-confirmed identities from long-lived tracks
        mid-tracking, letting zone propagation stamp a wrong person onto the
        blank re-registered state (live misID incident, 2026-06-06).
        """
        current_time = time.time()
        to_remove = [
            track_id for track_id, state in self._track_states.items()
            if current_time - state.last_seen_time > max_age_seconds
        ]

        for track_id in to_remove:
            self._forget(track_id)

        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} old track states")

    def _forget(self, global_track_id: str) -> None:
        """Drop all per-track identity state (linker, core manager, face score log)."""
        self._track_states.pop(global_track_id, None)
        self._id_manager.on_track_removed(global_track_id)
        if self._face_score_logger:
            self._face_score_logger.forget_track(global_track_id)

    @staticmethod
    def gallery_track_key(global_track_id: str) -> int:
        """Numeric per-track key of the shared Re-ID gallery's track buffer."""
        return hash(global_track_id) % (10**9)

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
