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
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from typing import TYPE_CHECKING

from src.config import Config, FaceRecognitionConfig, ReIDConfig
from src.database.repository import Repository
from src.recognition.age_gate import GalleryImage, face_crop
from src.recognition.face_recognizer import FaceRecognizer
from src.recognition.id_method import IdMethod
from src.recognition.identification_manager import IdentificationManager
from src.recognition.reid_extractor import ReIDExtractor
from src.tracking.attribution import ANCHOR_SAMPLE_MAX_AGE_SEC, Overlap, PersonAppearance, posture
from src.utils.jsonl_writer import DailyJsonlWriter
from src.utils.profiler import profiler

if TYPE_CHECKING:
    from src.recognition.age_gate import AgeGate
    from src.recognition.face_score_logger import FaceScoreLogger
    from src.recognition.unidentified_face_manager import UnidentifiedFaceManager

logger = logging.getLogger(__name__)

# An owned face crop this recent may give the track an age sample (attribution.skeleton_face)
AGE_FACE_FRESH_SEC = 2.0


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

    # Latest Re-ID sample of the ISOLATED box (time.time() in last_appearance_at); a face
    # confirmation reports it, if fresh, to the person's face-confirmed appearance
    # (attribution.PersonAppearance, shared by all tracks)
    last_appearance: Optional[np.ndarray] = None
    last_appearance_at: float = 0.0
    # The box's overlap with other boxes this frame (attribution.track_overlaps), logged with gallery samples
    last_overlap: Optional[Overlap] = None
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
    # Naming latency metric (metrics/naming/, written when the track state ends)
    first_named_at: Optional[float] = None
    first_named_by: IdMethod = IdMethod.NONE
    first_face_at: Optional[float] = None
    named_persons: list[int] = field(default_factory=list)   # each person the track was named, in order
    revocations: int = 0
    first_camera_id: Optional[str] = None

    # Last processing time (idle-based eviction in cleanup_old_tracks)
    last_seen_time: float = field(default_factory=time.time)

    # Apparent-age samples of this track (age_gate.AgeGate, shared by all tracks):
    # confirm_identity refuses a person they rule out
    age_samples: list[float] = field(default_factory=list)
    age_gate: Optional["AgeGate"] = None
    camera_id: Optional[str] = None
    last_age_sample: float = 0.0
    # Latest owned face crop (age samples use it while fresh)
    last_face_crop: Optional[np.ndarray] = None
    last_face_crop_time: float = 0.0

    @property
    def is_identified(self) -> bool:
        """Check if track has been linked to a known person."""
        return self.person_id is not None

    def confirm_identity(self, person_id: int, confidence: float, method: IdMethod) -> bool:
        """Confirm identity assignment — every naming path goes through here.

        A weaker method cannot overwrite a stronger one (IdMethod.strength):
        Re-ID cannot overwrite face, face can correct Re-ID or handover.
        A person the track's apparent age rules out (AgeGate.veto) is refused, for
        appearance-based methods (face only with age.veto_face, AgeGate.vetoes).

        Returns:
            True if the track now carries this identity
        """
        if method.strength < self.identified_by.strength:
            logger.debug(
                f"Track {self.global_track_id}: ignoring {method}, "
                f"already identified by {self.identified_by}"
            )
            return False
        if self.age_gate is not None and self.age_gate.vetoes(method) and self.age_gate.veto(
                person_id, self.age_samples, self.global_track_id, str(method), camera_id=self.camera_id):
            return False
        self.person_id = person_id
        self.identification_confidence = confidence
        self.identified_by = method
        self.identified_at = time.time()
        if self.first_named_at is None:
            self.first_named_at, self.first_named_by = self.identified_at, method
        if method == IdMethod.FACE and self.first_face_at is None:
            self.first_face_at = self.identified_at
        if self.named_persons[-1:] != [person_id]:
            self.named_persons.append(person_id)
        if method == IdMethod.FACE and self.appearances is not None \
                and time.time() - self.last_appearance_at <= ANCHOR_SAMPLE_MAX_AGE_SEC:
            self.appearances.face_confirmed(person_id, self.global_track_id, self.last_appearance)
        logger.info(
            f"Track {self.global_track_id} identified as person {person_id} "
            f"via {method} (conf={confidence:.2f})"
        )
        return True


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
        age_gate: Optional["AgeGate"] = None,
        naming_log: Optional[DailyJsonlWriter] = None,
        reid_feed_log: Optional[DailyJsonlWriter] = None,
        gallery_scores_log: Optional[DailyJsonlWriter] = None,
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
            age_gate: Optional apparent-age veto (age.enabled); samples isolated tracks
            naming_log: Optional per-track naming latency log (metrics/naming/, tools/analyze_naming.py)
            reid_feed_log: Optional log of every buffered gallery sample with the box's overlap (metrics/reid_feed/)
        """
        self.face_config = face_config
        self.reid_config = reid_config
        self.repository = repository
        self._unidentified_face_manager = unidentified_face_manager
        self._face_score_logger = face_score_logger
        self._age_gate = age_gate
        self._naming_log = naming_log
        self._reid_feed_log = reid_feed_log
        self._gallery_scores_log = gallery_scores_log
        self._person_ages_refreshed_at = 0.0
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
    def age_gate(self) -> Optional["AgeGate"]:
        """Apparent-age veto (None when age.enabled is off)."""
        return self._age_gate

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
                age_gate=self._age_gate,
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
        appearance_settled: bool = True,
        keypoints: Optional[np.ndarray] = None,
        overlap: Optional[Overlap] = None,
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
            appearance_settled: The box's appearance did not just change (attribution.py);
                otherwise a single strong face may not name it, only the streak
            keypoints: Pose keypoints of the box's detection (attribution.skeleton_face)
            overlap: The box's overlap with other boxes (attribution.track_overlaps), logged
                with each gallery sample (metrics/reid_feed/)

        Returns:
            IdentificationResult with identification status
        """
        state = self.register_track(global_track_id)
        current_time = time.time()

        state.last_seen_time = current_time
        state.last_overlap = overlap
        if camera_id is not None:
            state.camera_id = camera_id
            state.first_camera_id = state.first_camera_id or camera_id

        # Appearance evidence: an isolated box (an overlapped one shows two people), standing
        # when reid.gallery_needs_standing (a sitting/lying crop is mostly furniture)
        feeds_appearance = isolated and self._standing_for_gallery(keypoints)
        if precomputed_reid is not None and feeds_appearance:
            state.last_appearance, state.last_appearance_at = precomputed_reid[0], current_time

        # Face until face-confirmed; an appearance-based label is re-checked less
        # often, so face can still correct or upgrade it (IdMethod.strength)
        if self.face_config.enabled and state.identified_by != IdMethod.FACE:
            interval = (self.face_config.reid_confirmation_interval if state.is_identified
                        else self.face_config.detection_interval)
            if force_face_check or current_time - state.last_face_check > interval / 2.0:
                state.last_face_check = current_time
                self._try_face_identification(state, frame, bbox, person_crop, feeds_appearance, camera_id,
                                              allow_strong=appearance_settled, keypoints=keypoints)

        # Age: the box's own face on any track, the body only when isolated
        self._sample_age(state, person_crop, camera_id, isolated, keypoints)

        # Appearance evidence only for isolated tracks (attribution.py)
        if isolated:
            if feeds_appearance:
                self._update_reid_gallery(state, person_crop, precomputed_reid, camera_id)
            if not state.is_identified:
                self._try_gallery_identification(state, person_crop, camera_id, persons_present, precomputed_reid)

        return self.get_identity(global_track_id)

    def _standing_for_gallery(self, keypoints: Optional[np.ndarray]) -> bool:
        """Whether the box's appearance may feed the gallery: standing skeleton (attribution.posture)
        when reid.gallery_needs_standing; a box-only detector (no keypoints) is not gated."""
        return not self.reid_config.gallery_needs_standing or keypoints is None or posture(keypoints) == "standing"

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
        state.revocations += 1
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
        feeds_appearance: bool,
        camera_id: Optional[str] = None,
        allow_strong: bool = True,
        keypoints: Optional[np.ndarray] = None,
    ) -> bool:
        """Production face step: `IdentificationManager.attempt_face` + DB persistence.

        Returns:
            True if the track just became face-identified
        """
        attempt = self._id_manager.attempt_face(frame, bbox, state.global_track_id, camera_id,
                                                allow_strong=allow_strong, keypoints=keypoints)
        if attempt is None or attempt.ambiguous:
            return False
        state.face_embedding = attempt.embedding
        if self._age_gate is not None:
            state.last_face_crop = face_crop(attempt.region, attempt.face.bbox)
            state.last_face_crop_time = time.time()
        if not attempt.confirmed:
            return False

        best = attempt.scores.best_entry
        score = attempt.scores.top1
        previous_person_id, previous_method = state.person_id, state.identified_by
        if not state.confirm_identity(best.person_id, score, IdMethod.FACE):
            return False

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

        if self.reid_gallery_manager and feeds_appearance:   # isolated (+ standing) box
            if self.reid_gallery_manager.update_track_embedding(
                    self.gallery_track_key(state.global_track_id), person_crop, best.person_name):
                self._log_reid_feed(state, "face_confirmed")
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
        state.last_appearance, state.last_appearance_at = embedding, time.time()
        if not self._appearances.consistent(state.person_id, embedding):
            return
        if quality >= self.reid_config.min_visibility:
            if self.reid_gallery_manager.update_track_embedding(
                    self.gallery_track_key(state.global_track_id), crop, person_name,
                    embedding=embedding, quality=quality):
                self._log_reid_feed(state, "appearance")

    # ==================== Gallery Recheck ====================

    def recheck_interval(self, track_age: float) -> float:
        """Seconds between gallery rechecks of an unidentified track of this age (seconds)."""
        cfg = self.reid_config
        if cfg.gallery_recheck_age_factor <= 0:
            return cfg.gallery_recheck_interval
        return min(cfg.gallery_recheck_interval,
                   max(cfg.gallery_recheck_min_interval, cfg.gallery_recheck_age_factor * track_age))

    def _try_gallery_identification(
        self,
        state: TrackIdentityState,
        person_crop: np.ndarray,
        camera_id: Optional[str] = None,
        persons_present: frozenset[int] = frozenset(),
        precomputed_reid: Optional[tuple[np.ndarray, float]] = None,
    ) -> bool:
        """Try to identify an unidentified (isolated) track via the shared Re-ID gallery.

        Time-gated by recheck_interval(): frequent while the track is young (a track
        that missed the new-track match at birth otherwise needed min_consecutive_matches
        x 10 s, longer than a door passage — 2026-10-21 Virag/Judit stayed "hallway").

        Args:
            state: Track identity state
            person_crop: Person crop image
            camera_id: Camera ID for event logging
            persons_present: Persons held by other live tracks on this camera
            precomputed_reid: This frame's Re-ID of the crop (else extracted here, ~30 ms)

        Returns:
            True if identification succeeded
        """
        current_time = time.time()
        if current_time - state.last_gallery_check < self.recheck_interval(current_time - state.first_seen):
            return False

        state.last_gallery_check = current_time

        if not self.reid_gallery_manager:
            return False

        with profiler.measure("ReIDGallery.recheck"):
            match_result = self.reid_gallery_manager.match_new_track(person_crop, *(precomputed_reid or (None, None)))
        self._log_gallery_scores(state, camera_id, match_result)

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

        if not state.confirm_identity(person.id, match_result.score, IdMethod.REID_GALLERY):
            state.gallery_candidate_person_id = None
            state.gallery_consecutive_matches = 0
            return False

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

    # ==================== Apparent Age ====================

    def _sample_age(self, state: TrackIdentityState, person_crop: np.ndarray, camera_id: Optional[str],
                    isolated: bool = True, keypoints: Optional[np.ndarray] = None) -> None:
        """Add an age sample to a track; the newest age.max_samples are kept (rolling).

        Uses the track's latest owned face while fresh (attribution.skeleton_face: valid on
        an overlapped box too), plus the body only when the box is isolated (an overlapped
        box shows two people); no face and not isolated -> no sample. With age.needs_standing
        a sitting/lying skeleton (attribution.posture) gives no sample at all.
        age.sample_interval_sec apart, age.refresh_interval_sec once the window is full, so
        the age follows a ByteTrack switch (live 2026-10-23: Judit's 7 samples froze Virag's
        box at 32 y and refused 138 accepted Virag faces). Once the samples rule out the
        track's current name, the name is revoked (the veto also blocks naming it so again).
        ~55 ms per sample on the Jetson (TensorRT FP16).
        """
        gate = self._age_gate
        if gate is None:
            return
        now = time.time()
        full = len(state.age_samples) >= gate.config.max_samples
        interval = gate.config.refresh_interval_sec if full else gate.config.sample_interval_sec
        if now - state.last_age_sample < interval:
            return
        pose = posture(keypoints)
        if gate.config.needs_standing and pose in ("sitting", "lying"):
            return
        use_face = now - state.last_face_crop_time <= AGE_FACE_FRESH_SEC
        if not use_face and not isolated:
            return
        state.last_age_sample = now
        self.refresh_person_ages(now)
        age = gate.sample(state.global_track_id, camera_id, state.last_face_crop if use_face else None,
                          person_crop if isolated else None, posture=pose)
        if age is None:
            return
        state.age_samples = (state.age_samples + [age])[-gate.config.max_samples:]
        if state.is_identified and gate.vetoes(state.identified_by):
            veto = gate.veto(state.person_id, state.age_samples, state.global_track_id,
                             str(state.identified_by), action="revoked", camera_id=camera_id)
            if veto:
                self.revoke_identity(
                    state.global_track_id, camera_id,
                    f"age {veto.track_age:.0f}y vs person {veto.person_age:.0f}y (x{veto.ratio:.1f})")

    def refresh_person_ages(self, now: Optional[float] = None, max_age_sec: float = 60.0) -> None:
        """Re-derive person ages from the face gallery: a DB read, plus ~50 ms per gallery
        image that has no stored age from the current model yet (new enrolments)."""
        now = time.time() if now is None else now
        if self._age_gate is None or now - self._person_ages_refreshed_at < max_age_sec:
            return
        self._person_ages_refreshed_at = now
        recognizer = self.face_recognizer
        if recognizer is None:
            return
        with profiler.measure("AgeGate.refresh_person_ages"):
            self._age_gate.refresh_person_ages(
                [GalleryImage(*row) for row in self.repository.get_face_gallery_images()],
                lambda image: recognizer.detect_faces(image, min_face_size=20).faces,
                store=self.repository.set_face_apparent_age,
            )

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

        if from_state.is_identified and not to_state.confirm_identity(
                from_state.person_id, from_state.identification_confidence, IdMethod.HANDOVER):
            return

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

    def _log_naming(self, state: TrackIdentityState) -> None:
        """One record per ended track: how long it took to name it, by what, and how stable the name was (~10µs)."""
        since = lambda t: round(t - state.first_seen, 2) if t is not None else None
        self._naming_log.write({
            "track": state.global_track_id,
            "camera": state.first_camera_id,
            "last_camera": state.camera_id,
            "duration_sec": round(state.last_seen_time - state.first_seen, 2),
            "named_after_sec": since(state.first_named_at),
            "first_method": str(state.first_named_by) if state.first_named_at is not None else None,
            "face_after_sec": since(state.first_face_at),
            "final_person_id": state.person_id,
            "final_method": str(state.identified_by) if state.person_id is not None else None,
            "persons": state.named_persons,
            "revocations": state.revocations,
        }, datetime.fromtimestamp(state.first_seen, tz=timezone.utc))

    def _log_gallery_scores(self, state: TrackIdentityState, camera_id: Optional[str], match) -> None:
        """One record per gallery recheck: every person's score (~10µs); `streak` is the candidate's
        consecutive matches BEFORE this check, `person_id` the track's name at that moment."""
        if self._gallery_scores_log is None or not match.scores:
            return
        self._gallery_scores_log.write({
            "track": state.global_track_id,
            "camera": camera_id,
            "person_id": state.person_id,
            "method": str(state.identified_by) if state.person_id is not None else None,
            "scores": {name: round(score, 4) for name, score in match.scores.items()},
            "best": match.best_person_name,
            "matched": match.matched,
            "threshold": self.reid_gallery_manager.similarity_threshold,
            "candidate_person_id": state.gallery_candidate_person_id,
            "streak": state.gallery_consecutive_matches,
            "age_sec": round(time.time() - state.first_seen, 1),
        }, datetime.now(timezone.utc))

    def _log_reid_feed(self, state: TrackIdentityState, path: str) -> None:
        """One record per buffered gallery sample: whose track, and how much other boxes (live or
        recently lost) covered it at that moment (~10µs). Joins reid_embeddings by track and time."""
        if self._reid_feed_log is None:
            return
        self._reid_feed_log.write({
            "track": state.global_track_id,
            "camera": state.camera_id,
            "person_id": state.person_id,
            "method": str(state.identified_by),
            "path": path,
            "overlap": state.last_overlap.as_dict() if state.last_overlap is not None else None,
        }, datetime.now(timezone.utc))

    def _forget(self, global_track_id: str) -> None:
        """Drop all per-track identity state (linker, core manager, face score log)."""
        state = self._track_states.pop(global_track_id, None)
        if state is not None and self._naming_log is not None:
            self._log_naming(state)
        self._id_manager.on_track_removed(global_track_id)
        if self._face_score_logger:
            self._face_score_logger.forget_track(global_track_id)
        if self._age_gate:
            self._age_gate.forget_track(global_track_id)

    @staticmethod
    def gallery_track_key(global_track_id: str) -> int:
        """Numeric per-track key of the shared Re-ID gallery's track buffer."""
        return hash(global_track_id) % (10**9)

    def warmup(self):
        """Warm up models."""
        self._id_manager.warmup()
        if self._age_gate is not None:
            try:
                self.refresh_person_ages()   # loads the age model and estimates the gallery once
            except Exception as e:
                logger.error(f"Age gate warmup failed: {e}")

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
