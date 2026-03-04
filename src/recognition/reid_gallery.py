"""Re-ID Gallery Manager for cross-camera person re-identification.

This module provides a gallery-based Re-ID system that:
1. Stores embeddings by person name (not track ID) for persistence across tracks
2. Keeps multiple embeddings per person for stable matching
3. Stores crops for debug visualization
4. Handles expiration of old entries
5. Supports single-person mode to avoid confusion with multiple people
"""

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from datetime import datetime

import cv2
import numpy as np

from src.recognition.reid_extractor import ReIDExtractor, cosine_similarity, is_grayscale_image

if TYPE_CHECKING:
    from src.recognition.face_recognizer import FaceRecognizer
    from src.database.repository import Repository

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingWithCrop:
    """An embedding stored with its source crop image path."""
    embedding: np.ndarray
    crop_path: Optional[str] = None  # Path to crop file on disk (not in memory)
    timestamp: float = field(default_factory=time.time)
    db_id: Optional[int] = None  # Reference to reid_embeddings.id

    def load_crop(self) -> Optional[np.ndarray]:
        """Load crop image from disk."""
        if self.crop_path and Path(self.crop_path).exists():
            return cv2.imread(self.crop_path)
        return None

    def delete_crop_file(self):
        """Delete the crop file from disk."""
        if self.crop_path:
            try:
                Path(self.crop_path).unlink(missing_ok=True)
            except Exception:
                pass


def _save_crop_file(crop: np.ndarray, cache_dir: str, prefix: str = "track") -> Optional[str]:
    """Save a crop image to disk and return the file path."""
    if crop is None:
        return None
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_prefix = prefix.replace(" ", "_").replace("/", "-")
    filepath = path / f"{safe_prefix}_{timestamp}.jpg"
    cv2.imwrite(str(filepath), crop)
    return str(filepath)


@dataclass
class TrackEmbedding:
    """A single Re-ID embedding captured from an active track."""
    embedding: np.ndarray
    crop_path: Optional[str] = None  # Saved to disk at extraction time

    def load_crop(self) -> Optional[np.ndarray]:
        if self.crop_path and Path(self.crop_path).exists():
            return cv2.imread(self.crop_path)
        return None

    def delete_crop_file(self):
        if self.crop_path:
            try:
                Path(self.crop_path).unlink(missing_ok=True)
            except Exception:
                pass


@dataclass
class TrackReIDData:
    """Accumulated Re-ID data for an active track."""
    person_name: Optional[str] = None
    crop_cache_path: Optional[str] = None
    entries: list[TrackEmbedding] = field(default_factory=list)
    max_entries: int = 10

    def add(self, embedding: np.ndarray, crop: np.ndarray, person_name: Optional[str] = None):
        if person_name:
            self.person_name = person_name
        crop_path = _save_crop_file(crop, self.crop_cache_path, "track") if self.crop_cache_path else None
        self.entries.append(TrackEmbedding(embedding=embedding, crop_path=crop_path))
        if len(self.entries) > self.max_entries:
            old = self.entries[:-self.max_entries]
            for e in old:
                e.delete_crop_file()
            self.entries = self.entries[-self.max_entries:]

    def cleanup(self):
        """Delete all crop files."""
        for e in self.entries:
            e.delete_crop_file()
        self.entries.clear()


@dataclass
class GalleryEntry:
    """Entry in the Re-ID gallery for a known person."""

    person_name: str
    entries: list[EmbeddingWithCrop] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)
    max_embeddings: int = 10
    crop_cache_path: Optional[str] = None  # Base path for crop storage

    def _save_crop_to_disk(self, crop: np.ndarray) -> Optional[str]:
        """Save crop to disk and return the file path."""
        if self.crop_cache_path is None or crop is None:
            return None
        return _save_crop_file(crop, self.crop_cache_path, self.person_name)

    def add_embedding(self, embedding: np.ndarray, crop: Optional[np.ndarray] = None, db_id: Optional[int] = None):
        """Add an embedding to the gallery entry."""
        crop_path = self._save_crop_to_disk(crop) if crop is not None else None
        self._add_embedding_with_path(embedding, crop_path, db_id=db_id)

    def _add_embedding_with_path(self, embedding: np.ndarray, crop_path: Optional[str] = None, db_id: Optional[int] = None):
        """Add an embedding with an already-saved crop path."""
        self.entries.append(EmbeddingWithCrop(
            embedding=embedding,
            crop_path=crop_path,
            db_id=db_id,
        ))
        self.last_seen = time.time()

        # Keep only the most recent entries, delete old crop files
        if len(self.entries) > self.max_embeddings:
            old_entries = self.entries[:-self.max_embeddings]
            for old_entry in old_entries:
                old_entry.delete_crop_file()
            self.entries = self.entries[-self.max_embeddings:]

    def match(self, query_embedding: np.ndarray) -> tuple[float, Optional[np.ndarray], Optional[int]]:
        """Match a query embedding against all gallery embeddings.

        Returns:
            Tuple of (max_similarity_score, best_matching_crop loaded from disk, db_id)
        """
        if not self.entries:
            return 0.0, None, None

        max_score = 0.0
        best_entry = None
        for entry in self.entries:
            score = cosine_similarity(query_embedding, entry.embedding)
            if score > max_score:
                max_score = score
                best_entry = entry

        # Load crop from disk only for the best match
        best_crop = best_entry.load_crop() if best_entry else None
        db_id = best_entry.db_id if best_entry else None
        return max_score, best_crop, db_id

    def cleanup_crop_files(self):
        """Delete all crop files for this entry."""
        for entry in self.entries:
            entry.delete_crop_file()

    @property
    def embeddings(self) -> list[np.ndarray]:
        """Get list of embeddings (for backward compatibility)."""
        return [e.embedding for e in self.entries]

    @property
    def crop(self) -> Optional[np.ndarray]:
        """Get the latest crop (for backward compatibility)."""
        if self.entries:
            return self.entries[-1].load_crop()
        return None


@dataclass
class MatchResult:
    """Result of a Re-ID gallery match."""

    person_name: Optional[str] = None
    score: float = 0.0
    matched: bool = False
    gallery_crop: Optional[np.ndarray] = None
    best_person_name: Optional[str] = None  # Best match even if below threshold
    best_score: float = 0.0
    db_id: Optional[int] = None  # Reference to reid_embeddings.id of the matched embedding


class ReIDGalleryManager:
    """Manages Re-ID gallery for cross-camera person re-identification.

    The gallery is keyed by person name (not track ID) so identity persists
    across multiple tracks of the same person.

    Key features:
    - Stores multiple embeddings per person for stable matching
    - Keeps crops for debug visualization
    - Supports single-person mode to avoid confusion
    - Automatic expiration of old entries
    """

    def __init__(
        self,
        reid_extractor: ReIDExtractor,
        similarity_threshold: float = 0.65,
        max_reappear_time_sec: float = 300.0,
        max_embeddings_per_person: int = 10,
        crop_cache_path: Optional[str] = None,
        gallery_snapshot_dir: Optional[str] = None,
        face_recognizer: Optional["FaceRecognizer"] = None,
        repository: Optional["Repository"] = None,
        debug_saver: Optional['DebugImageSaver'] = None,
    ):
        """Initialize the gallery manager.

        Args:
            reid_extractor: Re-ID feature extractor
            similarity_threshold: Minimum similarity for a match
            max_reappear_time_sec: Time before gallery entries expire
            max_embeddings_per_person: Maximum embeddings to store per person
            crop_cache_path: Path for storing crop images on disk (reduces memory)
            gallery_snapshot_dir: Permanent directory for gallery crops (survives cleanup)
            face_recognizer: Optional face recognizer for multi-face detection
            repository: Optional repository for database persistence
            debug_saver: Optional debug image saver
        """
        self.reid_extractor = reid_extractor
        self.similarity_threshold = similarity_threshold
        self.max_reappear_time_sec = max_reappear_time_sec
        self.max_embeddings_per_person = max_embeddings_per_person
        self.crop_cache_path = crop_cache_path
        self.gallery_snapshot_dir = gallery_snapshot_dir
        self.face_recognizer = face_recognizer
        self.repository = repository
        self.debug_saver = debug_saver

        # Create crop cache directory if specified
        if self.crop_cache_path:
            Path(self.crop_cache_path).mkdir(parents=True, exist_ok=True)

        # Gallery keyed by person name
        self._gallery: dict[str, GalleryEntry] = {}

        # Track data for active tracks (before they're lost)
        self._track_data: dict[int, TrackReIDData] = {}

    @property
    def gallery_size(self) -> int:
        """Number of persons in the gallery."""
        return len(self._gallery)

    @property
    def gallery_persons(self) -> list[str]:
        """List of person names in the gallery."""
        return list(self._gallery.keys())

    def cleanup_expired(self) -> list[str]:
        """Remove expired gallery entries and their crop files.

        Returns:
            List of expired person names
        """
        current_time = time.time()
        expired = [
            name for name, entry in self._gallery.items()
            if current_time - entry.last_seen > self.max_reappear_time_sec
        ]

        for name in expired:
            # Clean up crop files before removing entry
            self._gallery[name].cleanup_crop_files()
            del self._gallery[name]
            logger.info(f"Re-ID gallery expired: {name}")

        return expired

    def _has_multiple_faces(self, crop: np.ndarray) -> bool:
        """Check if a crop contains multiple faces (multiple people).

        Uses lower min_face_size (from config.multi_face_min_size) because we only
        need to detect presence of faces, not quality for recognition.

        Returns:
            True if 2+ faces detected
        """
        if self.face_recognizer is None or crop.size == 0:
            return False

        try:
            # Use lower min_face_size for multi-face detection
            # Gallery crops often have small faces that would be filtered
            # by the normal recognition threshold
            min_size = self.face_recognizer.config.multi_face_min_size
            face_result = self.face_recognizer.detect_faces(crop, min_face_size=min_size)
            return len(face_result.faces) > 1
        except Exception:
            return False

    def update_track_embedding(
        self,
        track_id: int,
        crop: np.ndarray,
        person_name: Optional[str] = None,
        num_persons_in_frame: int = 1,
    ) -> bool:
        """Update Re-ID embedding for an active track.

        Should be called periodically for face-identified tracks.
        Automatically skips grayscale/IR images and crops with multiple faces.

        Args:
            track_id: Track ID
            crop: Person crop image
            person_name: Person name if identified
            num_persons_in_frame: Number of persons detected in the frame

        Returns:
            True if embedding was updated
        """
        # Skip when multiple persons in frame to avoid confusion
        if num_persons_in_frame > 1:
            return False

        # Skip grayscale/IR images - Re-ID relies on color features
        if is_grayscale_image(crop):
            return False

        # Skip if crop contains multiple faces (multiple people in bounding box)
        if self._has_multiple_faces(crop):
            return False

        embedding, quality = self.reid_extractor.extract(crop, return_quality=True)
        if quality < 0.3:
            return False

        if track_id not in self._track_data:
            self._track_data[track_id] = TrackReIDData(
                max_entries=self.max_embeddings_per_person,
                crop_cache_path=self.crop_cache_path,
            )

        self._track_data[track_id].add(embedding, crop, person_name)

        return True

    def on_track_lost(self, track_id: int, was_face_identified: bool, global_track_id: Optional[str] = None) -> bool:
        """Handle a track being lost.

        If the track was face-identified, store its embeddings in the gallery.

        Args:
            track_id: Integer key used for _track_data lookup
            was_face_identified: Whether the track was identified by face recognition
            global_track_id: Original global track ID string for event logging

        Returns:
            True if embeddings were stored in gallery
        """
        if track_id not in self._track_data:
            return False

        data = self._track_data.pop(track_id)

        # Only store if face-identified (not Re-ID identified)
        if not was_face_identified:
            data.cleanup()
            return False

        if not data.person_name or not data.entries:
            data.cleanup()
            return False

        person_name = data.person_name

        # Store/update gallery entry
        if person_name not in self._gallery:
            self._gallery[person_name] = GalleryEntry(
                person_name=person_name,
                max_embeddings=self.max_embeddings_per_person,
                crop_cache_path=self.crop_cache_path,
            )

        entry = self._gallery[person_name]

        # Get person_id from repository if available
        person_id = None
        if self.repository:
            person = self.repository.get_person_by_name(person_name)
            if person:
                person_id = person.id

        event_track_id = global_track_id or f"global_{track_id}"
        last_db_id = None

        for te in data.entries:
            # Copy crop to permanent gallery snapshot dir so it survives cache cleanup
            permanent_crop_path = te.crop_path
            if te.crop_path and self.gallery_snapshot_dir:
                try:
                    dest_dir = Path(self.gallery_snapshot_dir)
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest_path = dest_dir / Path(te.crop_path).name
                    shutil.copy2(te.crop_path, dest_path)
                    permanent_crop_path = str(dest_path)
                except Exception:
                    logger.debug("Failed to copy gallery crop to permanent dir", exc_info=True)

            db_id = None
            if self.repository:
                reid_emb = self.repository.add_reid_embedding(
                    camera_id="unknown",
                    embedding=te.embedding,
                    track_id=event_track_id,
                    person_id=person_id,
                    snapshot_path=permanent_crop_path,
                )
                db_id = reid_emb.id
                last_db_id = db_id

            entry._add_embedding_with_path(te.embedding, permanent_crop_path, db_id=db_id)

        # Emit a single event for the whole flush (not one per embedding)
        if self.repository and last_db_id is not None:
            self.repository.create_event(
                camera_id="unknown",
                event_type="reid_gallery_updated",
                track_id=event_track_id,
                person_id=person_id,
                reid_embedding_id=last_db_id,
                extra_data={
                    "policy": "append",
                    "embeddings_added": len(data.entries),
                    "gallery_count": len(entry.embeddings),
                    "original_reid_embedding_id": last_db_id,
                }
            )

        entry.last_seen = time.time()

        # Save debug image using the last crop
        last_crop = data.entries[-1].load_crop()
        if self.debug_saver and last_crop is not None:
            self.debug_saver.save_gallery_stored(last_crop, person_name, len(entry.embeddings))

        logger.info(f"Track {event_track_id} lost - Re-ID gallery updated: {person_name} ({len(entry.embeddings)} embeddings)")

        return True

    def clear_track_data(self, track_id: int):
        """Clear track data without storing to gallery."""
        data = self._track_data.pop(track_id, None)
        if data:
            data.cleanup()

    def match_new_track(
        self,
        crop: np.ndarray,
        num_persons_in_frame: int = 1,
    ) -> MatchResult:
        """Try to match a new track against the gallery.

        Args:
            crop: Person crop image
            num_persons_in_frame: Number of persons in the frame

        Returns:
            MatchResult with match information
        """
        result = MatchResult()

        # Skip when multiple persons to avoid confusion
        if num_persons_in_frame > 1:
            return result

        # Skip if gallery is empty
        if not self._gallery:
            return result

        # Extract embedding
        embedding, quality = self.reid_extractor.extract(crop, return_quality=True)
        if quality < 0.3:
            return result

        # Match against all gallery entries
        best_name = None
        best_score = 0.0
        best_crop = None

        for person_name, entry in self._gallery.items():
            score, match_crop, db_id = entry.match(embedding)
            logger.debug(
                f"  gallery '{person_name}': score={score:.3f}, "
                f"embeddings={len(entry.entries)}"
            )
            if score > best_score:
                best_score = score
                best_name = person_name
                best_crop = match_crop
                result.db_id = db_id

        result.best_score = best_score
        result.best_person_name = best_name
        result.score = best_score

        logger.debug(
            f"Re-ID gallery match_new_track: best={best_name}, score={best_score:.3f}, "
            f"threshold={self.similarity_threshold}"
        )

        if best_name and best_score > self.similarity_threshold:
            result.person_name = best_name
            result.matched = True
            result.gallery_crop = best_crop  # This is now the actual best matching crop

            # Update gallery timestamp (DON'T remove - person might leave and return)
            self._gallery[best_name].last_seen = time.time()

            logger.info(f"Re-ID match: {best_name} (score={best_score:.2f})")

            # Save debug image
            if self.debug_saver and best_crop is not None:
                self.debug_saver.save_reid_match(best_crop, crop, best_name, best_score)

        return result

    def get_reid_score(
        self,
        crop: np.ndarray,
        num_persons_in_frame: int = 1,
    ) -> float:
        """Get best Re-ID score for a crop without committing to a match.

        Used for display purposes to show current Re-ID confidence.

        Args:
            crop: Person crop image
            num_persons_in_frame: Number of persons in frame

        Returns:
            Best similarity score, or -1 if no gallery or multiple persons
        """
        if num_persons_in_frame > 1:
            return -1.0

        if not self._gallery:
            return -1.0

        embedding, quality = self.reid_extractor.extract(crop, return_quality=True)
        if quality < 0.3:
            return -1.0

        best_score = 0.0
        for entry in self._gallery.values():
            score, _, _ = entry.match(embedding)
            if score > best_score:
                best_score = score

        return best_score


class DebugImageSaver:
    """Saves debug images for Re-ID and face recognition events."""

    def __init__(self, debug_dir: str = "data/debug/reid"):
        """Initialize the debug image saver.

        Args:
            debug_dir: Directory to save debug images
        """
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    def _get_filename(self, prefix: str, label: str = "") -> Path:
        """Generate a filename with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        safe_label = label.replace(" ", "_").replace("/", "-")
        if safe_label:
            filename = f"{prefix}_{timestamp}_{safe_label}.jpg"
        else:
            filename = f"{prefix}_{timestamp}.jpg"
        return self.debug_dir / filename

    def save_face_recognized(
        self,
        crop: np.ndarray,
        face_bbox: tuple[int, int, int, int],
        person_name: str,
        confidence: float,
    ) -> Path:
        """Save debug image when face is recognized.

        Args:
            crop: Person crop image
            face_bbox: Face bounding box (x1, y1, x2, y2)
            person_name: Recognized person name
            confidence: Recognition confidence

        Returns:
            Path to saved image
        """
        img = crop.copy()
        fx1, fy1, fx2, fy2 = face_bbox
        cv2.rectangle(img, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
        cv2.putText(
            img, f"{person_name} ({confidence:.2f})",
            (fx1, fy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

        path = self._get_filename("face_recognized", person_name)
        cv2.imwrite(str(path), img)
        logger.debug(f"Saved face recognition debug image: {path}")
        return path

    def save_gallery_stored(
        self,
        crop: np.ndarray,
        person_name: str,
        num_embeddings: int,
    ) -> Path:
        """Save debug image when Re-ID gallery is updated.

        Args:
            crop: Person crop image
            person_name: Person name
            num_embeddings: Number of embeddings stored

        Returns:
            Path to saved image
        """
        img = crop.copy()
        cv2.putText(
            img, f"{person_name} ({num_embeddings} emb)",
            (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

        path = self._get_filename("gallery_stored", person_name)
        cv2.imwrite(str(path), img)
        logger.debug(f"Saved gallery stored debug image: {path}")
        return path

    def save_reid_match(
        self,
        gallery_crop: np.ndarray,
        match_crop: np.ndarray,
        person_name: str,
        score: float,
    ) -> Path:
        """Save side-by-side debug image when Re-ID match is found.

        Args:
            gallery_crop: Original gallery crop
            match_crop: Matched person crop
            person_name: Person name
            score: Match score

        Returns:
            Path to saved image
        """
        combined = self._create_side_by_side(
            gallery_crop, match_crop,
            f"Gallery: {person_name}", f"Match: {score:.2f}"
        )

        path = self._get_filename("reid_match", person_name)
        cv2.imwrite(str(path), combined)
        logger.debug(f"Saved Re-ID match debug image: {path}")
        return path

    def _create_side_by_side(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        label1: str,
        label2: str,
    ) -> np.ndarray:
        """Create a side-by-side comparison image."""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        target_h = max(h1, h2, 200)

        # Resize to same height
        if h1 != target_h:
            scale = target_h / h1
            img1 = cv2.resize(img1, (int(w1 * scale), target_h))
        if h2 != target_h:
            scale = target_h / h2
            img2 = cv2.resize(img2, (int(w2 * scale), target_h))

        # Add labels
        img1_labeled = img1.copy()
        img2_labeled = img2.copy()
        cv2.putText(img1_labeled, label1, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img2_labeled, label2, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Concatenate with separator
        separator = np.zeros((target_h, 5, 3), dtype=np.uint8)
        separator[:] = (255, 255, 255)
        combined = np.hstack([img1_labeled, separator, img2_labeled])

        return combined
