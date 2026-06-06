"""Database repository for CRUD operations."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.database.models import (
    Camera,
    CameraOverlap,
    Event,
    FaceEmbedding,
    Person,
    Track,
    TrackSighting,
    UnidentifiedFace,
    ReIDEmbedding,
    init_database,
)

logger = logging.getLogger(__name__)


class Repository:
    """Database repository for all CRUD operations."""

    def __init__(self, db_path: str):
        """Initialize repository.

        Args:
            db_path: Path to SQLite database or ":memory:"
        """
        if db_path != ":memory:":
            # Ensure directory exists
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._engine, self._session_maker = init_database(db_path)
        if db_path == ":memory:":
            logger.info("In-memory database initialized")
        else:
            logger.info(f"Database initialized at {db_path}")

    def get_session(self) -> Session:
        """Get a new database session."""
        return self._session_maker()

    def checkpoint(self):
        """Force WAL checkpoint to flush all pending writes to the main database file.

        Call this on shutdown to prevent corruption if the process is killed.
        """
        try:
            with self._engine.connect() as conn:
                conn.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
                conn.commit()
            logger.info("Database WAL checkpoint completed")
        except Exception as e:
            logger.error(f"WAL checkpoint failed: {e}")

    # ==================== Person Operations ====================

    def create_person(self, name: str) -> Person:
        """Create a new person.

        Args:
            name: Person's name

        Returns:
            Created Person object
        """
        with self.get_session() as session:
            person = Person(name=name)
            session.add(person)
            session.commit()
            session.refresh(person)
            logger.info(f"Created person: {person.id} - {name}")
            return person

    def get_person(self, person_id: int) -> Optional[Person]:
        """Get person by ID."""
        with self.get_session() as session:
            return session.query(Person).filter(Person.id == person_id).first()

    def get_person_by_name(self, name: str) -> Optional[Person]:
        """Get person by name."""
        with self.get_session() as session:
            return session.query(Person).filter(Person.name == name).first()

    def get_all_persons(self, active_only: bool = True) -> list[Person]:
        """Get all persons."""
        with self.get_session() as session:
            query = session.query(Person)
            if active_only:
                query = query.filter(Person.is_active == True)
            return query.all()

    def update_person_zone(
        self, person_id: int, zone: Optional[str], is_estimated: bool
    ) -> Optional[tuple[Optional[str], bool]]:
        """Update the current zone state on a Person record.

        Args:
            person_id: Person ID
            zone: Zone name (or None to clear)
            is_estimated: True if person is no longer actively observed in this zone

        Returns:
            Tuple of (previous_zone, previous_is_estimated) if person was found
            and the zone actually changed, else None (no change or person not found).
        """
        with self.get_session() as session:
            person = session.query(Person).filter(Person.id == person_id).first()
            if not person:
                return None

            prev_zone = person.current_zone
            prev_estimated = person.zone_is_estimated

            # Skip write if nothing changed
            if prev_zone == zone and prev_estimated == is_estimated:
                return None

            person.current_zone = zone
            person.zone_updated_at = datetime.utcnow()
            person.zone_is_estimated = is_estimated
            session.commit()
            return (prev_zone, prev_estimated)

    def delete_person(self, person_id: int) -> bool:
        """Delete a person (soft delete by setting is_active=False)."""
        with self.get_session() as session:
            person = session.query(Person).filter(Person.id == person_id).first()
            if person:
                person.is_active = False
                session.commit()
                logger.info(f"Deleted person: {person_id}")
                return True
            return False

    # ==================== Face Embedding Operations ====================

    def add_face_embedding(
        self, person_id: int, embedding: np.ndarray, source_image: Optional[str] = None
    ) -> FaceEmbedding:
        """Add face embedding for a person.

        Args:
            person_id: Person ID
            embedding: 512-dim face embedding
            source_image: Path to source image

        Returns:
            Created FaceEmbedding object
        """
        with self.get_session() as session:
            # Convert embedding to bytes
            embedding_bytes = embedding.astype(np.float32).tobytes()

            face_emb = FaceEmbedding(
                person_id=person_id,
                embedding=embedding_bytes,
                source_image=source_image,
            )
            session.add(face_emb)
            session.commit()
            session.refresh(face_emb)
            logger.debug(f"Added face embedding for person {person_id}")
            return face_emb

    def get_face_embeddings(self, person_id: int) -> list[tuple[int, np.ndarray]]:
        """Get all face embeddings for a person.

        Returns:
            List of (embedding_id, embedding_array) tuples
        """
        with self.get_session() as session:
            embeddings = (
                session.query(FaceEmbedding)
                .filter(FaceEmbedding.person_id == person_id)
                .all()
            )
            result = []
            for emb in embeddings:
                arr = np.frombuffer(emb.embedding, dtype=np.float32)
                result.append((emb.id, arr))
            return result

    def get_all_face_embeddings(self) -> list[tuple[int, int, np.ndarray]]:
        """Get all face embeddings for all active persons.

        Returns:
            List of (person_id, embedding_id, embedding_array) tuples
        """
        with self.get_session() as session:
            embeddings = (
                session.query(FaceEmbedding)
                .join(Person)
                .filter(Person.is_active == True)
                .all()
            )
            result = []
            for emb in embeddings:
                arr = np.frombuffer(emb.embedding, dtype=np.float32)
                result.append((emb.person_id, emb.id, arr))
            return result

    def get_face_embedding(self, embedding_id: int) -> Optional[FaceEmbedding]:
        """Get face embedding by ID."""
        with self.get_session() as session:
            return session.query(FaceEmbedding).filter(FaceEmbedding.id == embedding_id).first()

    # ==================== Track Operations ====================

    def create_track(
        self, track_id: str, camera_id: str, person_id: Optional[int] = None
    ) -> Track:
        """Create a new global track.

        Args:
            track_id: Global track ID
            camera_id: Camera where track was created
            person_id: Optional linked person ID

        Returns:
            Created Track object
        """
        with self.get_session() as session:
            track = Track(
                id=track_id,
                person_id=person_id,
                last_camera_id=camera_id,
                status="active",
            )
            session.add(track)
            session.commit()
            session.refresh(track)
            logger.debug(f"Created track: {track_id}")
            return track

    def get_track(self, track_id: str) -> Optional[Track]:
        """Get track by ID."""
        with self.get_session() as session:
            return session.query(Track).filter(Track.id == track_id).first()

    def update_track(
        self,
        track_id: str,
        person_id: Optional[int] = None,
        camera_id: Optional[str] = None,
        status: Optional[str] = None,
        reid_embedding: Optional[np.ndarray] = None,
        face_embedding: Optional[np.ndarray] = None,
        extra_data: Optional[dict] = None,
    ) -> Optional[Track]:
        """Update track fields.

        Args:
            extra_data: If provided, merges into existing extra_data (does not replace).
        """
        with self.get_session() as session:
            track = session.query(Track).filter(Track.id == track_id).first()
            if not track:
                return None

            if person_id is not None:
                track.person_id = person_id
            if camera_id is not None:
                track.last_camera_id = camera_id
            if status is not None:
                track.status = status
            if reid_embedding is not None:
                track.reid_embedding = reid_embedding.astype(np.float32).tobytes()
            if face_embedding is not None:
                track.face_embedding = face_embedding.astype(np.float32).tobytes()
            if extra_data is not None:
                current = track.extra_data or {}
                current.update(extra_data)
                track.extra_data = current

            track.last_seen = datetime.utcnow()
            session.commit()
            session.refresh(track)
            return track

    def get_active_tracks(self) -> list[Track]:
        """Get all active tracks."""
        with self.get_session() as session:
            return session.query(Track).filter(Track.status == "active").all()

    def get_recent_tracks(self, hours: int = 24) -> list[Track]:
        """Get tracks seen within the last N hours."""
        with self.get_session() as session:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            return session.query(Track).filter(Track.last_seen >= cutoff).all()

    def archive_old_tracks(self, hours: int = 24) -> int:
        """Archive tracks older than N hours.

        Returns:
            Number of tracks archived
        """
        with self.get_session() as session:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            count = (
                session.query(Track)
                .filter(Track.last_seen < cutoff, Track.status == "active")
                .update({"status": "archived"})
            )
            session.commit()
            if count > 0:
                logger.info(f"Archived {count} old tracks")
            return count

    # ==================== Event Operations ====================

    def _to_json_safe(self, obj):
        """Recursively convert numpy types to native Python types for JSON.
        
        - np.generic scalars -> Python scalars
        - np.ndarray -> list
        - dict/list/tuple -> recurse
        - others returned as-is
        """
        import numpy as np
        if isinstance(obj, dict):
            return {self._to_json_safe(k): self._to_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_json_safe(x) for x in obj]
        # numpy scalars
        if isinstance(obj, np.generic):
            return obj.item()
        # numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def create_event(
        self,
        camera_id: str,
        event_type: str,
        track_id: Optional[str] = None,
        person_id: Optional[int] = None,
        face_embedding_id: Optional[int] = None,
        reid_embedding_id: Optional[int] = None,
        confidence: Optional[float] = None,
        snapshot_path: Optional[str] = None,
        extra_data: Optional[dict] = None,
    ) -> Event:
        """Create a new event.

        Args:
            camera_id: Camera where event occurred
            event_type: Type of event
            track_id: Associated track ID
            person_id: Associated person ID
            face_embedding_id: Associated face embedding ID
            reid_embedding_id: Associated Re-ID embedding ID
            confidence: Detection/recognition confidence
            snapshot_path: Path to snapshot image
            extra_data: Additional metadata

        Returns:
            Created Event object
        """
        with self.get_session() as session:
            # Sanitize JSON payloads and numeric values
            safe_extra = self._to_json_safe(extra_data or {})
            safe_conf = self._to_json_safe(confidence) if confidence is not None else None

            event = Event(
                camera_id=str(camera_id),
                event_type=str(event_type),
                track_id=str(track_id) if track_id is not None else None,
                person_id=person_id,
                face_embedding_id=face_embedding_id,
                reid_embedding_id=reid_embedding_id,
                confidence=safe_conf,
                snapshot_path=str(snapshot_path) if snapshot_path is not None else None,
                extra_data=safe_extra,
            )
            session.add(event)
            session.commit()
            session.refresh(event)
            logger.debug(f"Created event: {event_type} on {camera_id}")
            return event

    def get_events(
        self,
        camera_id: Optional[str] = None,
        person_id: Optional[int] = None,
        person_name: Optional[str] = None,
        event_type: Optional[str] = None,
        track_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Event]:
        """Query events with filters."""
        with self.get_session() as session:
            query = session.query(Event)

            if camera_id:
                query = query.filter(Event.camera_id == camera_id)
            if person_name and not person_id:
                from src.database.models import Person
                person = session.query(Person).filter(Person.name == person_name).first()
                person_id = person.id if person else -1
            if person_id:
                query = query.filter(Event.person_id == person_id)
            if event_type:
                query = query.filter(Event.event_type == event_type)
            if track_id:
                query = query.filter(Event.track_id == track_id)
            if since:
                query = query.filter(Event.timestamp >= since)

            return query.order_by(Event.timestamp.desc()).offset(offset).limit(limit).all()

    def cleanup_old_events(self, days: int = 30) -> int:
        """Delete events older than N days.

        Returns:
            Number of events deleted
        """
        with self.get_session() as session:
            cutoff = datetime.utcnow() - timedelta(days=days)
            count = session.query(Event).filter(Event.timestamp < cutoff).delete()
            session.commit()
            if count > 0:
                logger.info(f"Deleted {count} old events")
            return count

    # ==================== ReID Embedding Operations ====================

    def add_reid_embedding(
        self,
        camera_id: str,
        embedding: np.ndarray,
        track_id: Optional[str] = None,
        person_id: Optional[int] = None,
        quality: Optional[float] = None,
        visibility: Optional[float] = None,
        snapshot_path: Optional[str] = None,
    ) -> ReIDEmbedding:
        """Add a Re-ID embedding.

        Args:
            camera_id: Camera ID
            embedding: Re-ID embedding vector
            track_id: Optional track ID
            person_id: Optional person ID
            quality: Quality score
            visibility: Visibility score
            snapshot_path: Path to snapshot

        Returns:
            Created ReIDEmbedding object
        """
        with self.get_session() as session:
            reid_emb = ReIDEmbedding(
                camera_id=camera_id,
                track_id=track_id,
                person_id=person_id,
                embedding=embedding.astype(np.float32).tobytes(),
                quality=quality,
                visibility=visibility,
                snapshot_path=snapshot_path,
            )
            session.add(reid_emb)
            session.commit()
            session.refresh(reid_emb)
            return reid_emb

    def get_reid_embeddings(self, person_id: int, limit: int = 100) -> list[ReIDEmbedding]:
        """Get Re-ID embeddings for a person."""
        with self.get_session() as session:
            return (
                session.query(ReIDEmbedding)
                .filter(ReIDEmbedding.person_id == person_id)
                .order_by(ReIDEmbedding.timestamp.desc())
                .limit(limit)
                .all()
            )

    def cleanup_old_reid_embeddings(self, days: int = 30) -> int:
        """Delete Re-ID embeddings older than N days."""
        with self.get_session() as session:
            cutoff = datetime.utcnow() - timedelta(days=days)
            count = session.query(ReIDEmbedding).filter(ReIDEmbedding.timestamp < cutoff).delete()
            session.commit()
            return count

    # ==================== Track Sighting Operations ====================

    def create_track_sighting(
        self,
        track_id: str,
        camera_id: str,
        entry_zone: Optional[str] = None,
        snapshot_path: Optional[str] = None,
    ) -> TrackSighting:
        """Record a track sighting on a camera."""
        with self.get_session() as session:
            sighting = TrackSighting(
                track_id=track_id,
                camera_id=camera_id,
                entered_at=datetime.utcnow(),
                entry_zone=entry_zone,
                snapshot_path=snapshot_path,
            )
            session.add(sighting)
            session.commit()
            session.refresh(sighting)
            return sighting

    def end_track_sighting(
        self, track_id: str, camera_id: str, exit_zone: Optional[str] = None
    ) -> Optional[TrackSighting]:
        """Mark the end of a track sighting."""
        with self.get_session() as session:
            sighting = (
                session.query(TrackSighting)
                .filter(
                    TrackSighting.track_id == track_id,
                    TrackSighting.camera_id == camera_id,
                    TrackSighting.exited_at == None,
                )
                .first()
            )
            if sighting:
                sighting.exited_at = datetime.utcnow()
                sighting.exit_zone = exit_zone
                session.commit()
                session.refresh(sighting)
            return sighting

    # ==================== Unidentified Face Operations ====================

    def add_unidentified_face(
        self,
        camera_id: str,
        embedding: np.ndarray,
        image_path: str,
        quality_score: float,
        track_id: Optional[str] = None,
        best_match_person_id: Optional[int] = None,
        best_match_score: Optional[float] = None,
        blur_score: Optional[float] = None,
        face_size: Optional[int] = None,
    ) -> UnidentifiedFace:
        """Add an unidentified face for manual review.

        Args:
            camera_id: Camera where face was detected
            embedding: 512-dim face embedding
            image_path: Path to saved face image
            quality_score: Computed quality score (0-1)
            track_id: Optional track ID
            best_match_person_id: ID of best matching person (if any)
            best_match_score: Similarity score to best match
            blur_score: Laplacian variance (higher = sharper)
            face_size: Face bounding box width in pixels

        Returns:
            Created UnidentifiedFace object
        """
        with self.get_session() as session:
            embedding_bytes = embedding.astype(np.float32).tobytes()

            face = UnidentifiedFace(
                camera_id=camera_id,
                track_id=track_id,
                best_match_person_id=best_match_person_id,
                best_match_score=best_match_score,
                embedding=embedding_bytes,
                image_path=image_path,
                quality_score=quality_score,
                blur_score=blur_score,
                face_size=face_size,
            )
            session.add(face)
            session.commit()
            session.refresh(face)
            logger.debug(f"Added unidentified face {face.id} from {camera_id}")
            return face

    def get_unidentified_faces(
        self,
        camera_id: Optional[str] = None,
        reviewed: Optional[bool] = None,
        dismissed: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[UnidentifiedFace]:
        """Query unidentified faces with filters.

        Args:
            camera_id: Filter by camera
            reviewed: Filter by reviewed status
            dismissed: Filter by dismissed status
            limit: Maximum results
            offset: Skip first N results

        Returns:
            List of matching UnidentifiedFace objects
        """
        with self.get_session() as session:
            query = session.query(UnidentifiedFace)

            if camera_id:
                query = query.filter(UnidentifiedFace.camera_id == camera_id)
            if reviewed is not None:
                query = query.filter(UnidentifiedFace.reviewed == reviewed)
            if dismissed is not None:
                query = query.filter(UnidentifiedFace.dismissed == dismissed)

            return (
                query.order_by(UnidentifiedFace.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )

    def get_unidentified_face(self, face_id: int) -> Optional[UnidentifiedFace]:
        """Get unidentified face by ID."""
        with self.get_session() as session:
            return (
                session.query(UnidentifiedFace)
                .filter(UnidentifiedFace.id == face_id)
                .first()
            )

    def get_unidentified_face_embedding(self, face_id: int) -> Optional[np.ndarray]:
        """Get embedding for an unidentified face.

        Returns:
            Embedding as numpy array, or None if not found
        """
        with self.get_session() as session:
            face = (
                session.query(UnidentifiedFace)
                .filter(UnidentifiedFace.id == face_id)
                .first()
            )
            if face:
                return np.frombuffer(face.embedding, dtype=np.float32)
            return None

    def get_recent_unidentified_embeddings(
        self, camera_id: str, limit: int = 20
    ) -> list[np.ndarray]:
        """Get recent embeddings for diversity check.

        Args:
            camera_id: Camera to check
            limit: Maximum embeddings to return

        Returns:
            List of embedding arrays
        """
        with self.get_session() as session:
            faces = (
                session.query(UnidentifiedFace)
                .filter(
                    UnidentifiedFace.camera_id == camera_id,
                    UnidentifiedFace.dismissed == False,
                )
                .order_by(UnidentifiedFace.created_at.desc())
                .limit(limit)
                .all()
            )
            return [np.frombuffer(f.embedding, dtype=np.float32) for f in faces]

    def update_unidentified_face(
        self,
        face_id: int,
        reviewed: Optional[bool] = None,
        dismissed: Optional[bool] = None,
    ) -> Optional[UnidentifiedFace]:
        """Update unidentified face status.

        Args:
            face_id: Face ID
            reviewed: Mark as reviewed
            dismissed: Mark as dismissed

        Returns:
            Updated UnidentifiedFace or None if not found
        """
        with self.get_session() as session:
            face = (
                session.query(UnidentifiedFace)
                .filter(UnidentifiedFace.id == face_id)
                .first()
            )
            if not face:
                return None

            if reviewed is not None:
                face.reviewed = reviewed
            if dismissed is not None:
                face.dismissed = dismissed

            session.commit()
            session.refresh(face)
            return face

    def delete_unidentified_face(self, face_id: int) -> bool:
        """Delete an unidentified face.

        Args:
            face_id: Face ID to delete

        Returns:
            True if deleted, False if not found
        """
        with self.get_session() as session:
            face = (
                session.query(UnidentifiedFace)
                .filter(UnidentifiedFace.id == face_id)
                .first()
            )
            if face:
                session.delete(face)
                session.commit()
                logger.debug(f"Deleted unidentified face {face_id}")
                return True
            return False

    def cleanup_unidentified_faces_for_camera(
        self, camera_id: str, max_count: int
    ) -> int:
        """Keep only the newest N unidentified faces per camera.

        Args:
            camera_id: Camera ID
            max_count: Maximum faces to keep

        Returns:
            Number of faces deleted
        """
        with self.get_session() as session:
            # Get faces to keep (newest first)
            keep_faces = (
                session.query(UnidentifiedFace.id)
                .filter(UnidentifiedFace.camera_id == camera_id)
                .order_by(UnidentifiedFace.created_at.desc())
                .limit(max_count)
                .all()
            )
            keep_ids = [f.id for f in keep_faces]

            # Delete older faces
            if keep_ids:
                count = (
                    session.query(UnidentifiedFace)
                    .filter(
                        UnidentifiedFace.camera_id == camera_id,
                        ~UnidentifiedFace.id.in_(keep_ids),
                    )
                    .delete(synchronize_session=False)
                )
            else:
                # Keep none - delete all
                count = (
                    session.query(UnidentifiedFace)
                    .filter(UnidentifiedFace.camera_id == camera_id)
                    .delete()
                )

            session.commit()
            if count > 0:
                logger.info(f"Cleaned up {count} old unidentified faces from {camera_id}")
            return count

    def cleanup_old_unidentified_faces(self, days: int = 30) -> int:
        """Delete unidentified faces older than N days.

        Args:
            days: Days to retain

        Returns:
            Number of faces deleted
        """
        with self.get_session() as session:
            cutoff = datetime.utcnow() - timedelta(days=days)
            count = (
                session.query(UnidentifiedFace)
                .filter(UnidentifiedFace.created_at < cutoff)
                .delete()
            )
            session.commit()
            if count > 0:
                logger.info(f"Deleted {count} old unidentified faces")
            return count

    def get_unidentified_faces_summary(self) -> dict[str, int]:
        """Get count of unidentified faces by camera.

        Returns:
            Dictionary of camera_id -> count
        """
        with self.get_session() as session:
            from sqlalchemy import func

            results = (
                session.query(
                    UnidentifiedFace.camera_id,
                    func.count(UnidentifiedFace.id).label("count"),
                )
                .filter(UnidentifiedFace.dismissed == False)
                .group_by(UnidentifiedFace.camera_id)
                .all()
            )
            return {r.camera_id: r.count for r in results}

    def assign_unidentified_face_to_person(
        self, face_id: int, person_id: int, new_image_path: Optional[str] = None
    ) -> Optional[FaceEmbedding]:
        """Assign an unidentified face to a person.

        Moves the embedding to the person's face gallery.

        Args:
            face_id: Unidentified face ID
            person_id: Person to assign to
            new_image_path: Optional new path for the source image

        Returns:
            Created FaceEmbedding or None if face not found
        """
        with self.get_session() as session:
            face = (
                session.query(UnidentifiedFace)
                .filter(UnidentifiedFace.id == face_id)
                .first()
            )
            if not face:
                return None

            # Get embedding as numpy array
            embedding = np.frombuffer(face.embedding, dtype=np.float32)

            # Create face embedding for person
            face_emb = FaceEmbedding(
                person_id=person_id,
                embedding=face.embedding,  # Already bytes
                source_image=new_image_path or face.image_path,
            )
            session.add(face_emb)

            # Mark as reviewed and delete the unidentified face entry
            session.delete(face)
            session.commit()
            session.refresh(face_emb)

            logger.info(
                f"Assigned unidentified face {face_id} to person {person_id}"
            )
            return face_emb
