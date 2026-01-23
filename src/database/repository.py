"""Database repository for CRUD operations."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

from src.database.models import (
    Camera,
    CameraOverlap,
    Event,
    FaceEmbedding,
    Person,
    Track,
    TrackSighting,
    init_database,
)

logger = logging.getLogger(__name__)


class Repository:
    """Database repository for all CRUD operations."""

    def __init__(self, db_path: str):
        """Initialize repository.

        Args:
            db_path: Path to SQLite database
        """
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._engine, self._session_maker = init_database(db_path)
        logger.info(f"Database initialized at {db_path}")

    def get_session(self) -> Session:
        """Get a new database session."""
        return self._session_maker()

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
    ) -> Optional[Track]:
        """Update track fields."""
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

    def create_event(
        self,
        camera_id: str,
        event_type: str,
        track_id: Optional[str] = None,
        person_id: Optional[int] = None,
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
            confidence: Detection/recognition confidence
            snapshot_path: Path to snapshot image
            extra_data: Additional metadata

        Returns:
            Created Event object
        """
        with self.get_session() as session:
            event = Event(
                camera_id=camera_id,
                event_type=event_type,
                track_id=track_id,
                person_id=person_id,
                confidence=confidence,
                snapshot_path=snapshot_path,
                extra_data=extra_data or {},
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
        event_type: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[Event]:
        """Query events with filters."""
        with self.get_session() as session:
            query = session.query(Event)

            if camera_id:
                query = query.filter(Event.camera_id == camera_id)
            if person_id:
                query = query.filter(Event.person_id == person_id)
            if event_type:
                query = query.filter(Event.event_type == event_type)
            if since:
                query = query.filter(Event.timestamp >= since)

            return query.order_by(Event.timestamp.desc()).limit(limit).all()

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
