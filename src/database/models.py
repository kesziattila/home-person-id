"""SQLAlchemy database models."""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()


class Person(Base):
    """Known person (enrolled identity)."""

    __tablename__ = "persons"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    # Relationships
    face_embeddings = relationship("FaceEmbedding", back_populates="person", cascade="all, delete-orphan")
    reid_embeddings = relationship("ReIDEmbedding", back_populates="person", cascade="all, delete-orphan")
    events = relationship("Event", back_populates="person")
    tracks = relationship("Track", back_populates="person")

    def __repr__(self):
        return f"<Person(id={self.id}, name='{self.name}')>"


class FaceEmbedding(Base):
    """Face embedding for a person (multiple per person for accuracy)."""

    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=False)
    embedding = Column(LargeBinary, nullable=False)  # 512-dim float32 vector
    source_image = Column(String(512))  # Original image path
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    person = relationship("Person", back_populates="face_embeddings")
    events = relationship("Event", back_populates="face_embedding")

    def __repr__(self):
        return f"<FaceEmbedding(id={self.id}, person_id={self.person_id})>"


class ReIDEmbedding(Base):
    """Re-ID embedding for a person."""

    __tablename__ = "reid_embeddings"

    id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=True)
    track_id = Column(String(64), ForeignKey("tracks.id"), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    camera_id = Column(String(64), nullable=False)
    embedding = Column(LargeBinary, nullable=False)
    quality = Column(Float, nullable=True)
    visibility = Column(Float, nullable=True)
    snapshot_path = Column(String(512), nullable=True)

    # Relationships
    person = relationship("Person", back_populates="reid_embeddings")
    track = relationship("Track", back_populates="reid_embeddings")
    events = relationship("Event", back_populates="reid_embedding")

    def __repr__(self):
        return f"<ReIDEmbedding(id={self.id}, person_id={self.person_id}, track_id='{self.track_id}')>"


class Track(Base):
    """Global track (cross-camera person tracking)."""

    __tablename__ = "tracks"

    id = Column(String(64), primary_key=True)  # "global_42"
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=True)
    reid_embedding = Column(LargeBinary, nullable=True)  # Latest Re-ID embedding
    face_embedding = Column(LargeBinary, nullable=True)  # Face embedding if captured
    first_seen = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_seen = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_camera_id = Column(String(64))
    status = Column(String(32), default="active")  # 'active', 'lost', 'archived'
    extra_data = Column(JSON, default=dict)  # Additional metadata (cameras_seen, etc.)

    # Relationships
    person = relationship("Person", back_populates="tracks")
    sightings = relationship("TrackSighting", back_populates="track", cascade="all, delete-orphan")
    events = relationship("Event", back_populates="track")
    reid_embeddings = relationship("ReIDEmbedding", back_populates="track")

    def __repr__(self):
        return f"<Track(id='{self.id}', person_id={self.person_id}, status='{self.status}')>"


class TrackSighting(Base):
    """Track sighting on a specific camera."""

    __tablename__ = "track_sightings"

    id = Column(Integer, primary_key=True)
    track_id = Column(String(64), ForeignKey("tracks.id"), nullable=False)
    camera_id = Column(String(64), nullable=False)
    entered_at = Column(DateTime, nullable=False)
    exited_at = Column(DateTime, nullable=True)
    entry_zone = Column(String(64))  # Where in frame they entered
    exit_zone = Column(String(64))  # Where in frame they exited
    snapshot_path = Column(String(512))

    # Relationships
    track = relationship("Track", back_populates="sightings")

    def __repr__(self):
        return f"<TrackSighting(track_id='{self.track_id}', camera_id='{self.camera_id}')>"


class Event(Base):
    """Detection/recognition event (activity log)."""

    __tablename__ = "events"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    camera_id = Column(String(64), nullable=False)
    event_type = Column(String(64), nullable=False)  # 'track_created', 'person_identified', etc.
    track_id = Column(String(64), ForeignKey("tracks.id"), nullable=True)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=True)
    face_embedding_id = Column(Integer, ForeignKey("face_embeddings.id"), nullable=True)
    reid_embedding_id = Column(Integer, ForeignKey("reid_embeddings.id"), nullable=True)
    confidence = Column(Float)
    snapshot_path = Column(String(512))
    extra_data = Column(JSON, default=dict)  # Additional event metadata

    # Relationships
    track = relationship("Track", back_populates="events")
    person = relationship("Person", back_populates="events")
    face_embedding = relationship("FaceEmbedding", back_populates="events")
    reid_embedding = relationship("ReIDEmbedding", back_populates="events")

    def __repr__(self):
        return f"<Event(id={self.id}, type='{self.event_type}', camera='{self.camera_id}')>"


class Camera(Base):
    """Camera configuration (stored in DB for persistence)."""

    __tablename__ = "cameras"

    id = Column(String(64), primary_key=True)
    name = Column(String(255), nullable=False)
    rtsp_url = Column(String(512), nullable=False)
    is_active = Column(Boolean, default=True)
    config = Column(JSON, default=dict)  # fps, resolution, detection zones

    def __repr__(self):
        return f"<Camera(id='{self.id}', name='{self.name}')>"


class CameraOverlap(Base):
    """Camera topology (overlaps for handover)."""

    __tablename__ = "camera_overlaps"

    id = Column(Integer, primary_key=True)
    camera1_id = Column(String(64), nullable=False)
    camera2_id = Column(String(64), nullable=False)
    cam1_exit_zone = Column(JSON)  # [x1, y1, x2, y2] normalized
    cam2_entry_zone = Column(JSON)  # [x1, y1, x2, y2] normalized
    max_handover_sec = Column(Float, default=3.0)

    def __repr__(self):
        return f"<CameraOverlap(cam1='{self.camera1_id}', cam2='{self.camera2_id}')>"


class UnidentifiedFace(Base):
    """Unidentified face detection for manual review."""

    __tablename__ = "unidentified_faces"

    id = Column(Integer, primary_key=True)
    camera_id = Column(String(64), nullable=False)
    track_id = Column(String(64), nullable=True)

    # Best match info
    best_match_person_id = Column(Integer, ForeignKey("persons.id"), nullable=True)
    best_match_score = Column(Float, nullable=True)

    # Face data
    embedding = Column(LargeBinary, nullable=False)  # 512-dim float32 vector
    image_path = Column(String(512), nullable=False)

    # Quality metrics
    quality_score = Column(Float, nullable=False)
    blur_score = Column(Float, nullable=True)
    face_size = Column(Integer, nullable=True)  # Face bbox width in pixels

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    reviewed = Column(Boolean, default=False)
    dismissed = Column(Boolean, default=False)

    # Relationships
    best_match_person = relationship("Person")

    def __repr__(self):
        return f"<UnidentifiedFace(id={self.id}, camera='{self.camera_id}', score={self.best_match_score})>"


def init_database(db_path: str) -> tuple:
    """Initialize database and return engine and session maker.

    Args:
        db_path: Path to SQLite database file (or ":memory:")

    Returns:
        Tuple of (engine, SessionLocal)
    """
    if db_path == ":memory:":
        url = "sqlite://"
    else:
        url = f"sqlite:///{db_path}"

    # Configure SQLite for multi-threaded access:
    # - check_same_thread=False: Allow connections from any thread
    # - SQLite's WAL mode provides better concurrency for readers/writers
    connect_args = {"check_same_thread": False}
    engine = create_engine(url, echo=False, connect_args=connect_args)

    # Enable WAL mode for better concurrent access
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
        conn.execute(text("PRAGMA busy_timeout=5000"))  # Wait up to 5s on locks
        conn.execute(text("PRAGMA synchronous=NORMAL"))  # Crash-safe with WAL
        conn.commit()

    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, SessionLocal
