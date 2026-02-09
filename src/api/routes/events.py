"""API routes for events, persons, and location tracking."""

import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.database.repository import Repository

logger = logging.getLogger(__name__)


# Response models
class PersonResponse(BaseModel):
    id: int
    name: str
    face_count: int
    is_active: bool
    created_at: datetime
    last_seen: Optional[datetime] = None


class EventResponse(BaseModel):
    id: int
    timestamp: datetime
    camera_id: str
    event_type: str
    track_id: Optional[str] = None
    person_id: Optional[int] = None
    person_name: Optional[str] = None
    face_embedding_id: Optional[int] = None
    reid_embedding_id: Optional[int] = None
    confidence: Optional[float] = None
    snapshot_path: Optional[str] = None
    extra_data: Optional[dict] = None


class LocationResponse(BaseModel):
    person_id: int
    person_name: str
    current_camera_id: Optional[str] = None
    current_zone: Optional[str] = None
    last_seen: Optional[datetime] = None
    status: str  # 'active', 'lost', 'archived'


class TrackSightingResponse(BaseModel):
    camera_id: str
    entered_at: datetime
    exited_at: Optional[datetime] = None
    entry_zone: Optional[str] = None
    exit_zone: Optional[str] = None
    duration_sec: Optional[float] = None


class TimelineResponse(BaseModel):
    person_id: int
    person_name: str
    sightings: List[TrackSightingResponse]


class ActiveTrackResponse(BaseModel):
    track_id: str
    person_id: Optional[int] = None
    person_name: Optional[str] = None
    camera_id: str
    first_seen: datetime
    last_seen: datetime
    duration_sec: float


def create_events_router(repository: Repository) -> APIRouter:
    """Create events router with repository dependency.

    Args:
        repository: Database repository instance

    Returns:
        Configured APIRouter
    """
    router = APIRouter(prefix="/api/v1", tags=["events"])

    @router.get("/events", response_model=List[EventResponse])
    async def get_events(
        camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
        person_id: Optional[int] = Query(None, description="Filter by person ID"),
        event_type: Optional[str] = Query(None, description="Filter by event type"),
        track_id: Optional[str] = Query(None, description="Filter by track ID"),
        since_hours: int = Query(24, description="Hours to look back", ge=1, le=720),
        limit: int = Query(50, description="Maximum events to return", ge=1, le=1000),
        offset: int = Query(0, description="Offset for pagination", ge=0),
    ) -> List[EventResponse]:
        """Get recent events with optional filters."""
        since = datetime.utcnow() - timedelta(hours=since_hours)

        events = repository.get_events(
            camera_id=camera_id,
            person_id=person_id,
            event_type=event_type,
            track_id=track_id,
            since=since,
            limit=limit,
            offset=offset,
        )

        result = []
        for event in events:
            person_name = None
            if event.person_id:
                person = repository.get_person(event.person_id)
                person_name = person.name if person else None

            result.append(EventResponse(
                id=event.id,
                timestamp=event.timestamp,
                camera_id=event.camera_id,
                event_type=event.event_type,
                track_id=event.track_id,
                person_id=event.person_id,
                person_name=person_name,
                face_embedding_id=event.face_embedding_id,
                reid_embedding_id=event.reid_embedding_id,
                confidence=event.confidence,
                snapshot_path=event.snapshot_path,
                extra_data=event.extra_data,
            ))

        return result

    @router.get("/events/{event_id}/snapshot")
    async def get_event_snapshot(event_id: int):
        """Get the snapshot for an event."""
        with repository.get_session() as session:
            from src.database.models import Event
            event = session.query(Event).filter(Event.id == event_id).first()
            if not event or not event.snapshot_path:
                raise HTTPException(status_code=404, detail="Snapshot not found")
            
            if not os.path.exists(event.snapshot_path):
                raise HTTPException(status_code=404, detail="Snapshot file not found")
            
            return FileResponse(event.snapshot_path, media_type="image/jpeg")

    @router.get("/reid-embeddings/{embedding_id}/image")
    async def get_reid_embedding_image(embedding_id: int):
        """Get the snapshot for a Re-ID embedding."""
        with repository.get_session() as session:
            from src.database.models import ReIDEmbedding
            embedding = session.query(ReIDEmbedding).filter(ReIDEmbedding.id == embedding_id).first()
            if not embedding or not embedding.snapshot_path:
                raise HTTPException(status_code=404, detail="Snapshot not found")
            
            if not os.path.exists(embedding.snapshot_path):
                raise HTTPException(status_code=404, detail="Snapshot file not found")
            
            return FileResponse(embedding.snapshot_path, media_type="image/jpeg")

    @router.get("/persons", response_model=List[PersonResponse])
    async def get_persons(
        active_only: bool = Query(True, description="Show only active persons"),
    ) -> List[PersonResponse]:
        """Get all registered persons."""
        persons = repository.get_all_persons(active_only=active_only)

        result = []
        for person in persons:
            embeddings = repository.get_face_embeddings(person.id)

            # Get most recent event for this person to find last_seen
            recent_events = repository.get_events(person_id=person.id, limit=1)
            last_seen = recent_events[0].timestamp if recent_events else None

            result.append(PersonResponse(
                id=person.id,
                name=person.name,
                face_count=len(embeddings),
                is_active=person.is_active,
                created_at=person.created_at,
                last_seen=last_seen,
            ))

        return result

    @router.get("/persons/{person_id}/location", response_model=LocationResponse)
    async def get_person_location(person_id: int) -> LocationResponse:
        """Get current location of a person."""
        person = repository.get_person(person_id)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")

        # Find most recent active track for this person
        active_tracks = repository.get_active_tracks()
        person_track = None
        for track in active_tracks:
            if track.person_id == person_id:
                person_track = track
                break

        if person_track:
            return LocationResponse(
                person_id=person.id,
                person_name=person.name,
                current_camera_id=person_track.last_camera_id,
                current_zone=person_track.extra_data.get("zone") if person_track.extra_data else None,
                last_seen=person_track.last_seen,
                status=person_track.status,
            )
        else:
            # Not currently active, check recent tracks
            with repository.get_session() as session:
                from src.database.models import Track
                recent_track = (
                    session.query(Track)
                    .filter(Track.person_id == person_id)
                    .order_by(Track.last_seen.desc())
                    .first()
                )

                if recent_track:
                    return LocationResponse(
                        person_id=person.id,
                        person_name=person.name,
                        current_camera_id=recent_track.last_camera_id,
                        current_zone=recent_track.extra_data.get("zone") if recent_track.extra_data else None,
                        last_seen=recent_track.last_seen,
                        status=recent_track.status,
                    )
                else:
                    return LocationResponse(
                        person_id=person.id,
                        person_name=person.name,
                        status="never_seen",
                    )

    @router.get("/persons/{person_id}/timeline", response_model=TimelineResponse)
    async def get_person_timeline(
        person_id: int,
        since_hours: int = Query(24, description="Hours to look back", ge=1, le=720),
    ) -> TimelineResponse:
        """Get movement timeline for a person."""
        person = repository.get_person(person_id)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")

        since = datetime.utcnow() - timedelta(hours=since_hours)

        # Get all tracks for this person in the time range
        with repository.get_session() as session:
            from src.database.models import Track, TrackSighting

            tracks = (
                session.query(Track)
                .filter(Track.person_id == person_id, Track.last_seen >= since)
                .all()
            )

            sightings = []
            for track in tracks:
                track_sightings = (
                    session.query(TrackSighting)
                    .filter(TrackSighting.track_id == track.id)
                    .order_by(TrackSighting.entered_at.desc())
                    .all()
                )

                for sighting in track_sightings:
                    duration_sec = None
                    if sighting.exited_at:
                        duration_sec = (sighting.exited_at - sighting.entered_at).total_seconds()

                    sightings.append(TrackSightingResponse(
                        camera_id=sighting.camera_id,
                        entered_at=sighting.entered_at,
                        exited_at=sighting.exited_at,
                        entry_zone=sighting.entry_zone,
                        exit_zone=sighting.exit_zone,
                        duration_sec=duration_sec,
                    ))

            # Sort by entry time descending
            sightings.sort(key=lambda s: s.entered_at, reverse=True)

            return TimelineResponse(
                person_id=person.id,
                person_name=person.name,
                sightings=sightings,
            )

    @router.get("/tracks/active", response_model=List[ActiveTrackResponse])
    async def get_active_tracks() -> List[ActiveTrackResponse]:
        """Get all currently active tracks."""
        tracks = repository.get_active_tracks()

        result = []
        for track in tracks:
            person_name = None
            if track.person_id:
                person = repository.get_person(track.person_id)
                person_name = person.name if person else None

            duration_sec = (track.last_seen - track.first_seen).total_seconds()

            result.append(ActiveTrackResponse(
                track_id=track.id,
                person_id=track.person_id,
                person_name=person_name,
                camera_id=track.last_camera_id,
                first_seen=track.first_seen,
                last_seen=track.last_seen,
                duration_sec=duration_sec,
            ))

        return result

    return router
