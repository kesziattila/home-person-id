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
    zone_is_estimated: Optional[bool] = None
    zone_updated_at: Optional[datetime] = None
    estimated_location: Optional[str] = None
    time_since_seen_sec: Optional[float] = None
    last_seen: Optional[datetime] = None
    status: str  # 'active', 'lost', 'archived'


class StayResponse(BaseModel):
    zone: Optional[str] = None
    camera_id: str
    entered_at: datetime
    exited_at: Optional[datetime] = None  # null => current stay
    duration_sec: Optional[float] = None
    is_estimated: bool = False
    is_current: bool = False
    snapshot_event_id: Optional[int] = None  # feed to /events/{id}/snapshot


class TimelineResponse(BaseModel):
    person_id: int
    person_name: str
    segments: List[StayResponse]


class ActiveTrackResponse(BaseModel):
    track_id: str
    person_id: Optional[int] = None
    person_name: Optional[str] = None
    camera_id: str
    zone: Optional[str] = None
    first_seen: datetime
    last_seen: datetime
    duration_sec: float


def create_events_router(
    repository: Repository, timeline_min_stay_sec: float = 5.0
) -> APIRouter:
    """Create events router with repository dependency.

    Args:
        repository: Database repository instance
        timeline_min_stay_sec: Merge/hide movement-timeline zone stays shorter
            than this many seconds (0 disables flicker merging)

    Returns:
        Configured APIRouter
    """
    router = APIRouter(prefix="/api/v1", tags=["events"])

    @router.get("/events", response_model=List[EventResponse])
    async def get_events(
        camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
        person_id: Optional[int] = Query(None, description="Filter by person ID"),
        person_name: Optional[str] = Query(None, description="Filter by person name"),
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
            person_name=person_name,
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

    @router.get("/events/{event_id}/from_snapshot")
    async def get_event_from_snapshot(event_id: int):
        """Get the source camera crop for cross_camera_propagation events."""
        with repository.get_session() as session:
            from src.database.models import Event
            event = session.query(Event).filter(Event.id == event_id).first()
            if not event or not event.extra_data:
                raise HTTPException(status_code=404, detail="Event not found or has no extra data")

            from_snapshot_path = event.extra_data.get("from_snapshot_path")
            if not from_snapshot_path:
                raise HTTPException(status_code=404, detail="No source snapshot for this event")

            if not os.path.exists(from_snapshot_path):
                raise HTTPException(status_code=404, detail="Source snapshot file not found")

            return FileResponse(from_snapshot_path, media_type="image/jpeg")

    @router.get("/events/{event_id}/camera_snapshot/{camera_id}")
    async def get_house_overview_camera_snapshot(event_id: int, camera_id: str):
        """Get a camera frame snapshot stored in a house_overview event's extra_data."""
        with repository.get_session() as session:
            from src.database.models import Event
            event = session.query(Event).filter(Event.id == event_id).first()
            if not event or not event.extra_data:
                raise HTTPException(status_code=404, detail="Event not found")

            paths = event.extra_data.get("camera_snapshot_paths", {})
            path = paths.get(camera_id)
            if not path:
                raise HTTPException(status_code=404, detail="No snapshot for this camera")

            if not os.path.exists(path):
                raise HTTPException(status_code=404, detail="Snapshot file not found")

            return FileResponse(path, media_type="image/jpeg")

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
        """Get current location of a person.

        Zone information is read directly from the Person record (always up-to-date).
        Active track info (camera, last_seen) still comes from tracks.
        """
        person = repository.get_person(person_id)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")

        # Zone state comes directly from Person
        current_zone = person.current_zone
        zone_is_estimated = person.zone_is_estimated
        zone_updated_at = person.zone_updated_at

        # Find most recent active track for camera/last_seen
        active_tracks = repository.get_active_tracks()
        person_track = None
        for track in active_tracks:
            if track.person_id == person_id:
                person_track = track
                break

        if person_track:
            extra = person_track.extra_data or {}
            # Use Person zone if available, else fall back to track extra_data
            effective_zone = current_zone or extra.get("zone")
            time_since = (
                (datetime.utcnow() - person_track.last_seen).total_seconds()
                if person_track.last_seen else None
            )
            return LocationResponse(
                person_id=person.id,
                person_name=person.name,
                current_camera_id=person_track.last_camera_id,
                current_zone=effective_zone,
                zone_is_estimated=zone_is_estimated,
                zone_updated_at=zone_updated_at,
                estimated_location=effective_zone,
                time_since_seen_sec=time_since,
                last_seen=person_track.last_seen,
                status=person_track.status,
            )
        else:
            # Not currently active, check recent tracks for last_seen/camera
            with repository.get_session() as session:
                from src.database.models import Track
                recent_track = (
                    session.query(Track)
                    .filter(Track.person_id == person_id)
                    .order_by(Track.last_seen.desc())
                    .first()
                )

                if recent_track:
                    extra = recent_track.extra_data or {}
                    # Use Person zone if available, else fall back to track extra_data
                    effective_zone = current_zone or extra.get("zone")
                    estimated = current_zone or extra.get("estimated_zone", effective_zone)
                    time_since = (
                        (datetime.utcnow() - recent_track.last_seen).total_seconds()
                        if recent_track.last_seen else None
                    )
                    return LocationResponse(
                        person_id=person.id,
                        person_name=person.name,
                        current_camera_id=recent_track.last_camera_id,
                        current_zone=effective_zone,
                        zone_is_estimated=zone_is_estimated,
                        zone_updated_at=zone_updated_at,
                        estimated_location=estimated,
                        time_since_seen_sec=time_since,
                        last_seen=recent_track.last_seen,
                        status=recent_track.status,
                    )
                else:
                    return LocationResponse(
                        person_id=person.id,
                        person_name=person.name,
                        current_zone=current_zone,
                        zone_is_estimated=zone_is_estimated,
                        zone_updated_at=zone_updated_at,
                        status="never_seen",
                    )

    @router.get("/persons/{person_id}/timeline", response_model=TimelineResponse)
    async def get_person_timeline(
        person_id: int,
        since_hours: int = Query(24, description="Hours to look back", ge=1, le=720),
    ) -> TimelineResponse:
        """Get movement timeline for a person as a list of zone "stays".

        Built from ``person_zone_change`` events (not raw track sightings):
        consecutive same-zone observations are collapsed into a single stay, and
        sub-threshold zone flickers (A -> b -> A) are merged into the surrounding
        zone, so the result is a room-level movement overview rather than a flood
        of per-track segments.
        """
        person = repository.get_person(person_id)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")

        since = datetime.utcnow() - timedelta(hours=since_hours)

        events = repository.get_events(
            person_id=person_id,
            event_type="person_zone_change",
            since=since,
            limit=2000,
        )
        # get_events returns newest-first; we need chronological order to build stays
        events.sort(key=lambda e: e.timestamp)

        # Phase 1 — collapse consecutive same-zone events into raw stays
        raw: list[StayResponse] = []
        for e in events:
            ex = e.extra_data or {}
            zone = ex.get("new_zone")
            est = bool(ex.get("is_estimated", False))
            if raw and raw[-1].zone == zone:
                s = raw[-1]
                if s.snapshot_event_id is None and e.snapshot_path and not est:
                    s.snapshot_event_id = e.id
                if not est:
                    s.is_estimated = False  # a confirmed observation upgrades the stay
            else:
                raw.append(StayResponse(
                    zone=zone,
                    camera_id=e.camera_id,
                    entered_at=e.timestamp,
                    is_estimated=est,
                    snapshot_event_id=(e.id if e.snapshot_path and not est else None),
                ))

        # exited_at = next stay's entered_at; last stay stays open (current)
        for i in range(len(raw) - 1):
            raw[i].exited_at = raw[i + 1].entered_at

        # Phase 2 — merge sub-threshold flickers flanked by the same zone (A -> b -> A).
        # Genuine room-to-room transitions (A -> B) are preserved.
        if timeline_min_stay_sec > 0:
            stays: list[StayResponse] = []
            i = 0
            while i < len(raw):
                s = raw[i]
                dur = (
                    (s.exited_at - s.entered_at).total_seconds()
                    if s.exited_at else None
                )
                is_flicker = (
                    dur is not None
                    and dur < timeline_min_stay_sec
                    and stays
                    and i + 1 < len(raw)
                    and stays[-1].zone == raw[i + 1].zone
                )
                if is_flicker:
                    # Fuse previous stay, this flicker, and the next same-zone stay.
                    prev = stays[-1]
                    nxt = raw[i + 1]
                    prev.exited_at = nxt.exited_at
                    if prev.snapshot_event_id is None and nxt.snapshot_event_id is not None:
                        prev.snapshot_event_id = nxt.snapshot_event_id
                    if not nxt.is_estimated:
                        prev.is_estimated = False
                    i += 2  # consumed the flicker and the following stay
                else:
                    stays.append(s)
                    i += 1
        else:
            stays = raw

        # Phase 3 — finalize durations and current flag
        for s in stays:
            s.duration_sec = (
                (s.exited_at - s.entered_at).total_seconds()
                if s.exited_at else None
            )
            s.is_current = s.exited_at is None

        stays.sort(key=lambda s: s.entered_at, reverse=True)  # newest first

        return TimelineResponse(
            person_id=person.id,
            person_name=person.name,
            segments=stays,
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
                zone=track.extra_data.get("zone") if track.extra_data else None,
                first_seen=track.first_seen,
                last_seen=track.last_seen,
                duration_sec=duration_sec,
            ))

        return result

    return router
