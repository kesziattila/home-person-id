"""API routes for unidentified faces management."""

import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.database.repository import Repository
from src.config import FaceRecognitionConfig, UnidentifiedFacesConfig

logger = logging.getLogger(__name__)


# Response models
class MatchScore(BaseModel):
    person_id: int
    person_name: str
    score: float


class UnidentifiedFaceResponse(BaseModel):
    id: int
    camera_id: str
    track_id: Optional[str] = None
    best_match_person_id: Optional[int] = None
    best_match_person_name: Optional[str] = None
    best_match_score: Optional[float] = None
    quality_score: float
    blur_score: Optional[float] = None
    face_size: Optional[int] = None
    frontalness: Optional[float] = None
    det_confidence: Optional[float] = None
    created_at: datetime
    reviewed: bool
    dismissed: bool
    # Set when a better crop of the same track replaced this suggestion
    superseded_by_id: Optional[int] = None
    # Good enough to enroll ("Learning"): quality >= learning_min_quality
    learnable: bool = False
    # The two closest persons in the CURRENT face gallery (live scoring rule,
    # identification_manager.score_against_gallery), best first. Enrolment value:
    # clearly the right person but not too close to them (new view), far from the
    # runner-up (a close family member makes the image risky)
    match_scores: List[MatchScore] = []
    # Suggestions of other tracks the current gallery still fails on, at least
    # enrol_coverage_similarity similar to this one
    coverage: int = 0
    # Enrolment value: "recommended" (helps where recognition now fails), "risky" (a second
    # person is close), "review" (neither) — enrolment_label()
    enrol_label: str = "review"
    # Why: close_runner_up | low_quality | low_coverage | no_gallery | covers_failures
    enrol_reason: str = ""


class UnidentifiedFacesSummaryResponse(BaseModel):
    total: int
    by_camera: dict


class AssignRequest(BaseModel):
    person_id: int


class AssignResponse(BaseModel):
    success: bool
    message: str
    embedding_id: Optional[int] = None


def enrolment_label(
    scores: List[MatchScore], coverage: int, quality: float, config: UnidentifiedFacesConfig,
) -> tuple[str, str]:
    """How worth enrolling a suggestion is (assuming it goes to its closest person): (label, reason).

    risky: the runner-up person is within enrol_min_margin — the face may pull their faces;
    recommended: clearly one person, it resembles >= enrol_min_coverage other tracks'
    still-unrecognised suggestions (a situation where recognition keeps failing) AND it is a
    good picture (quality >= enrol_min_quality);
    review: otherwise — little expected gain or a poor picture (2026-10-19, docs/TUNING_LOG.md).
    """
    if not scores:
        return "review", "no_gallery"
    if len(scores) >= 2 and scores[0].score - scores[1].score < config.enrol_min_margin:
        return "risky", "close_runner_up"
    if coverage < config.enrol_min_coverage:
        return "review", "low_coverage"
    if quality < config.enrol_min_quality:
        return "review", "low_quality"
    return "recommended", "covers_failures"


def create_unidentified_faces_router(
    repository: Repository,
    face_config: Optional[FaceRecognitionConfig] = None,
    faces_dir: str = "data/faces",
    config: Optional[UnidentifiedFacesConfig] = None,
) -> APIRouter:
    """Create unidentified faces router with repository dependency.

    Args:
        repository: Database repository instance
        face_config: Face recognition configuration (optional)
        faces_dir: Directory to store registered face images
        config: Unidentified faces settings (learning_min_quality, enrol_* thresholds)

    Returns:
        Configured APIRouter
    """
    router = APIRouter(prefix="/api/v1", tags=["unidentified-faces"])
    faces_path = Path(faces_dir)
    config = config or UnidentifiedFacesConfig()

    def learnable(face) -> bool:
        # Rows scored by the hand-made formula (before 2026-10-18, they carry
        # blur_score) are on another scale; they expire with retention_days
        return face.blur_score is None and face.quality_score >= config.learning_min_quality

    gallery_cache = {"entries": [], "names": {}, "loaded_at": 0.0}

    def face_gallery():
        """Face gallery entries and person names, reloaded at most every 60 s."""
        from src.recognition.identification_manager import FaceGalleryEntry
        if time.time() - gallery_cache["loaded_at"] > 60.0:
            names = {p.id: p.name for p in repository.get_all_persons(active_only=True)}
            gallery_cache["entries"] = [
                FaceGalleryEntry(person_id=pid, person_name=names.get(pid, str(pid)), embedding=emb, embedding_id=eid)
                for pid, eid, emb in repository.get_all_face_embeddings()
            ]
            gallery_cache["names"] = names
            gallery_cache["loaded_at"] = time.time()
        return gallery_cache["entries"], gallery_cache["names"]

    pool_cache = {"ids": np.zeros(0, int), "tracks": np.zeros(0, object), "emb": np.zeros((0, 512)), "loaded_at": 0.0}

    def unit(blob: bytes) -> np.ndarray:
        e = np.frombuffer(blob, dtype=np.float32)
        return e / (np.linalg.norm(e) or 1.0)

    face_threshold = face_config.similarity_threshold if face_config else FaceRecognitionConfig().similarity_threshold
    face_min_margin = face_config.min_margin if face_config else FaceRecognitionConfig().min_margin

    def still_failing(face) -> bool:
        """The current gallery still does not accept this suggestion (live match rule). Older
        suggestions the gallery has since learned are no failure any more: on 2026-10-19 8 of
        the 11 faces #1403 'covered' were already accepted, which inflated its +N."""
        from src.recognition.identification_manager import score_against_gallery
        entries, _ = face_gallery()
        return not score_against_gallery(unit(face.embedding), entries).accepts(face_threshold, face_min_margin)

    def suggestion_pool():
        """Suggestions the gallery still fails on (not dismissed, quality-model rows), INCLUDING
        archived crops (superseded by a better crop of the same track): more tracks / days to
        measure a candidate against (2026-10-19: 130 -> 380 rows). Reloaded at most every 60 s."""
        if time.time() - pool_cache["loaded_at"] > 60.0:
            rows = [f for f in repository.get_unidentified_faces(dismissed=False, archived=None, limit=100000)
                    if f.embedding and f.blur_score is None and still_failing(f)]
            pool_cache["ids"] = np.array([f.id for f in rows], dtype=int)
            pool_cache["tracks"] = np.array([f.track_id for f in rows], dtype=object)
            pool_cache["emb"] = np.stack([unit(f.embedding) for f in rows]) if rows else np.zeros((0, 512))
            pool_cache["loaded_at"] = time.time()
        return pool_cache["ids"], pool_cache["tracks"], pool_cache["emb"]

    def invalidate():
        gallery_cache["loaded_at"] = pool_cache["loaded_at"] = 0.0

    def coverage(face) -> int:
        """OTHER tracks with a still-unrecognised suggestion at least enrol_coverage_similarity similar
        to this face. Counted per track: archived crops give one track several rows."""
        if not face.embedding:
            return 0
        ids, tracks, emb = suggestion_pool()
        if not len(ids):
            return 0
        sims = emb @ unit(face.embedding)
        other = (ids != face.id) & ((tracks != face.track_id) if face.track_id else True)
        hit = (sims >= config.enrol_coverage_similarity) & other
        return len({t if t else f"#{i}" for t, i in zip(tracks[hit], ids[hit])})

    def match_scores(face) -> List[MatchScore]:
        if not face.embedding:
            return []
        from src.recognition.identification_manager import score_against_gallery
        entries, names = face_gallery()
        scores = score_against_gallery(unit(face.embedding), entries)
        best = sorted(scores.per_person.items(), key=lambda kv: -kv[1])[:2]
        return [MatchScore(person_id=pid, person_name=names.get(pid, str(pid)), score=round(score, 3))
                for pid, score in best]

    def to_response(face) -> UnidentifiedFaceResponse:
        scores = match_scores(face)
        cov = coverage(face)
        label, reason = enrolment_label(scores, cov, face.quality_score, config)
        best_match_name = None
        if face.best_match_person_id:
            person = repository.get_person(face.best_match_person_id)
            if person:
                best_match_name = person.name
        return UnidentifiedFaceResponse(
            id=face.id,
            camera_id=face.camera_id,
            track_id=face.track_id,
            best_match_person_id=face.best_match_person_id,
            best_match_person_name=best_match_name,
            best_match_score=face.best_match_score,
            quality_score=face.quality_score,
            blur_score=face.blur_score,
            face_size=face.face_size,
            frontalness=face.frontalness,
            det_confidence=face.det_confidence,
            created_at=face.created_at,
            reviewed=face.reviewed,
            dismissed=face.dismissed,
            superseded_by_id=face.superseded_by_id,
            learnable=learnable(face),
            match_scores=scores,
            coverage=cov,
            enrol_label=label,
            enrol_reason=reason,
        )

    @router.get("/unidentified-faces", response_model=List[UnidentifiedFaceResponse])
    async def list_unidentified_faces(
        camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
        reviewed: Optional[bool] = Query(None, description="Filter by reviewed status"),
        dismissed: Optional[bool] = Query(False, description="Filter by dismissed status"),
        limit: int = Query(50, description="Maximum faces to return", ge=1, le=200),
        offset: int = Query(0, description="Number of faces to skip", ge=0),
        archived: Optional[bool] = Query(
            False, description="Filter by archived status (replaced by a better crop of the same track)"),
    ) -> List[UnidentifiedFaceResponse]:
        """List unidentified faces with optional filters."""
        faces = repository.get_unidentified_faces(
            camera_id=camera_id,
            reviewed=reviewed,
            dismissed=dismissed,
            limit=limit,
            offset=offset,
            archived=archived,
        )

        return [to_response(face) for face in faces]

    @router.get("/unidentified-faces/summary", response_model=UnidentifiedFacesSummaryResponse)
    async def get_unidentified_faces_summary() -> UnidentifiedFacesSummaryResponse:
        """Get summary count of unidentified faces by camera."""
        by_camera = repository.get_unidentified_faces_summary()
        total = sum(by_camera.values())

        return UnidentifiedFacesSummaryResponse(
            total=total,
            by_camera=by_camera,
        )

    @router.get("/unidentified-faces/{face_id}")
    async def get_unidentified_face(face_id: int) -> UnidentifiedFaceResponse:
        """Get a specific unidentified face by ID."""
        face = repository.get_unidentified_face(face_id)
        if not face:
            raise HTTPException(status_code=404, detail="Unidentified face not found")

        return to_response(face)

    @router.get("/unidentified-faces/{face_id}/image")
    async def get_unidentified_face_image(face_id: int):
        """Get the image for an unidentified face."""
        face = repository.get_unidentified_face(face_id)
        if not face:
            raise HTTPException(status_code=404, detail="Unidentified face not found")

        if not face.image_path:
            raise HTTPException(status_code=404, detail="No image path stored")

        if not os.path.exists(face.image_path):
            raise HTTPException(status_code=404, detail="Image file not found")

        return FileResponse(
            face.image_path,
            media_type="image/jpeg",
            filename=os.path.basename(face.image_path)
        )

    @router.post("/unidentified-faces/{face_id}/assign", response_model=AssignResponse)
    async def assign_unidentified_face(
        face_id: int,
        request: AssignRequest,
    ) -> AssignResponse:
        """Assign an unidentified face to a person.

        This moves the face embedding to the person's gallery.
        """
        face = repository.get_unidentified_face(face_id)
        if not face:
            raise HTTPException(status_code=404, detail="Unidentified face not found")

        person = repository.get_person(request.person_id)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")

        # Enrolment value at the time of the decision, for calibrating enrol_* on live data
        value = to_response(face)

        # Prepare new image path in person's gallery
        new_image_path = None
        if face.image_path and os.path.exists(face.image_path):
            person_dir = faces_path / f"person_{person.id}"
            person_dir.mkdir(parents=True, exist_ok=True)
            
            filename = os.path.basename(face.image_path)
            # Add prefix to distinguish from directly uploaded images if needed, 
            # but using original filename is fine
            dest_path = person_dir / filename
            
            try:
                shutil.move(face.image_path, str(dest_path))
                new_image_path = str(dest_path)
            except Exception as e:
                logger.error(f"Failed to move unidentified face image: {e}")
                # Continue with original path if move fails, or fail? 
                # Let's try to keep it consistent.
                new_image_path = face.image_path

        # Assign to person
        embedding = repository.assign_unidentified_face_to_person(
            face_id, request.person_id, new_image_path=new_image_path
        )
        if not embedding:
            raise HTTPException(status_code=500, detail="Failed to assign face to person")

        logger.info(f"Assigned unidentified face {face_id} to person {person.name} (ID: {person.id}): "
                    f"{value.enrol_label}, coverage {value.coverage}, "
                    f"{[(s.person_name, s.score) for s in value.match_scores]}, quality {face.quality_score:.2f}")
        repository.create_event(
            camera_id=face.camera_id,
            event_type="face_enrolled",
            track_id=face.track_id,
            person_id=person.id,
            extra_data={
                "suggestion_id": face_id, "embedding_id": embedding.id, "enrol_label": value.enrol_label,
                "enrol_reason": value.enrol_reason,
                "coverage": value.coverage, "quality": round(face.quality_score, 3),
                "match_scores": [s.model_dump() for s in value.match_scores],
            },
        )
        invalidate()  # the next match scores / coverage see the new gallery and pool

        return AssignResponse(
            success=True,
            message=f"Face assigned to {person.name}",
            embedding_id=embedding.id,
        )

    @router.post("/unidentified-faces/{face_id}/dismiss")
    async def dismiss_unidentified_face(face_id: int):
        """Mark an unidentified face as dismissed.

        Dismissed faces are hidden from the default list but not deleted.
        """
        face = repository.get_unidentified_face(face_id)
        if not face:
            raise HTTPException(status_code=404, detail="Unidentified face not found")

        updated = repository.update_unidentified_face(face_id, dismissed=True, reviewed=True)
        if not updated:
            raise HTTPException(status_code=500, detail="Failed to dismiss face")

        invalidate()
        return {"success": True, "message": "Face dismissed"}

    @router.post("/unidentified-faces/{face_id}/review")
    async def mark_face_reviewed(face_id: int):
        """Mark an unidentified face as reviewed (but not dismissed)."""
        face = repository.get_unidentified_face(face_id)
        if not face:
            raise HTTPException(status_code=404, detail="Unidentified face not found")

        updated = repository.update_unidentified_face(face_id, reviewed=True)
        if not updated:
            raise HTTPException(status_code=500, detail="Failed to update face")

        return {"success": True, "message": "Face marked as reviewed"}

    @router.delete("/unidentified-faces/{face_id}")
    async def delete_unidentified_face(face_id: int):
        """Delete an unidentified face permanently."""
        face = repository.get_unidentified_face(face_id)
        if not face:
            raise HTTPException(status_code=404, detail="Unidentified face not found")

        # Delete image file if it exists
        if face.image_path and os.path.exists(face.image_path):
            try:
                os.remove(face.image_path)
            except Exception as e:
                logger.warning(f"Could not delete image file: {e}")

        # Delete from database
        deleted = repository.delete_unidentified_face(face_id)
        if not deleted:
            raise HTTPException(status_code=500, detail="Failed to delete face")

        invalidate()
        return {"success": True, "message": "Face deleted"}

    return router
