"""API routes for unidentified faces management."""

import logging
import os
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.database.repository import Repository
from src.config import FaceRecognitionConfig

logger = logging.getLogger(__name__)


# Response models
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
    created_at: datetime
    reviewed: bool
    dismissed: bool


class UnidentifiedFacesSummaryResponse(BaseModel):
    total: int
    by_camera: dict


class AssignRequest(BaseModel):
    person_id: int


class AssignResponse(BaseModel):
    success: bool
    message: str
    embedding_id: Optional[int] = None


def create_unidentified_faces_router(
    repository: Repository,
    face_config: Optional[FaceRecognitionConfig] = None,
) -> APIRouter:
    """Create unidentified faces router with repository dependency.

    Args:
        repository: Database repository instance
        face_config: Face recognition configuration (optional)

    Returns:
        Configured APIRouter
    """
    router = APIRouter(prefix="/api/v1", tags=["unidentified-faces"])

    @router.get("/unidentified-faces", response_model=List[UnidentifiedFaceResponse])
    async def list_unidentified_faces(
        camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
        reviewed: Optional[bool] = Query(None, description="Filter by reviewed status"),
        dismissed: Optional[bool] = Query(False, description="Filter by dismissed status"),
        limit: int = Query(50, description="Maximum faces to return", ge=1, le=200),
        offset: int = Query(0, description="Number of faces to skip", ge=0),
    ) -> List[UnidentifiedFaceResponse]:
        """List unidentified faces with optional filters."""
        faces = repository.get_unidentified_faces(
            camera_id=camera_id,
            reviewed=reviewed,
            dismissed=dismissed,
            limit=limit,
            offset=offset,
        )

        result = []
        for face in faces:
            # Get best match person name if available
            best_match_name = None
            if face.best_match_person_id:
                person = repository.get_person(face.best_match_person_id)
                if person:
                    best_match_name = person.name

            result.append(UnidentifiedFaceResponse(
                id=face.id,
                camera_id=face.camera_id,
                track_id=face.track_id,
                best_match_person_id=face.best_match_person_id,
                best_match_person_name=best_match_name,
                best_match_score=face.best_match_score,
                quality_score=face.quality_score,
                blur_score=face.blur_score,
                face_size=face.face_size,
                created_at=face.created_at,
                reviewed=face.reviewed,
                dismissed=face.dismissed,
            ))

        return result

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

        # Get best match person name if available
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
            created_at=face.created_at,
            reviewed=face.reviewed,
            dismissed=face.dismissed,
        )

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

        # Assign to person
        embedding = repository.assign_unidentified_face_to_person(face_id, request.person_id)
        if not embedding:
            raise HTTPException(status_code=500, detail="Failed to assign face to person")

        logger.info(f"Assigned unidentified face {face_id} to person {person.name} (ID: {person.id})")

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

        return {"success": True, "message": "Face deleted"}

    return router
