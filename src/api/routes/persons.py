"""API routes for person management including image upload."""

import base64
import logging
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.config import FaceRecognitionConfig
from src.database.repository import Repository
from src.utils.image_utils import crop_with_margin

logger = logging.getLogger(__name__)


# Response models
class FaceImageResponse(BaseModel):
    id: int
    source_image: Optional[str] = None
    has_image: bool
    created_at: datetime


class PersonDetailResponse(BaseModel):
    id: int
    name: str
    is_active: bool
    created_at: datetime
    face_count: int
    images: List[FaceImageResponse]


class PersonCreateRequest(BaseModel):
    name: str


class PersonCreateResponse(BaseModel):
    id: int
    name: str
    message: str


class ImageUploadResponse(BaseModel):
    success: bool
    faces_added: int
    message: str


def create_persons_router(
    repository: Repository,
    face_config: FaceRecognitionConfig,
    faces_dir: str = "data/faces"
) -> APIRouter:
    """Create persons router with repository dependency.

    Args:
        repository: Database repository instance
        face_config: Face recognition configuration
        faces_dir: Directory to store uploaded face images

    Returns:
        Configured APIRouter
    """
    router = APIRouter(prefix="/api/v1", tags=["persons"])

    # Ensure faces directory exists
    faces_path = Path(faces_dir)
    faces_path.mkdir(parents=True, exist_ok=True)

    # Lazy load face recognizer
    _face_recognizer = None

    def get_face_recognizer():
        nonlocal _face_recognizer
        if _face_recognizer is None:
            from src.recognition.face_recognizer import FaceRecognizer
            _face_recognizer = FaceRecognizer(face_config)
        return _face_recognizer

    @router.post("/persons", response_model=PersonCreateResponse)
    async def create_person(request: PersonCreateRequest) -> PersonCreateResponse:
        """Create a new person."""
        if not request.name.strip():
            raise HTTPException(status_code=400, detail="Name cannot be empty")

        # Check if person already exists
        existing = repository.get_person_by_name(request.name.strip())
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"Person with name '{request.name}' already exists"
            )

        person = repository.create_person(request.name.strip())
        logger.info(f"Created person via API: {person.id} - {person.name}")

        return PersonCreateResponse(
            id=person.id,
            name=person.name,
            message=f"Person '{person.name}' created successfully"
        )

    @router.get("/persons/{person_id}/detail", response_model=PersonDetailResponse)
    async def get_person_detail(person_id: int) -> PersonDetailResponse:
        """Get person details including all face images."""
        person = repository.get_person(person_id)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")

        # Get face embeddings with source images
        from src.database.models import FaceEmbedding
        with repository.get_session() as session:
            embeddings = (
                session.query(FaceEmbedding)
                .filter(FaceEmbedding.person_id == person_id)
                .order_by(FaceEmbedding.created_at.desc())
                .all()
            )

            images = []
            for emb in embeddings:
                has_image = False
                if emb.source_image:
                    # Check if file exists
                    has_image = os.path.exists(emb.source_image)

                images.append(FaceImageResponse(
                    id=emb.id,
                    source_image=emb.source_image,
                    has_image=has_image,
                    created_at=emb.created_at,
                ))

            return PersonDetailResponse(
                id=person.id,
                name=person.name,
                is_active=person.is_active,
                created_at=person.created_at,
                face_count=len(embeddings),
                images=images,
            )

    @router.get("/faces/image/{image_id}")
    async def get_face_image(image_id: int):
        """Get a specific face image by embedding ID."""
        from src.database.models import FaceEmbedding
        with repository.get_session() as session:
            embedding = session.query(FaceEmbedding).filter(
                FaceEmbedding.id == image_id
            ).first()

            if not embedding:
                raise HTTPException(status_code=404, detail="Face image not found")

            if not embedding.source_image:
                raise HTTPException(status_code=404, detail="No source image stored")

            if not os.path.exists(embedding.source_image):
                raise HTTPException(status_code=404, detail="Image file not found")

            return FileResponse(
                embedding.source_image,
                media_type="image/jpeg",
                filename=os.path.basename(embedding.source_image)
            )

    @router.post("/persons/{person_id}/images", response_model=ImageUploadResponse)
    async def upload_person_images(
        person_id: int,
        files: List[UploadFile] = File(...)
    ) -> ImageUploadResponse:
        """Upload face images for a person."""
        person = repository.get_person(person_id)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")

        recognizer = get_face_recognizer()
        faces_added = 0
        errors = []

        # Create person-specific directory
        person_dir = faces_path / f"person_{person_id}"
        person_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            try:
                # Read image data
                contents = await file.read()
                nparr = np.frombuffer(contents, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if image is None:
                    errors.append(f"{file.filename}: Could not decode image")
                    continue

                # Detect faces
                result = recognizer.detect_faces(image)
                if not result.faces:
                    errors.append(f"{file.filename}: No face detected")
                    continue

                # Process each detected face
                for i, face in enumerate(result.faces):
                    embedding = recognizer.extract_embedding(image, face)
                    if embedding is None:
                        continue

                    # Save the image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_id = uuid.uuid4().hex[:8]
                    filename = f"{timestamp}_{unique_id}.jpg"
                    image_path = person_dir / filename

                    # Crop face region with margin
                    face_crop, _ = crop_with_margin(image, face.bbox, margin_ratio=0.3)
                    cv2.imwrite(str(image_path), face_crop)

                    # Store embedding with image path
                    repository.add_face_embedding(
                        person_id,
                        embedding,
                        str(image_path)
                    )
                    faces_added += 1

            except Exception as e:
                errors.append(f"{file.filename}: {str(e)}")
                logger.error(f"Error processing uploaded image: {e}")

        if faces_added == 0 and errors:
            raise HTTPException(
                status_code=400,
                detail=f"No faces added. Errors: {'; '.join(errors)}"
            )

        message = f"Added {faces_added} face(s)"
        if errors:
            message += f". Errors: {'; '.join(errors)}"

        return ImageUploadResponse(
            success=faces_added > 0,
            faces_added=faces_added,
            message=message
        )

    @router.post("/persons/{person_id}/images/base64", response_model=ImageUploadResponse)
    async def upload_person_image_base64(
        person_id: int,
        image_data: str = Form(...),
        filename: str = Form(default="uploaded.jpg")
    ) -> ImageUploadResponse:
        """Upload a face image as base64 encoded data."""
        person = repository.get_person(person_id)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")

        recognizer = get_face_recognizer()

        try:
            # Decode base64 image
            # Handle data URL format (e.g., "data:image/jpeg;base64,...")
            if "," in image_data:
                image_data = image_data.split(",", 1)[1]

            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise HTTPException(status_code=400, detail="Could not decode image")

            # Detect faces
            result = recognizer.detect_faces(image)
            if not result.faces:
                raise HTTPException(status_code=400, detail="No face detected in image")

            # Create person-specific directory
            person_dir = faces_path / f"person_{person_id}"
            person_dir.mkdir(parents=True, exist_ok=True)

            faces_added = 0
            for i, face in enumerate(result.faces):
                embedding = recognizer.extract_embedding(image, face)
                if embedding is None:
                    continue

                # Save the image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = uuid.uuid4().hex[:8]
                save_filename = f"{timestamp}_{unique_id}.jpg"
                image_path = person_dir / save_filename

                # Crop face region with margin
                face_crop, _ = crop_with_margin(image, face.bbox, margin_ratio=0.3)
                cv2.imwrite(str(image_path), face_crop)

                # Store embedding with image path
                repository.add_face_embedding(
                    person_id,
                    embedding,
                    str(image_path)
                )
                faces_added += 1

            return ImageUploadResponse(
                success=faces_added > 0,
                faces_added=faces_added,
                message=f"Added {faces_added} face(s)"
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing base64 image: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @router.delete("/persons/{person_id}/images/{image_id}")
    async def delete_face_image(person_id: int, image_id: int):
        """Delete a specific face image."""
        person = repository.get_person(person_id)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")

        from src.database.models import FaceEmbedding
        with repository.get_session() as session:
            embedding = session.query(FaceEmbedding).filter(
                FaceEmbedding.id == image_id,
                FaceEmbedding.person_id == person_id
            ).first()

            if not embedding:
                raise HTTPException(status_code=404, detail="Face image not found")

            # Delete the file if it exists
            if embedding.source_image and os.path.exists(embedding.source_image):
                try:
                    os.remove(embedding.source_image)
                except Exception as e:
                    logger.warning(f"Could not delete image file: {e}")

            # Delete from database
            session.delete(embedding)
            session.commit()

            return {"success": True, "message": "Face image deleted"}

    @router.delete("/persons/{person_id}")
    async def delete_person(person_id: int):
        """Delete (deactivate) a person."""
        person = repository.get_person(person_id)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")

        success = repository.delete_person(person_id)
        if success:
            return {"success": True, "message": f"Person '{person.name}' deleted"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete person")

    return router
