"""Unit tests for person management API routes."""

import io
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.config import FaceRecognitionConfig
from src.database.repository import Repository
from src.api.routes.persons import create_persons_router


@pytest.fixture(scope="function")
def repository():
    """Create temporary file-based database repository for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    repo = Repository(db_path)
    yield repo

    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def face_config():
    """Create face recognition config for testing."""
    return FaceRecognitionConfig(
        enabled=True,
        model="buffalo_l",
        similarity_threshold=0.6,
    )


@pytest.fixture
def temp_faces_dir():
    """Create temporary faces directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def client(repository, face_config, temp_faces_dir):
    """Create test client with persons router."""
    app = FastAPI()
    persons_router = create_persons_router(
        repository,
        face_config,
        faces_dir=temp_faces_dir
    )
    app.include_router(persons_router)
    return TestClient(app)


@pytest.fixture
def sample_person(repository):
    """Create a sample person."""
    return repository.create_person("Test Person")


class TestCreatePersonEndpoint:
    """Tests for POST /api/v1/persons endpoint."""

    def test_create_person(self, client):
        """Test creating a new person."""
        response = client.post(
            "/api/v1/persons",
            json={"name": "John Doe"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "John Doe"
        assert "id" in data
        assert data["message"] == "Person 'John Doe' created successfully"

    def test_create_person_empty_name(self, client):
        """Test creating person with empty name."""
        response = client.post(
            "/api/v1/persons",
            json={"name": "   "}
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_create_person_duplicate_name(self, client, sample_person):
        """Test creating person with duplicate name."""
        response = client.post(
            "/api/v1/persons",
            json={"name": "Test Person"}
        )
        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]


class TestPersonDetailEndpoint:
    """Tests for GET /api/v1/persons/{person_id}/detail endpoint."""

    def test_get_person_detail(self, client, sample_person):
        """Test getting person details."""
        response = client.get(f"/api/v1/persons/{sample_person.id}/detail")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == sample_person.id
        assert data["name"] == "Test Person"
        assert data["is_active"] == True
        assert "images" in data
        assert "face_count" in data

    def test_get_person_detail_not_found(self, client):
        """Test getting non-existent person details."""
        response = client.get("/api/v1/persons/999/detail")
        assert response.status_code == 404

    def test_get_person_detail_with_images(self, client, repository, sample_person):
        """Test getting person details with face images."""
        # Add some embeddings
        embedding = np.random.rand(512).astype(np.float32)
        repository.add_face_embedding(sample_person.id, embedding, "/path/to/image1.jpg")
        repository.add_face_embedding(sample_person.id, embedding, None)  # No source

        response = client.get(f"/api/v1/persons/{sample_person.id}/detail")
        assert response.status_code == 200
        data = response.json()
        assert data["face_count"] == 2
        assert len(data["images"]) == 2


class TestDeleteImageEndpoint:
    """Tests for DELETE /api/v1/persons/{person_id}/images/{image_id} endpoint."""

    def test_delete_image(self, client, repository, sample_person):
        """Test deleting a face image."""
        # Add an embedding first
        embedding = np.random.rand(512).astype(np.float32)
        repository.add_face_embedding(sample_person.id, embedding, "/fake/path.jpg")

        # Get the image ID
        detail_resp = client.get(f"/api/v1/persons/{sample_person.id}/detail")
        images = detail_resp.json()["images"]
        assert len(images) > 0
        image_id = images[0]["id"]

        # Delete it
        response = client.delete(f"/api/v1/persons/{sample_person.id}/images/{image_id}")
        assert response.status_code == 200
        assert response.json()["success"] == True

        # Verify it's deleted
        detail_resp = client.get(f"/api/v1/persons/{sample_person.id}/detail")
        assert len(detail_resp.json()["images"]) == 0

    def test_delete_image_not_found(self, client, sample_person):
        """Test deleting non-existent image."""
        response = client.delete(f"/api/v1/persons/{sample_person.id}/images/999")
        assert response.status_code == 404

    def test_delete_image_wrong_person(self, client, repository, sample_person):
        """Test deleting image belonging to different person."""
        # Create another person
        other_person = repository.create_person("Other Person")

        # Add embedding to original person
        embedding = np.random.rand(512).astype(np.float32)
        repository.add_face_embedding(sample_person.id, embedding, "/fake/path.jpg")

        # Get the image ID
        detail_resp = client.get(f"/api/v1/persons/{sample_person.id}/detail")
        image_id = detail_resp.json()["images"][0]["id"]

        # Try to delete using wrong person ID
        response = client.delete(f"/api/v1/persons/{other_person.id}/images/{image_id}")
        assert response.status_code == 404


class TestDeletePersonEndpoint:
    """Tests for DELETE /api/v1/persons/{person_id} endpoint."""

    def test_delete_person(self, client, sample_person, repository):
        """Test deleting a person."""
        response = client.delete(f"/api/v1/persons/{sample_person.id}")
        assert response.status_code == 200
        assert response.json()["success"] == True

        # Verify person is deactivated
        person = repository.get_person(sample_person.id)
        assert person.is_active == False

    def test_delete_person_not_found(self, client):
        """Test deleting non-existent person."""
        response = client.delete("/api/v1/persons/999")
        assert response.status_code == 404


class TestGetFaceImageEndpoint:
    """Tests for GET /api/v1/faces/image/{image_id} endpoint."""

    def test_get_face_image_not_found(self, client):
        """Test getting non-existent face image."""
        response = client.get("/api/v1/faces/image/999")
        assert response.status_code == 404

    def test_get_face_image_no_source(self, client, repository, sample_person):
        """Test getting face image with no source file."""
        # Add embedding without source image
        embedding = np.random.rand(512).astype(np.float32)
        repository.add_face_embedding(sample_person.id, embedding, None)

        # Get image ID
        detail_resp = client.get(f"/api/v1/persons/{sample_person.id}/detail")
        images = detail_resp.json()["images"]
        image_id = images[0]["id"]

        # Try to get image
        response = client.get(f"/api/v1/faces/image/{image_id}")
        assert response.status_code == 404
        assert "No source image" in response.json()["detail"]

    def test_get_face_image_file_missing(self, client, repository, sample_person):
        """Test getting face image when file doesn't exist."""
        # Add embedding with non-existent file
        embedding = np.random.rand(512).astype(np.float32)
        repository.add_face_embedding(sample_person.id, embedding, "/nonexistent/path.jpg")

        # Get image ID
        detail_resp = client.get(f"/api/v1/persons/{sample_person.id}/detail")
        images = detail_resp.json()["images"]
        image_id = images[0]["id"]

        # Try to get image
        response = client.get(f"/api/v1/faces/image/{image_id}")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_face_image_success(self, client, repository, sample_person, temp_faces_dir):
        """Test successfully getting a face image."""
        import cv2

        # Create actual image file
        image_path = os.path.join(temp_faces_dir, "test_face.jpg")
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:] = (255, 200, 150)
        cv2.imwrite(image_path, img)

        # Add embedding with real file
        embedding = np.random.rand(512).astype(np.float32)
        repository.add_face_embedding(sample_person.id, embedding, image_path)

        # Get image ID
        detail_resp = client.get(f"/api/v1/persons/{sample_person.id}/detail")
        images = detail_resp.json()["images"]
        image_id = images[0]["id"]

        # Get image
        response = client.get(f"/api/v1/faces/image/{image_id}")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"


class TestImageUploadEndpoint:
    """Tests for POST /api/v1/persons/{person_id}/images endpoint."""

    def test_upload_webp_image(self, client, repository, sample_person, temp_faces_dir):
        """Test uploading a WebP image."""
        from PIL import Image

        # Create a WebP image in memory
        img = Image.new('RGB', (200, 200), color=(255, 200, 150))
        webp_buffer = io.BytesIO()
        img.save(webp_buffer, format='WEBP')
        webp_buffer.seek(0)

        # Mock the face recognizer to avoid loading actual models
        with patch('src.recognition.face_recognizer.FaceRecognizer') as MockRecognizer:
            mock_recognizer = MagicMock()
            MockRecognizer.return_value = mock_recognizer

            # Create mock face detection result
            mock_face = MagicMock()
            mock_face.bbox = [50, 50, 150, 150]
            mock_result = MagicMock()
            mock_result.faces = [mock_face]
            mock_recognizer.detect_faces.return_value = mock_result
            mock_recognizer.extract_embedding.return_value = np.random.rand(512).astype(np.float32)

            response = client.post(
                f"/api/v1/persons/{sample_person.id}/images",
                files={"files": ("test.webp", webp_buffer, "image/webp")}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert data["faces_added"] == 1

    def test_upload_png_image(self, client, repository, sample_person, temp_faces_dir):
        """Test uploading a PNG image."""
        from PIL import Image

        # Create a PNG image in memory
        img = Image.new('RGB', (200, 200), color=(100, 150, 200))
        png_buffer = io.BytesIO()
        img.save(png_buffer, format='PNG')
        png_buffer.seek(0)

        with patch('src.recognition.face_recognizer.FaceRecognizer') as MockRecognizer:
            mock_recognizer = MagicMock()
            MockRecognizer.return_value = mock_recognizer

            mock_face = MagicMock()
            mock_face.bbox = [50, 50, 150, 150]
            mock_result = MagicMock()
            mock_result.faces = [mock_face]
            mock_recognizer.detect_faces.return_value = mock_result
            mock_recognizer.extract_embedding.return_value = np.random.rand(512).astype(np.float32)

            response = client.post(
                f"/api/v1/persons/{sample_person.id}/images",
                files={"files": ("test.png", png_buffer, "image/png")}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert data["faces_added"] == 1

    def test_upload_jpeg_image(self, client, repository, sample_person, temp_faces_dir):
        """Test uploading a JPEG image."""
        from PIL import Image

        # Create a JPEG image in memory
        img = Image.new('RGB', (200, 200), color=(200, 100, 50))
        jpeg_buffer = io.BytesIO()
        img.save(jpeg_buffer, format='JPEG')
        jpeg_buffer.seek(0)

        with patch('src.recognition.face_recognizer.FaceRecognizer') as MockRecognizer:
            mock_recognizer = MagicMock()
            MockRecognizer.return_value = mock_recognizer

            mock_face = MagicMock()
            mock_face.bbox = [50, 50, 150, 150]
            mock_result = MagicMock()
            mock_result.faces = [mock_face]
            mock_recognizer.detect_faces.return_value = mock_result
            mock_recognizer.extract_embedding.return_value = np.random.rand(512).astype(np.float32)

            response = client.post(
                f"/api/v1/persons/{sample_person.id}/images",
                files={"files": ("test.jpg", jpeg_buffer, "image/jpeg")}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert data["faces_added"] == 1

    def test_upload_no_face_detected(self, client, repository, sample_person, temp_faces_dir):
        """Test uploading image with no face detected."""
        from PIL import Image

        img = Image.new('RGB', (200, 200), color=(0, 0, 0))
        jpeg_buffer = io.BytesIO()
        img.save(jpeg_buffer, format='JPEG')
        jpeg_buffer.seek(0)

        with patch('src.recognition.face_recognizer.FaceRecognizer') as MockRecognizer:
            mock_recognizer = MagicMock()
            MockRecognizer.return_value = mock_recognizer

            # Return no faces
            mock_result = MagicMock()
            mock_result.faces = []
            mock_recognizer.detect_faces.return_value = mock_result

            response = client.post(
                f"/api/v1/persons/{sample_person.id}/images",
                files={"files": ("test.jpg", jpeg_buffer, "image/jpeg")}
            )

            assert response.status_code == 400
            assert "No face detected" in response.json()["detail"]
