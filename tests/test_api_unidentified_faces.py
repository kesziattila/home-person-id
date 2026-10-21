"""Unit tests for API unidentified faces routes."""

import os
import tempfile

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.unidentified_faces import create_unidentified_faces_router
from src.database.repository import Repository


@pytest.fixture(scope="function")
def repository():
    """Create temporary file-based database repository for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    repo = Repository(db_path)
    yield repo

    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def client(repository):
    """Create test client with fresh app and repository."""
    app = FastAPI()
    router = create_unidentified_faces_router(repository)
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def sample_data(repository, tmp_path):
    """Populate database with sample unidentified faces."""
    # Create persons for matching
    person1 = repository.create_person("John Doe")
    person2 = repository.create_person("Jane Smith")

    # Create fake embeddings
    embedding1 = np.random.randn(512).astype(np.float32)
    embedding2 = np.random.randn(512).astype(np.float32)
    embedding3 = np.random.randn(512).astype(np.float32)

    # Create fake image files
    img_path1 = tmp_path / "face1.jpg"
    img_path2 = tmp_path / "face2.jpg"
    img_path3 = tmp_path / "face3.jpg"

    # Write minimal JPEG data (1x1 black pixel JPEG)
    minimal_jpeg = bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
        0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
        0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
        0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
        0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
        0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
        0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
        0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
        0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
        0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
        0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
        0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
        0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72,
        0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45,
        0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
        0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
        0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3,
        0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
        0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9,
        0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
        0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4,
        0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01,
        0x00, 0x00, 0x3F, 0x00, 0xFB, 0xD5, 0xDB, 0x00, 0x31, 0xC4, 0x1F, 0xFF,
        0xD9
    ])

    img_path1.write_bytes(minimal_jpeg)
    img_path2.write_bytes(minimal_jpeg)
    img_path3.write_bytes(minimal_jpeg)

    # Create unidentified faces
    face1 = repository.add_unidentified_face(
        camera_id="camera1",
        embedding=embedding1,
        image_path=str(img_path1),
        quality_score=0.7,
        track_id="global_1",
        best_match_person_id=person1.id,
        best_match_score=0.55,
        blur_score=120.0,
        face_size=80,
    )

    face2 = repository.add_unidentified_face(
        camera_id="camera2",
        embedding=embedding2,
        image_path=str(img_path2),
        quality_score=0.5,
        track_id="global_2",
        best_match_person_id=None,
        best_match_score=None,
        blur_score=80.0,
        face_size=60,
    )

    face3 = repository.add_unidentified_face(
        camera_id="camera1",
        embedding=embedding3,
        image_path=str(img_path3),
        quality_score=0.8,
        track_id="global_3",
        best_match_person_id=person2.id,
        best_match_score=0.48,
        blur_score=150.0,
        face_size=100,
    )

    return {
        "person1": person1,
        "person2": person2,
        "face1": face1,
        "face2": face2,
        "face3": face3,
        "tmp_path": tmp_path,
    }


class TestListUnidentifiedFaces:
    """Tests for GET /api/v1/unidentified-faces endpoint."""

    def test_list_empty(self, client):
        """Test listing when no faces exist."""
        response = client.get("/api/v1/unidentified-faces")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_all(self, client, sample_data):
        """Test listing all unidentified faces."""
        response = client.get("/api/v1/unidentified-faces")
        assert response.status_code == 200
        faces = response.json()
        assert len(faces) == 3

        # Check structure
        for face in faces:
            assert "id" in face
            assert "camera_id" in face
            assert "quality_score" in face
            assert "created_at" in face

    def test_list_filter_by_camera(self, client, sample_data):
        """Test filtering by camera ID."""
        response = client.get("/api/v1/unidentified-faces?camera_id=camera1")
        assert response.status_code == 200
        faces = response.json()
        assert len(faces) == 2
        assert all(f["camera_id"] == "camera1" for f in faces)

    def test_list_filter_dismissed(self, client, sample_data, repository):
        """Test filtering by dismissed status."""
        # Dismiss one face
        repository.update_unidentified_face(sample_data["face1"].id, dismissed=True)

        # Should not include dismissed by default
        response = client.get("/api/v1/unidentified-faces?dismissed=false")
        assert response.status_code == 200
        faces = response.json()
        assert len(faces) == 2
        assert all(not f["dismissed"] for f in faces)

        # Include dismissed
        response = client.get("/api/v1/unidentified-faces?dismissed=true")
        assert response.status_code == 200
        faces = response.json()
        assert len(faces) == 1
        assert faces[0]["dismissed"] is True

    def test_list_limit_offset(self, client, sample_data):
        """Test pagination with limit and offset."""
        response = client.get("/api/v1/unidentified-faces?limit=2")
        assert response.status_code == 200
        faces = response.json()
        assert len(faces) == 2

        response = client.get("/api/v1/unidentified-faces?limit=2&offset=2")
        assert response.status_code == 200
        faces = response.json()
        assert len(faces) == 1

    def test_list_includes_best_match_name(self, client, sample_data):
        """Test that best match person name is included."""
        response = client.get("/api/v1/unidentified-faces")
        assert response.status_code == 200
        faces = response.json()

        # Find face with best match
        face_with_match = next(
            (f for f in faces if f["best_match_person_id"] is not None), None
        )
        assert face_with_match is not None
        assert face_with_match["best_match_person_name"] in ["John Doe", "Jane Smith"]


class TestGetUnidentifiedFace:
    """Tests for GET /api/v1/unidentified-faces/{id} endpoint."""

    def test_get_by_id(self, client, sample_data):
        """Test getting a specific face by ID."""
        face_id = sample_data["face1"].id
        response = client.get(f"/api/v1/unidentified-faces/{face_id}")
        assert response.status_code == 200
        face = response.json()
        assert face["id"] == face_id
        assert face["camera_id"] == "camera1"
        assert face["quality_score"] == 0.7

    def test_get_not_found(self, client):
        """Test getting non-existent face."""
        response = client.get("/api/v1/unidentified-faces/999")
        assert response.status_code == 404


class TestGetUnidentifiedFaceImage:
    """Tests for GET /api/v1/unidentified-faces/{id}/image endpoint."""

    def test_get_image(self, client, sample_data):
        """Test getting face image."""
        face_id = sample_data["face1"].id
        response = client.get(f"/api/v1/unidentified-faces/{face_id}/image")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"

    def test_get_image_not_found(self, client):
        """Test getting image for non-existent face."""
        response = client.get("/api/v1/unidentified-faces/999/image")
        assert response.status_code == 404


class TestSummaryEndpoint:
    """Tests for GET /api/v1/unidentified-faces/summary endpoint."""

    def test_summary_empty(self, client):
        """Test summary when no faces exist."""
        response = client.get("/api/v1/unidentified-faces/summary")
        assert response.status_code == 200
        summary = response.json()
        assert summary["total"] == 0
        assert summary["by_camera"] == {}

    def test_summary_with_data(self, client, sample_data):
        """Test summary with faces."""
        response = client.get("/api/v1/unidentified-faces/summary")
        assert response.status_code == 200
        summary = response.json()
        assert summary["total"] == 3
        assert "camera1" in summary["by_camera"]
        assert "camera2" in summary["by_camera"]
        assert summary["by_camera"]["camera1"] == 2
        assert summary["by_camera"]["camera2"] == 1


class TestAssignFace:
    """Tests for POST /api/v1/unidentified-faces/{id}/assign endpoint."""

    def test_assign_to_person(self, client, sample_data, repository):
        """Test assigning face to a person."""
        face_id = sample_data["face2"].id
        person_id = sample_data["person1"].id

        response = client.post(
            f"/api/v1/unidentified-faces/{face_id}/assign",
            json={"person_id": person_id}
        )
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert "embedding_id" in result

        # Face should be deleted from unidentified faces
        assert repository.get_unidentified_face(face_id) is None

        # Person should have new embedding
        embeddings = repository.get_face_embeddings(person_id)
        assert len(embeddings) >= 1

    def test_assign_not_found_face(self, client, sample_data):
        """Test assigning non-existent face."""
        response = client.post(
            "/api/v1/unidentified-faces/999/assign",
            json={"person_id": sample_data["person1"].id}
        )
        assert response.status_code == 404

    def test_assign_not_found_person(self, client, sample_data):
        """Test assigning to non-existent person."""
        response = client.post(
            f"/api/v1/unidentified-faces/{sample_data['face1'].id}/assign",
            json={"person_id": 999}
        )
        assert response.status_code == 404


class TestDismissFace:
    """Tests for POST /api/v1/unidentified-faces/{id}/dismiss endpoint."""

    def test_dismiss_face(self, client, sample_data, repository):
        """Test dismissing a face."""
        face_id = sample_data["face1"].id
        response = client.post(f"/api/v1/unidentified-faces/{face_id}/dismiss")
        assert response.status_code == 200

        # Face should be marked as dismissed
        face = repository.get_unidentified_face(face_id)
        assert face.dismissed is True
        assert face.reviewed is True

    def test_dismiss_not_found(self, client):
        """Test dismissing non-existent face."""
        response = client.post("/api/v1/unidentified-faces/999/dismiss")
        assert response.status_code == 404


class TestDeleteFace:
    """Tests for DELETE /api/v1/unidentified-faces/{id} endpoint."""

    def test_delete_face(self, client, sample_data, repository):
        """Test deleting a face."""
        face_id = sample_data["face1"].id
        image_path = sample_data["face1"].image_path

        response = client.delete(f"/api/v1/unidentified-faces/{face_id}")
        assert response.status_code == 200

        # Face should be deleted from database
        assert repository.get_unidentified_face(face_id) is None

        # Image file should be deleted
        assert not os.path.exists(image_path)

    def test_delete_not_found(self, client):
        """Test deleting non-existent face."""
        response = client.delete("/api/v1/unidentified-faces/999")
        assert response.status_code == 404


class TestMarkReviewed:
    """Tests for POST /api/v1/unidentified-faces/{id}/review endpoint."""

    def test_mark_reviewed(self, client, sample_data, repository):
        """Test marking a face as reviewed."""
        face_id = sample_data["face1"].id
        response = client.post(f"/api/v1/unidentified-faces/{face_id}/review")
        assert response.status_code == 200

        # Face should be marked as reviewed but not dismissed
        face = repository.get_unidentified_face(face_id)
        assert face.reviewed is True
        assert face.dismissed is False

    def test_mark_reviewed_not_found(self, client):
        """Test marking non-existent face as reviewed."""
        response = client.post("/api/v1/unidentified-faces/999/review")
        assert response.status_code == 404


def test_match_scores_rank_the_two_closest_persons_in_the_current_gallery(repository, client, tmp_path):
    """Review aid (2026-10-19): the quality score alone did not tell whether a suggestion
    helps — a noisy Judit face close to Virag (0.21) made one more Virag face look like Judit.
    The API shows the two closest persons by the live scoring rule; an assigned face
    updates the gallery for the next request."""
    rng = np.random.default_rng(1)
    judit_dir, virag_dir = rng.standard_normal(512), rng.standard_normal(512)
    judit, virag = repository.create_person("Judit"), repository.create_person("Virag")
    repository.add_face_embedding(judit.id, judit_dir.astype(np.float32))
    repository.add_face_embedding(virag.id, virag_dir.astype(np.float32))
    face_emb = (judit_dir + 0.4 * virag_dir).astype(np.float32)
    face = repository.add_unidentified_face("hallway", face_emb, str(tmp_path / "f.jpg"), 0.5)

    (row,) = client.get("/api/v1/unidentified-faces").json()
    names = [s["person_name"] for s in row["match_scores"]]
    scores = [s["score"] for s in row["match_scores"]]
    assert names == ["Judit", "Virag"] and scores[0] > scores[1] > 0.2
    assert client.get(f"/api/v1/unidentified-faces/{face.id}").json()["match_scores"] == row["match_scores"]


def test_match_scores_empty_without_gallery(repository, client, tmp_path):
    repository.add_unidentified_face("hallway", np.ones(512, np.float32), str(tmp_path / "f.jpg"), 0.5)
    (row,) = client.get("/api/v1/unidentified-faces").json()
    assert row["match_scores"] == []


class TestEnrolmentValue:
    """'Recommended +N' / 'Risky' (2026-10-19 offline, 150 labelled candidates): coverage =
    other tracks' current suggestions >= 0.5 similar predicted the real gain of enrolling
    (Spearman 0.71; quality 0.14); the one seriously harmful candidate had a runner-up
    person ahead of its own (Judit 0.17 vs Virag 0.20)."""

    @staticmethod
    def _scores(*pairs):
        from src.api.routes.unidentified_faces import MatchScore
        return [MatchScore(person_id=i, person_name=n, score=s) for i, (n, s) in enumerate(pairs)]

    def test_label_rules(self):
        from src.api.routes.unidentified_faces import enrolment_label
        from src.config import UnidentifiedFacesConfig
        cfg = UnidentifiedFacesConfig()
        assert enrolment_label(self._scores(("Ati", 0.53), ("Judit", 0.15)), 8, 0.55, cfg) == ("recommended", "covers_failures")
        # live #1399: useful (coverage 5) but blurred and grainy (quality 0.45) -> not for the gallery
        assert enrolment_label(self._scores(("Ati", 0.48), ("Judit", 0.08)), 5, 0.45, cfg) == ("review", "low_quality")
        assert enrolment_label(self._scores(("Judit", 0.52), ("Virag", 0.16)), 1, 0.54, cfg) == ("review", "low_coverage")  # #1393
        assert enrolment_label(self._scores(("Judit", 0.17), ("Virag", 0.20)), 9, 0.6, cfg) == ("risky", "close_runner_up")
        assert enrolment_label(self._scores(("Judit", 0.30), ("Virag", 0.22)), 9, 0.6, cfg)[0] == "risky"
        assert enrolment_label([], 9, 0.6, cfg) == ("review", "no_gallery")

    def test_coverage_counts_similar_suggestions_of_other_tracks(self, repository, client, tmp_path):
        rng = np.random.default_rng(3)
        ati_dir, other = rng.standard_normal(512), rng.standard_normal(512)
        ati = repository.create_person("Ati")
        repository.add_face_embedding(ati.id, (ati_dir + 1.5 * rng.standard_normal(512)).astype(np.float32))
        situation = ati_dir + 0.3 * rng.standard_normal(512)          # a recurring view of Ati, below the threshold
        faces = [repository.add_unidentified_face(
            "living_room", (situation + 0.2 * rng.standard_normal(512)).astype(np.float32),
            str(tmp_path / f"{i}.jpg"), 0.55, track_id=f"global_{i}") for i in range(6)]
        same_track = repository.add_unidentified_face(
            "living_room", situation.astype(np.float32), str(tmp_path / "s.jpg"), 0.4, track_id="global_0")
        repository.add_unidentified_face("hallway", other.astype(np.float32), str(tmp_path / "o.jpg"), 0.4,
                                         track_id="global_9")
        body = client.get(f"/api/v1/unidentified-faces/{faces[0].id}").json()
        assert body["coverage"] == 5                      # the 5 other tracks, not its own track's crop
        assert body["enrol_label"] == "recommended"
        same = client.get(f"/api/v1/unidentified-faces/{same_track.id}").json()
        assert same["coverage"] == 5 and (same["enrol_label"], same["enrol_reason"]) == ("review", "low_quality")

    def test_assignment_logs_the_enrolment_value(self, repository, client, tmp_path):
        rng = np.random.default_rng(4)
        judit = repository.create_person("Judit")
        repository.add_face_embedding(judit.id, rng.standard_normal(512).astype(np.float32))
        face = repository.add_unidentified_face("hallway", rng.standard_normal(512).astype(np.float32),
                                                str(tmp_path / "f.jpg"), 0.5, track_id="global_1")
        assert client.post(f"/api/v1/unidentified-faces/{face.id}/assign", json={"person_id": judit.id}).status_code == 200
        (event,) = repository.get_events(event_type="face_enrolled")
        assert event.extra_data["suggestion_id"] == face.id and event.extra_data["enrol_label"] == "review"
        assert event.extra_data["coverage"] == 0 and event.extra_data["match_scores"][0]["person_name"] == "Judit"


def test_coverage_counts_archived_crops_once_per_track(repository, client, tmp_path):
    """Archived crops (superseded by a better crop of the same track) widen the pool to more
    tracks / days (live 2026-10-19: 130 -> 380 rows), but a track counts once."""
    rng = np.random.default_rng(6)
    ati_dir = rng.standard_normal(512)
    ati = repository.create_person("Ati")
    repository.add_face_embedding(ati.id, (ati_dir + 1.5 * rng.standard_normal(512)).astype(np.float32))
    situation = ati_dir + 0.3 * rng.standard_normal(512)
    crop = lambda name, track: repository.add_unidentified_face(
        "living_room", (situation + 0.2 * rng.standard_normal(512)).astype(np.float32),
        str(tmp_path / f"{name}.jpg"), 0.55, track_id=track)
    face = crop("f", "global_0")
    for i in range(1, 4):                       # 3 other tracks, each: best crop + 2 archived crops
        best = crop(f"b{i}", f"global_{i}")
        for k in range(2):
            repository.archive_unidentified_face(crop(f"a{i}{k}", f"global_{i}").id, best.id)
    old_track_best = crop("x", "global_7")      # a 4th track known only from an archived crop
    repository.archive_unidentified_face(crop("y", "global_8").id, old_track_best.id)
    body = client.get(f"/api/v1/unidentified-faces/{face.id}").json()
    assert body["coverage"] == 5                # global_1..3, global_7, global_8 — not 3 x 3 rows


def test_coverage_ignores_suggestions_the_gallery_now_recognises(repository, client, tmp_path):
    """Live 2026-10-19: 8 of the 11 faces #1403 'covered' were already accepted by the grown
    gallery — they are no failure any more and must not inflate its +N."""
    rng = np.random.default_rng(5)
    situation = rng.standard_normal(512)
    ati = repository.create_person("Ati")
    repository.add_face_embedding(ati.id, situation.astype(np.float32))      # gallery already has this view
    faces = [repository.add_unidentified_face(
        "living_room", (situation + 0.2 * rng.standard_normal(512)).astype(np.float32),
        str(tmp_path / f"{i}.jpg"), 0.4, track_id=f"global_{i}") for i in range(6)]
    body = client.get(f"/api/v1/unidentified-faces/{faces[0].id}").json()
    assert body["match_scores"][0]["score"] > 0.9
    assert body["coverage"] == 0 and body["enrol_label"] == "review"
