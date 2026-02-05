import pytest
import os
import shutil
from pathlib import Path
import numpy as np
from src.database.repository import Repository
from src.database.models import Base, UnidentifiedFace, Person

@pytest.fixture
def repo():
    repo = Repository(":memory:")
    Base.metadata.create_all(repo._engine)
    return repo

def test_assign_unidentified_face_moves_file(repo, tmp_path):
    # Setup directories
    unidentified_dir = tmp_path / "unidentified"
    unidentified_dir.mkdir()
    faces_dir = tmp_path / "faces"
    faces_dir.mkdir()
    
    # Create a dummy image
    img_path = unidentified_dir / "test_face.jpg"
    img_path.write_bytes(b"fake image data")
    
    # Create person and unidentified face in DB
    person = repo.create_person("Test Person")
    
    with repo.get_session() as session:
        face = UnidentifiedFace(
            camera_id="cam1",
            embedding=np.zeros(512, dtype=np.float32).tobytes(),
            image_path=str(img_path),
            quality_score=0.9
        )
        session.add(face)
        session.commit()
        face_id = face.id

    # Simulate what the API does
    person_dir = faces_dir / f"person_{person.id}"
    person_dir.mkdir(parents=True, exist_ok=True)
    dest_path = person_dir / "test_face.jpg"
    
    # Move file
    shutil.move(str(img_path), str(dest_path))
    
    # Call repository
    embedding = repo.assign_unidentified_face_to_person(face_id, person.id, new_image_path=str(dest_path))
    
    # Verify
    assert embedding is not None
    assert embedding.person_id == person.id
    assert embedding.source_image == str(dest_path)
    assert os.path.exists(str(dest_path))
    assert not os.path.exists(str(img_path))
    
    with repo.get_session() as session:
        assert session.query(UnidentifiedFace).count() == 0
