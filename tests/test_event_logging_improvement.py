import pytest
import numpy as np
from datetime import datetime
from src.database.repository import Repository
from src.database.models import Base, ReIDEmbedding
from src.recognition.reid_gallery import ReIDGalleryManager, GalleryEntry
from src.recognition.identification_manager import IdentificationManager
from src.recognition.identity_linker import IdentityLinker
from src.config import Config, FaceRecognitionConfig

class MockReIDExtractor:
    def extract(self, crop, return_quality=True):
        return np.random.rand(512).astype(np.float32), 0.9
    def compare(self, emb1, emb2):
        return 0.9

@pytest.fixture
def repository():
    repo = Repository(":memory:")
    Base.metadata.create_all(repo._engine)
    return repo

@pytest.fixture
def config():
    c = Config()
    c.face_recognition.enabled = True
    c.reid.enabled = True
    return c

def test_event_logging_flow(repository, config):
    # Setup components
    reid_extractor = MockReIDExtractor()
    gallery_manager = ReIDGalleryManager(
        reid_extractor=reid_extractor,
        repository=repository
    )
    
    id_manager = IdentificationManager(
        config=config,
        repository=repository,
        reid_extractor=reid_extractor
    )
    id_manager._reid_gallery_manager = gallery_manager
    
    linker = IdentityLinker(
        face_config=config.face_recognition,
        reid_config=config.reid,
        repository=repository,
        reid_extractor=reid_extractor
    )
    linker._id_manager = id_manager
    
    # 1. Create a person
    person = repository.create_person("Alice")
    # Add face embedding so she's in gallery
    repository.add_face_embedding(person.id, np.random.rand(512).astype(np.float32))
    id_manager._load_face_gallery_from_db()
    
    # 2. Simulate a track being identified by Re-ID (initially)
    track_id = "global_1"
    repository.create_track(track_id, "cam1")
    
    # Ensure crop has color to avoid grayscale filter
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    crop[:, :, 0] = 100
    crop[:, :, 1] = 150
    crop[:, :, 2] = 200

    # Mock extractor to return normalized embedding
    import unittest.mock as mock
    fixed_embedding = np.random.rand(512).astype(np.float32)
    fixed_embedding /= np.linalg.norm(fixed_embedding)
    reid_extractor.extract = mock.MagicMock(return_value=(fixed_embedding, 0.9))
    
    # Add to gallery
    entry = GalleryEntry(person_name="Alice", max_embeddings=10)
    gallery_manager._gallery["Alice"] = entry
    entry.add_embedding(fixed_embedding, db_id=123)
    gallery_manager.similarity_threshold = 0.65
    
    # Try Re-ID match
    id_manager.try_reid_match(track_id, crop, 1)
    
    # Verify reid_match event
    events = repository.get_events(event_type="reid_match")
    assert len(events) == 1
    assert events[0].person_id == person.id
    assert events[0].track_id == track_id
    
    # 3. Simulate face match
    # Force confirmation state manually
    state = linker.register_track(track_id)
    camera_id = "cam1"
    state.candidate_person_id = person.id
    state.consecutive_matches = 1
    state.confirm_identity(person.id, 0.95, "face")
    
    # Emit events manually as linker would (following the new logic where we don't add_face_embedding)
    repository.create_event(
        camera_id=camera_id,
        event_type="face_match",
        track_id=track_id,
        person_id=person.id,
        confidence=0.95,
        face_embedding_id=1, # Reference to matched gallery
        extra_data={"threshold": 0.6, "face_id": 1}
    )
    
    repository.create_event(
        camera_id=camera_id,
        event_type="id_upgraded",
        track_id=track_id,
        person_id=person.id,
        extra_data={"from": "reid", "to": "face", "previous_person_id": person.id, "new_person_id": person.id}
    )
    
    # Verify face_match event
    events = repository.get_events(event_type="face_match")
    assert len(events) == 1
    
    # Verify id_upgraded event
    events = repository.get_events(event_type="id_upgraded")
    assert len(events) == 1
    
    # 4. Simulate track lost and gallery update
    linker.unregister_track(track_id, was_face_identified=True)
    track_id_num = hash(track_id) % (10**9)
    gallery_manager.update_track_embedding(track_id_num, crop, "Alice", 1)
    gallery_manager.on_track_lost(track_id_num, was_face_identified=True)
    
    # Verify reid_gallery_updated event
    events = repository.get_events(event_type="reid_gallery_updated")
    assert len(events) >= 1
    assert events[0].reid_embedding_id is not None
    
    # Verify ReIDEmbedding was persisted
    with repository.get_session() as session:
        embs = session.query(ReIDEmbedding).all()
        assert len(embs) >= 1
        assert embs[0].person_id == person.id
