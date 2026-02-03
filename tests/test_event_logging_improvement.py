import pytest
import numpy as np
from datetime import datetime, timedelta
from src.database.repository import Repository
from src.database.models import Base
from src.recognition.reid_gallery import ReIDGalleryManager, GalleryEntry
from src.recognition.identification_manager import IdentificationManager
from src.recognition.identity_linker import IdentityLinker
from src.config import Config, FaceRecognitionConfig, ReIDConfig

class MockReIDExtractor:
    def extract(self, crop, return_quality=True):
        return np.random.rand(512).astype(np.float32), 0.9

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
    # Ensure track exists in DB so FK constraints or joins don't fail (though sqlite doesn't always enforce)
    repository.create_track(track_id, "cam1")
    
    # Use real objects instead of mocks where possible to ensure events are fired
    # We want match_result.matched = True
    # The try_reid_match uses self.reid_gallery_manager.match_new_track
    
    # Try Re-ID match
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    crop[0, 0] = [100, 100, 100] # Ensure not grayscale

    # We can use a real gallery match by adding to the gallery
    # gallery_manager.match_new_track will extract embedding from crop and compare
    # Mock extractor to return the same embedding as in gallery
    import unittest.mock as mock
    fixed_embedding = np.random.rand(512).astype(np.float32)
    reid_extractor.extract = mock.MagicMock(return_value=(fixed_embedding, 0.9))
    
    # Add to gallery
    entry = GalleryEntry(person_name="Alice")
    gallery_manager._gallery["Alice"] = entry
    entry.add_embedding(fixed_embedding, db_id=123)
    gallery_manager.similarity_threshold = 0.65
    
    id_manager.try_reid_match(track_id, crop, 1)
    
    # Verify reid_match event
    events = repository.get_events(event_type="reid_match")
    assert len(events) == 1
    assert events[0].person_id == person.id
    assert events[0].track_id == track_id
    
    # 3. Simulate Face identification (upgrade)
    class MockFaceRecognizer:
        def __init__(self):
            self.config = FaceRecognitionConfig()
        def detect_faces(self, crop, min_face_size=None):
            from src.recognition.face_recognizer import FaceDetectionResult, Face
            f = Face(bbox=(0,0,10,10), confidence=0.9)
            f.embedding = np.random.rand(512).astype(np.float32)
            return FaceDetectionResult(faces=[f])
        def compare_embeddings_batch(self, query, gallery):
            return np.array([0.95]) # High similarity
            
    id_manager._face_recognizer = MockFaceRecognizer()
    
    # IdentityLinker handles consecutive matches
    # First match
    linker.process_track(track_id, np.zeros((200, 200, 3), dtype=np.uint8), crop, (0,0,100,100))
    # Second match confirms
    linker.process_track(track_id, np.zeros((200, 200, 3), dtype=np.uint8), crop, (0,0,100,100)) 
    
    # Verify face_match event
    events = repository.get_events(event_type="face_match")
    assert len(events) == 1
    assert events[0].face_embedding_id is not None
    
    # Verify id_upgraded event
    events = repository.get_events(event_type="id_upgraded")
    assert len(events) == 1
    assert events[0].extra_data["from"] == "reid"
    assert events[0].extra_data["to"] == "face"
    
    # 4. Simulate track lost and gallery update
    linker.unregister_track(track_id, was_face_identified=True)
    
    # Prepare track data for gallery update
    track_id_num = hash(track_id) % (10**9)
    gallery_manager.update_track_embedding(track_id_num, crop, "Alice", 1)
    
    # Store to gallery
    gallery_manager.on_track_lost(track_id_num, was_face_identified=True)
    
    # Verify reid_gallery_updated event
    events = repository.get_events(event_type="reid_gallery_updated")
    assert len(events) >= 1
    assert events[0].reid_embedding_id is not None
    
    # Verify ReIDEmbedding was persisted
    with repository.get_session() as session:
        from src.database.models import ReIDEmbedding
        embs = session.query(ReIDEmbedding).all()
        assert len(embs) >= 1
        assert embs[0].person_id == person.id
