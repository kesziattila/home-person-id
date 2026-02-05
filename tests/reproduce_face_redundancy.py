import pytest
import numpy as np
from src.database.repository import Repository
from src.database.models import Base, FaceEmbedding
from src.recognition.identification_manager import IdentificationManager
from src.recognition.identity_linker import IdentityLinker
from src.config import Config, FaceRecognitionConfig

class MockFaceRecognizer:
    def __init__(self, threshold=0.6):
        self.config = FaceRecognitionConfig(similarity_threshold=threshold)
    def detect_faces(self, crop, min_face_size=None):
        from src.recognition.face_recognizer import FaceDetectionResult, Face
        f = Face(bbox=(0,0,10,10), confidence=0.9)
        f.embedding = np.random.rand(512).astype(np.float32)
        return FaceDetectionResult(faces=[f], frame_shape=crop.shape)
    def compare_embeddings_batch(self, query, gallery):
        return np.array([0.95]) # High similarity
    def extract_embedding(self, image, face):
        return face.embedding
    def warmup(self):
        pass

@pytest.fixture
def repository():
    repo = Repository(":memory:")
    Base.metadata.create_all(repo._engine)
    return repo

@pytest.fixture
def config():
    c = Config()
    c.face_recognition.enabled = True
    c.face_recognition.similarity_threshold = 0.6
    return c

def test_redundant_face_embedding_storage(repository, config):
    # Setup components
    id_manager = IdentificationManager(
        config=config,
        repository=repository
    )
    
    linker = IdentityLinker(
        face_config=config.face_recognition,
        reid_config=config.reid,
        repository=repository,
    )
    linker._id_manager = id_manager
    id_manager._face_recognizer = MockFaceRecognizer(threshold=config.face_recognition.similarity_threshold)
    
    # 1. Create a person with one face embedding
    person = repository.create_person("Alice")
    initial_embedding = np.random.rand(512).astype(np.float32)
    repository.add_face_embedding(person.id, initial_embedding)
    id_manager._load_face_gallery_from_db()
    
    with repository.get_session() as session:
        count = session.query(FaceEmbedding).filter(FaceEmbedding.person_id == person.id).count()
        assert count == 1
    
    # 2. Simulate multiple matches for the same person
    track_id = "global_1"
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = (0, 0, 100, 100)
    
    # IdentityLinker requires 2 consecutive matches to confirm
    print(f"Bbox: {bbox}, crop shape: {crop.shape}")
    print(f"Face config: {linker.face_config}")
    
    # Manually trigger identification manager recognition to see what it says
    print("Trying direct face recognition via id_manager...")
    res = id_manager.try_face_recognition("global_1", crop, 1, 1)
    print(f"Direct recognition result: {res}")
    
    linker.process_track(track_id, crop, crop, bbox, force_face_check=True)
    print(f"Candidate: {linker._track_states[track_id].candidate_person_id}, consecutive: {linker._track_states[track_id].consecutive_matches}")
    linker.process_track(track_id, crop, crop, bbox, force_face_check=True) # Confirmed here -> add_face_embedding called
    print(f"Confirmed: {linker._track_states[track_id].person_id}")
    
    with repository.get_session() as session:
        count = session.query(FaceEmbedding).filter(FaceEmbedding.person_id == person.id).count()
        print(f"Count after 1 confirmed track: {count}")
        # SHOULD BE 1 - no automatic addition
        assert count == 1 
    
    # 3. Simulate another confirmed track for the same person
    track_id_2 = "global_2"
    linker.process_track(track_id_2, crop, crop, bbox, force_face_check=True)
    linker.process_track(track_id_2, crop, crop, bbox, force_face_check=True) # Confirmed here
    
    with repository.get_session() as session:
        count = session.query(FaceEmbedding).filter(FaceEmbedding.person_id == person.id).count()
        print(f"Count after 2 confirmed tracks: {count}")
        # SHOULD STILL BE 1
        assert count == 1
