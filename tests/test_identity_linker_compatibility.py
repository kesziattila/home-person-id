import unittest
import numpy as np
from src.recognition.identity_linker import IdentityLinker, IdentificationResult
from src.config import FaceRecognitionConfig, ReIDConfig
from unittest.mock import MagicMock

class TestIdentityLinkerCompatibility(unittest.TestCase):
    def setUp(self):
        self.face_config = FaceRecognitionConfig(enabled=True)
        self.reid_config = ReIDConfig(enabled=True)
        self.repository = MagicMock()
        
        # Mock face recognizer and reid extractor
        self.face_recognizer = MagicMock()
        self.reid_extractor = MagicMock()
        self.reid_extractor.extract.return_value = (np.zeros(512), 0.9)
        
        self.linker = IdentityLinker(
            face_config=self.face_config,
            reid_config=self.reid_config,
            repository=self.repository,
            face_recognizer=self.face_recognizer,
            reid_extractor=self.reid_extractor
        )

    def test_identification_result_fields(self):
        """Test that IdentificationResult has all fields required by TrackRenderer."""
        result = IdentificationResult()
        
        # Check required fields for visualization compatibility with TrackIdentity
        self.assertTrue(hasattr(result, 'reid_score'))
        self.assertTrue(hasattr(result, 'reid_info'))
        self.assertTrue(hasattr(result, 'face_info'))
        self.assertTrue(hasattr(result, 'is_reid_identified'))
        self.assertTrue(hasattr(result, 'has_multiple_faces'))
        
        # Verify defaults
        self.assertEqual(result.reid_score, -1.0)
        self.assertIsNone(result.reid_info)
        self.assertIsNone(result.face_info)
        self.assertFalse(result.is_reid_identified)
        self.assertFalse(result.has_multiple_faces)

    def test_get_identity_populates_fields(self):
        """Test that get_identity initializes the compatibility fields."""
        track_id = "global_1"
        self.linker.register_track(track_id)
        
        result = self.linker.get_identity(track_id)
        self.assertIsNotNone(result)
        # reid_info should be initialized to None (not causing AttributeError)
        self.assertIsNone(result.reid_info)
        self.assertFalse(result.has_multiple_faces)

    def test_process_track_returns_compatibility_fields(self):
        """Test that process_track returns result with compatibility fields."""
        track_id = "global_1"
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        person_crop = np.zeros((50, 50, 3), dtype=np.uint8)
        bbox = (0, 0, 50, 50)
        
        result = self.linker.process_track(
            global_track_id=track_id,
            frame=frame,
            person_crop=person_crop,
            bbox=bbox
        )
        
        self.assertIsNotNone(result)
        self.assertIsNone(result.reid_info)
        self.assertFalse(result.has_multiple_faces)

if __name__ == "__main__":
    unittest.main()
