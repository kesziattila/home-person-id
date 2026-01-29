import unittest
import numpy as np
from src.recognition.identification_manager import IdentificationManager, TrackIdentity
from src.recognition.reid_gallery import ReIDGalleryManager, GalleryEntry, EmbeddingWithCrop
from src.visualization.render import TrackRenderer
from src.config import Config
from unittest.mock import MagicMock

class TestReIDClosestMatch(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.config.reid.similarity_threshold = 0.8  # High threshold
        
        # Mock ReIDExtractor
        self.reid_extractor = MagicMock()
        
        # Create Gallery Manager
        self.gallery_manager = ReIDGalleryManager(self.reid_extractor, self.config.reid)
        
        # Add a person to the gallery
        dummy_emb = np.zeros(512).astype(np.float32)
        dummy_emb[0] = 1.0  # Simple embedding
        
        entry = GalleryEntry(person_name="John Doe")
        # EmbeddingWithCrop expects embedding and optional crop_path
        entry.entries.append(EmbeddingWithCrop(embedding=dummy_emb))
        self.gallery_manager._gallery["John Doe"] = entry
        self.gallery_manager.similarity_threshold = 0.8
        
        # Create Identification Manager
        self.id_manager = IdentificationManager(self.config, reid_extractor=self.reid_extractor)
        self.id_manager._reid_gallery_manager = self.gallery_manager
        
        self.renderer = TrackRenderer()

    def test_reid_closest_match_below_threshold(self):
        """Test that closest match is shown even if below threshold."""
        # Query embedding slightly different from gallery
        query_emb = np.zeros(512).astype(np.float32)
        query_emb[0] = 1.0
        query_emb[1] = 1.0  # Vector [1, 1, 0...]
        # Gallery is [1, 0, 0...]
        # Cosine similarity: (1*1 + 1*0) / (sqrt(2) * 1) = 1/sqrt(2) ≈ 0.707
        
        self.reid_extractor.extract.return_value = (query_emb, 0.9)
        
        track_id = "global_1"
        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        crop[:, :, 0] = 255  # Solid Blue (non-grayscale)
        
        # Try match
        self.id_manager.try_reid_match(track_id, crop, num_persons=1)
        
        identity = self.id_manager.get_identity(track_id)
        
        # Should NOT be identified (0.707 < 0.8)
        self.assertFalse(identity.is_identified)
        self.assertEqual(identity.person_name, None)
        
        # But should have reid_info
        self.assertIsNotNone(identity.reid_info)
        self.assertEqual(identity.reid_info[0], "John Doe")
        self.assertAlmostEqual(identity.reid_info[1], 1.0/np.sqrt(2), places=5)
        
        # Test rendering
        label, color = self.renderer.get_track_label_and_color(
            track_id=track_id,
            identity=identity,
            camera_id="cam1",
            bbox=(0, 0, 10, 10),
            frame_w=100,
            frame_h=100
        )
        
        self.assertIn("(?)John Doe(0.71)", label)
        self.assertEqual(color, TrackRenderer.COLOR_UNIDENTIFIED)

    def test_reid_closest_match_above_threshold(self):
        """Test that confirmed match is shown normally when above threshold."""
        # Perfect match
        query_emb = np.zeros(512)
        query_emb[0] = 1.0
        
        self.reid_extractor.extract.return_value = (query_emb, 0.9)
        
        track_id = "global_1"
        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        crop[:, :, 0] = 255  # Solid Blue (non-grayscale)
        
        # Try match
        self.id_manager.try_reid_match(track_id, crop, num_persons=1)
        
        identity = self.id_manager.get_identity(track_id)
        
        # SHOULD be identified
        self.assertTrue(identity.is_identified)
        self.assertEqual(identity.person_name, "John Doe")
        
        # Test rendering
        label, color = self.renderer.get_track_label_and_color(
            track_id=track_id,
            identity=identity,
            camera_id="cam1",
            bbox=(0, 0, 10, 10),
            frame_w=100,
            frame_h=100
        )
        
        self.assertIn("John Doe", label)
        self.assertIn("R:1.00", label)
        self.assertEqual(color, TrackRenderer.COLOR_REID_IDENTIFIED)

if __name__ == "__main__":
    unittest.main()
