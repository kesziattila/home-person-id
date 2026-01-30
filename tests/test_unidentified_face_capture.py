"""Test unidentified face capture flow."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from src.recognition.identification_manager import IdentificationManager
from src.config import (
    Config,
    FaceRecognitionConfig,
    ReIDConfig,
    DetectionConfig,
    MotionConfig,
    TrackingConfig,
    DatabaseConfig,
    SnapshotConfig,
    LoggingConfig,
    UnidentifiedFacesConfig,
    CameraTopologyConfig,
)


class TestUnidentifiedFaceCapture:
    """Test that unidentified faces are captured correctly."""

    @pytest.fixture
    def config(self):
        """Create minimal config for testing."""
        return Config(
            cameras=[],
            camera_topology=CameraTopologyConfig(overlaps=[]),
            zones=[],
            detection=DetectionConfig(),
            motion=MotionConfig(),
            tracking=TrackingConfig(),
            face_recognition=FaceRecognitionConfig(
                enabled=True,
                model="buffalo_l",
                similarity_threshold=0.6,
                min_face_size=40,
            ),
            reid=ReIDConfig(enabled=False),
            database=DatabaseConfig(path=":memory:"),
            snapshots=SnapshotConfig(),
            logging=LoggingConfig(),
            unidentified_faces=UnidentifiedFacesConfig(enabled=True),
        )

    @pytest.fixture
    def mock_face_recognizer(self):
        """Create mock face recognizer."""
        recognizer = MagicMock()

        # Create a mock face with embedding
        mock_face = MagicMock()
        mock_face.embedding = np.random.randn(512).astype(np.float32)
        mock_face.bbox = np.array([10, 10, 50, 50])  # x1, y1, x2, y2

        # Mock face detection result
        mock_result = MagicMock()
        mock_result.faces = [mock_face]

        recognizer.detect_faces.return_value = mock_result
        recognizer.compare_embeddings.return_value = 0.3  # Below threshold

        return recognizer

    @pytest.fixture
    def mock_unidentified_manager(self):
        """Create mock unidentified face manager."""
        manager = MagicMock()
        manager.submit.return_value = True
        return manager

    def test_unidentified_face_submitted_when_below_threshold(
        self, config, mock_face_recognizer, mock_unidentified_manager
    ):
        """Test that unidentified face is submitted when below threshold."""
        # Create manager with mocked components
        manager = IdentificationManager(
            config=config,
            repository=None,
            face_gallery=[
                (1, "John", np.random.randn(512).astype(np.float32))
            ],
            unidentified_face_manager=mock_unidentified_manager,
        )

        # Inject mock face recognizer
        manager._face_recognizer = mock_face_recognizer

        # Create test image
        person_crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Call try_face_recognition
        result = manager.try_face_recognition(
            track_id="test_track",
            person_crop=person_crop,
            local_track_id=1,
            num_persons=1,
            camera_id="test_camera",
        )

        # Should return False (not identified)
        assert result is False

        # Unidentified face manager should have been called
        assert mock_unidentified_manager.submit.called, \
            "UnidentifiedFaceManager.submit() was NOT called!"

        # Check the call arguments
        call_args = mock_unidentified_manager.submit.call_args
        assert call_args.kwargs["camera_id"] == "test_camera"
        assert call_args.kwargs["track_id"] == "test_track"
        assert call_args.kwargs["embedding"] is not None
        print(f"submit() called with: {call_args.kwargs.keys()}")

    def test_unidentified_face_submitted_when_gallery_empty(
        self, config, mock_face_recognizer, mock_unidentified_manager
    ):
        """Test that unidentified face is submitted when gallery is empty."""
        # Create manager with empty gallery
        manager = IdentificationManager(
            config=config,
            repository=None,
            face_gallery=[],  # Empty gallery
            unidentified_face_manager=mock_unidentified_manager,
        )

        # Inject mock face recognizer
        manager._face_recognizer = mock_face_recognizer

        # Create test image
        person_crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Call try_face_recognition
        result = manager.try_face_recognition(
            track_id="test_track",
            person_crop=person_crop,
            local_track_id=1,
            num_persons=1,
            camera_id="test_camera",
        )

        # Should return False (not identified)
        assert result is False

        # Unidentified face manager should have been called
        assert mock_unidentified_manager.submit.called, \
            "UnidentifiedFaceManager.submit() was NOT called with empty gallery!"

    def test_unidentified_face_not_submitted_without_camera_id(
        self, config, mock_face_recognizer, mock_unidentified_manager
    ):
        """Test that unidentified face is NOT submitted without camera_id."""
        manager = IdentificationManager(
            config=config,
            repository=None,
            face_gallery=[
                (1, "John", np.random.randn(512).astype(np.float32))
            ],
            unidentified_face_manager=mock_unidentified_manager,
        )

        manager._face_recognizer = mock_face_recognizer
        person_crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Call WITHOUT camera_id
        result = manager.try_face_recognition(
            track_id="test_track",
            person_crop=person_crop,
            local_track_id=1,
            num_persons=1,
            camera_id=None,  # No camera_id
        )

        # Should return False
        assert result is False

        # Unidentified face manager should NOT have been called
        assert not mock_unidentified_manager.submit.called, \
            "UnidentifiedFaceManager.submit() should NOT be called without camera_id!"

    def test_face_identified_above_threshold(
        self, config, mock_face_recognizer, mock_unidentified_manager
    ):
        """Test that face is identified when above threshold."""
        # Make compare_embeddings return high score
        mock_face_recognizer.compare_embeddings.return_value = 0.8  # Above 0.6 threshold

        manager = IdentificationManager(
            config=config,
            repository=None,
            face_gallery=[
                (1, "John", np.random.randn(512).astype(np.float32))
            ],
            unidentified_face_manager=mock_unidentified_manager,
        )

        manager._face_recognizer = mock_face_recognizer
        person_crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        result = manager.try_face_recognition(
            track_id="test_track",
            person_crop=person_crop,
            local_track_id=1,
            num_persons=1,
            camera_id="test_camera",
        )

        # Should return True (identified)
        assert result is True

        # Unidentified face manager should NOT have been called
        assert not mock_unidentified_manager.submit.called, \
            "UnidentifiedFaceManager.submit() should NOT be called when face is identified!"

        # Check identity was set
        identity = manager.get_identity("test_track")
        assert identity.person_name == "John"
        assert identity.is_face_identified is True


class TestUnidentifiedFaceManagerSubmit:
    """Test UnidentifiedFaceManager.submit() directly."""

    def test_submit_adds_to_queue(self):
        """Test that submit adds task to queue."""
        from src.config import UnidentifiedFacesConfig
        from src.recognition.unidentified_face_manager import UnidentifiedFaceManager

        config = UnidentifiedFacesConfig(
            enabled=True,
            images_path="/tmp/test_unidentified",
            max_per_camera=20,
            min_quality_score=0.3,
            min_face_size=60,
        )

        # Create manager with mock repository
        mock_repo = MagicMock()
        manager = UnidentifiedFaceManager(config, mock_repo)

        # Don't start the worker thread for this test

        # Create test data
        face_crop = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
        embedding = np.random.randn(512).astype(np.float32)

        # Submit
        result = manager.submit(
            camera_id="test_camera",
            face_crop=face_crop,
            embedding=embedding,
            face_bbox=(10, 10, 70, 70),
            track_id="test_track",
            best_match_person_id=1,
            best_match_score=0.4,
        )

        assert result is True, "submit() should return True"
        assert not manager._queue.empty(), "Queue should have a task"

        # Get the task from queue
        task = manager._queue.get_nowait()
        assert task.camera_id == "test_camera"
        assert task.track_id == "test_track"
        print(f"Task in queue: camera={task.camera_id}, track={task.track_id}")

    def test_submit_disabled(self):
        """Test that submit returns False when disabled."""
        from src.config import UnidentifiedFacesConfig
        from src.recognition.unidentified_face_manager import UnidentifiedFaceManager

        config = UnidentifiedFacesConfig(
            enabled=False,  # Disabled
        )

        mock_repo = MagicMock()
        manager = UnidentifiedFaceManager(config, mock_repo)

        face_crop = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
        embedding = np.random.randn(512).astype(np.float32)

        result = manager.submit(
            camera_id="test_camera",
            face_crop=face_crop,
            embedding=embedding,
            face_bbox=(10, 10, 70, 70),
        )

        assert result is False, "submit() should return False when disabled"

    def test_submit_invalid_embedding(self):
        """Test that submit returns False with invalid embedding."""
        from src.config import UnidentifiedFacesConfig
        from src.recognition.unidentified_face_manager import UnidentifiedFaceManager

        config = UnidentifiedFacesConfig(enabled=True)
        mock_repo = MagicMock()
        manager = UnidentifiedFaceManager(config, mock_repo)

        face_crop = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
        bad_embedding = np.random.randn(256).astype(np.float32)  # Wrong size

        result = manager.submit(
            camera_id="test_camera",
            face_crop=face_crop,
            embedding=bad_embedding,
            face_bbox=(10, 10, 70, 70),
        )

        assert result is False, "submit() should return False with invalid embedding size"


class TestCropUtilities:
    """Test crop utility functions."""

    def test_crop_with_margin(self):
        """Test crop_with_margin utility."""
        from src.utils.image_utils import crop_with_margin

        # Create test image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bbox = (30, 30, 60, 60)

        crop, adjusted_bbox = crop_with_margin(image, bbox, margin_ratio=0.3)

        # Should have expanded bbox
        assert adjusted_bbox[0] < 30  # x1 moved left
        assert adjusted_bbox[1] < 30  # y1 moved up
        assert adjusted_bbox[2] > 60  # x2 moved right
        assert adjusted_bbox[3] > 60  # y2 moved down

        # Crop should not be empty
        assert crop.size > 0

    def test_crop_with_margin_bounds_check(self):
        """Test that crop_with_margin respects image bounds."""
        from src.utils.image_utils import crop_with_margin

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # Bbox at edge
        bbox = (0, 0, 30, 30)

        crop, adjusted_bbox = crop_with_margin(image, bbox, margin_ratio=0.5)

        # Should stay within bounds
        assert adjusted_bbox[0] >= 0
        assert adjusted_bbox[1] >= 0
        assert adjusted_bbox[2] <= 100
        assert adjusted_bbox[3] <= 100

    def test_crop_bbox(self):
        """Test crop_bbox utility."""
        from src.utils.image_utils import crop_bbox

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bbox = (10, 20, 50, 60)

        crop = crop_bbox(image, bbox)

        # Check dimensions
        assert crop.shape == (40, 40, 3)  # 60-20=40, 50-10=40

    def test_crop_face_region(self):
        """Test crop_face_region utility."""
        from src.utils.image_utils import crop_face_region

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bbox = (10, 20, 90, 80)  # Full body bbox

        face_region = crop_face_region(image, bbox, height_ratio=0.5)

        # Should be upper half
        expected_height = int((80 - 20) * 0.5)  # 30
        assert face_region.shape[0] == expected_height
        assert face_region.shape[1] == 80  # 90-10


class TestBlurDetection:
    """Test blur detection for quality scoring."""

    def test_motion_blur_detected(self):
        """Test that motion blur is detected by FFT-based sharpness.

        Laplacian variance fails to detect motion blur because edges still
        exist (just smeared). FFT-based detection catches it by measuring
        high-frequency content loss.
        """
        import cv2
        from pathlib import Path
        from src.recognition.unidentified_face_manager import UnidentifiedFaceManager
        from src.config import UnidentifiedFacesConfig

        # Load motion blurred test image
        test_image_path = Path("tests/test-images/sensitive/20260130_205338_088f1a6c.jpg")
        if not test_image_path.exists():
            pytest.skip("Motion blur test image not available")

        img = cv2.imread(str(test_image_path))
        assert img is not None, "Failed to load test image"

        # Create manager to access _compute_sharpness_fft
        config = UnidentifiedFacesConfig(enabled=True)
        mock_repo = MagicMock()
        manager = UnidentifiedFaceManager(config, mock_repo)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness_score = manager._compute_sharpness_fft(gray)

        # Motion blurred image should have LOW sharpness score (<0.5)
        assert sharpness_score < 0.5, (
            f"Motion blur not detected! Sharpness score {sharpness_score:.2f} "
            f"should be below 0.5 for motion-blurred images"
        )

    def test_sharp_image_high_score(self):
        """Test that sharp images get high sharpness score."""
        import cv2
        from src.recognition.unidentified_face_manager import UnidentifiedFaceManager
        from src.config import UnidentifiedFacesConfig

        # Create a sharp image with natural-looking noise and edges
        # Random texture with varying intensities (like a natural image)
        np.random.seed(42)
        img = np.random.randint(50, 200, (200, 200), dtype=np.uint8)

        # Add some sharp edges
        img[50:150, 90:110] = 30  # Dark vertical stripe
        img[80:120, 40:160] = 220  # Bright horizontal stripe

        # Add fine detail/texture (high frequency)
        for i in range(0, 200, 4):
            for j in range(0, 200, 4):
                if (i + j) % 8 == 0:
                    img[i:i+2, j:j+2] = np.clip(img[i:i+2, j:j+2] + 50, 0, 255)

        config = UnidentifiedFacesConfig(enabled=True)
        mock_repo = MagicMock()
        manager = UnidentifiedFaceManager(config, mock_repo)

        sharpness_score = manager._compute_sharpness_fft(img)

        # Sharp natural-looking image should have good sharpness score (>0.5)
        assert sharpness_score > 0.5, (
            f"Sharp image not detected! Sharpness score {sharpness_score:.2f} "
            f"should be above 0.5 for sharp images"
        )

    def test_gaussian_blur_detected(self):
        """Test that Gaussian blur is detected."""
        import cv2
        from src.recognition.unidentified_face_manager import UnidentifiedFaceManager
        from src.config import UnidentifiedFacesConfig

        # Create sharp image and blur it
        img = np.zeros((200, 200), dtype=np.uint8)
        for i in range(0, 200, 20):
            for j in range(0, 200, 20):
                if (i // 20 + j // 20) % 2 == 0:
                    img[i:i+20, j:j+20] = 255

        blurred = cv2.GaussianBlur(img, (21, 21), 0)

        config = UnidentifiedFacesConfig(enabled=True)
        mock_repo = MagicMock()
        manager = UnidentifiedFaceManager(config, mock_repo)

        sharpness_score = manager._compute_sharpness_fft(blurred)

        # Blurred image should have LOW sharpness score (<0.3)
        assert sharpness_score < 0.3, (
            f"Gaussian blur not detected! Sharpness score {sharpness_score:.2f} "
            f"should be below 0.3 for blurred images"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
