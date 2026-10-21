"""Test unidentified face capture flow."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from src.recognition.identification_manager import IdentificationManager, FaceGalleryEntry
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
        mock_face.landmarks = np.array([[22, 25], [38, 25], [30, 33], [24, 42], [36, 42]], dtype=np.float32)

        # Mock face detection result
        mock_result = MagicMock()
        mock_result.faces = [mock_face]

        recognizer.detect_faces.return_value = mock_result
        recognizer.compare_embeddings.return_value = 0.3  # Below threshold
        # Mock batch comparison to return array of similarities
        recognizer.compare_embeddings_batch.return_value = np.array([0.3])

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
                FaceGalleryEntry(person_id=1, person_name="John", embedding=np.random.randn(512).astype(np.float32))
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
            frame=person_crop,
            bbox=(0, 0, 100, 200),
            local_track_id=1,
            isolated=True,
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
        assert call_args.kwargs["aligned_face"].shape == (112, 112, 3)   # what the quality model scores
        assert call_args.kwargs["face_bbox"] == (10, 10, 50, 50)          # the face, not the margin crop
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
            frame=person_crop,
            bbox=(0, 0, 100, 200),
            local_track_id=1,
            isolated=True,
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
                FaceGalleryEntry(person_id=1, person_name="John", embedding=np.random.randn(512).astype(np.float32))
            ],
            unidentified_face_manager=mock_unidentified_manager,
        )

        manager._face_recognizer = mock_face_recognizer
        person_crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Call WITHOUT camera_id
        result = manager.try_face_recognition(
            track_id="test_track",
            frame=person_crop,
            bbox=(0, 0, 100, 200),
            local_track_id=1,
            isolated=True,
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
        mock_face_recognizer.compare_embeddings_batch.return_value = np.array([0.8])

        manager = IdentificationManager(
            config=config,
            repository=None,
            face_gallery=[
                FaceGalleryEntry(person_id=1, person_name="John", embedding=np.random.randn(512).astype(np.float32))
            ],
            unidentified_face_manager=mock_unidentified_manager,
        )

        manager._face_recognizer = mock_face_recognizer
        person_crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        attempt = lambda: manager.try_face_recognition(
            track_id="test_track",
            frame=person_crop,
            bbox=(0, 0, 100, 200),
            local_track_id=1,
            isolated=True,
            camera_id="test_camera",
        )

        # Confirmed on the FACE_CONSECUTIVE_MATCHES-th accepted attempt (same rule as production)
        assert attempt() is False
        assert attempt() is True

        # Unidentified face manager should NOT have been called
        assert not mock_unidentified_manager.submit.called, \
            "UnidentifiedFaceManager.submit() should NOT be called when face is identified!"

        # Check identity was set
        identity = manager.get_identity("test_track")
        assert identity.person_name == "John"
        assert identity.is_face_identified is True


    @pytest.mark.parametrize("scores, first_confirms", [
        ([0.62, 0.40], True),    # strong: top1 >= 0.58 and margin >= 0.15 -> named at once
        ([0.62, 0.50], False),   # top1 strong but close runner-up -> streak needed
        ([0.56, 0.30], False),   # accepted, below strong_threshold -> streak needed
    ])
    def test_strong_attempt_confirms_without_streak(
        self, config, mock_face_recognizer, mock_unidentified_manager, scores, first_confirms
    ):
        """One strong attempt names the track; weaker accepted ones still need two in a row.

        Arrival 2026-10-19: hallway tracks had a single accepted face each (0.59, 0.56) and
        stayed unnamed; offline on 10-17..19 logs top1 >= 0.58 / margin >= 0.15 alone was
        right 1343/1343 times (0.56: 4 wrong).
        """
        config.face_recognition.similarity_threshold = 0.54
        config.face_recognition.min_margin = 0.10
        config.face_recognition.strong_threshold = 0.58
        config.face_recognition.strong_min_margin = 0.15
        mock_face_recognizer.compare_embeddings_batch.return_value = np.array(scores)
        manager = IdentificationManager(
            config=config,
            repository=None,
            face_gallery=[
                FaceGalleryEntry(person_id=1, person_name="John", embedding=np.random.randn(512).astype(np.float32)),
                FaceGalleryEntry(person_id=2, person_name="Jane", embedding=np.random.randn(512).astype(np.float32)),
            ],
            unidentified_face_manager=mock_unidentified_manager,
        )
        manager._face_recognizer = mock_face_recognizer
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        attempt = manager.attempt_face(frame, (0, 0, 100, 200), "t", camera_id="cam")
        assert attempt.accepted
        assert attempt.strong is first_confirms
        assert attempt.confirmed is first_confirms
        assert manager.attempt_face(frame, (0, 0, 100, 200), "t", camera_id="cam").confirmed  # streak of two

    def test_strong_attempt_not_allowed_after_appearance_change(
        self, config, mock_face_recognizer, mock_unidentified_manager
    ):
        """allow_strong=False (box just switched person): a strong face needs the streak."""
        config.face_recognition.strong_threshold = 0.58
        mock_face_recognizer.compare_embeddings_batch.return_value = np.array([0.62, 0.30])
        manager = IdentificationManager(
            config=config, repository=None, unidentified_face_manager=mock_unidentified_manager,
            face_gallery=[
                FaceGalleryEntry(person_id=1, person_name="John", embedding=np.random.randn(512).astype(np.float32)),
                FaceGalleryEntry(person_id=2, person_name="Jane", embedding=np.random.randn(512).astype(np.float32)),
            ],
        )
        manager._face_recognizer = mock_face_recognizer
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        attempt = manager.attempt_face(frame, (0, 0, 100, 200), "t", camera_id="cam", allow_strong=False)
        assert attempt.accepted and not attempt.strong and not attempt.confirmed
        assert manager.attempt_face(frame, (0, 0, 100, 200), "t", camera_id="cam", allow_strong=False).confirmed

    def test_strong_attempt_off_by_default(self, config, mock_face_recognizer, mock_unidentified_manager):
        mock_face_recognizer.compare_embeddings_batch.return_value = np.array([0.95])
        manager = IdentificationManager(
            config=config, repository=None, unidentified_face_manager=mock_unidentified_manager,
            face_gallery=[FaceGalleryEntry(person_id=1, person_name="John",
                                           embedding=np.random.randn(512).astype(np.float32))],
        )
        manager._face_recognizer = mock_face_recognizer
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        attempt = manager.attempt_face(frame, (0, 0, 100, 200), "t", camera_id="cam")
        assert attempt.accepted and not attempt.strong and not attempt.confirmed


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
