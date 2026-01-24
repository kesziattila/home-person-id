"""Tests for multi-face detection filter in Re-ID gallery.

These tests verify that crops with multiple faces are filtered out
before storing in the Re-ID gallery.
"""

import cv2
import numpy as np
import pytest
from pathlib import Path

from src.config import FaceRecognitionConfig
from src.recognition.face_recognizer import FaceRecognizer


class TestMultiFaceDetection:
    """Test multi-face detection for Re-ID filtering."""

    @pytest.fixture
    def face_recognizer(self):
        """Create face recognizer with default config."""
        config = FaceRecognitionConfig(
            enabled=True,
            min_face_size=80,
        )
        return FaceRecognizer(config)

    def test_multi_face_detected_in_problematic_crop(self, face_recognizer):
        """Test that the problematic snapshot image has 2 detectable faces.

        This test uses a snapshot image that was incorrectly stored in the gallery
        despite containing 2 people. It documents that the default min_face_size=80
        is too large to detect the small faces in the crop.

        Bug: data/snapshots/reid_matches/20260124_083148_Judit_reid_0.91.jpg
        shows 2 people in the gallery crop but wasn't filtered because faces
        were too small (23x32, 25x33 pixels) for min_face_size=80.
        """
        # Load the problematic snapshot image
        snapshot_path = Path("data/snapshots/reid_matches/20260124_083148_Judit_reid_0.91.jpg")

        if not snapshot_path.exists():
            pytest.skip(f"Test image not found: {snapshot_path}")

        snapshot = cv2.imread(str(snapshot_path))
        assert snapshot is not None, f"Failed to load image: {snapshot_path}"

        gallery_crop, _ = extract_crops_from_snapshot(snapshot)

        # With default min_face_size=80, no faces should be detected
        # because the faces in this crop are only ~25x33 pixels
        result = face_recognizer.detect_faces(gallery_crop)

        print(f"\nGallery crop size: {gallery_crop.shape[:2]}")
        print(f"Faces detected with min_face_size=80: {result.count}")

        # This documents the root cause: default threshold is too high
        assert result.count == 0, (
            f"Expected 0 faces with min_face_size=80 (faces are too small), "
            f"but detected {result.count}"
        )

        # With lower min_face_size, we should detect 2 faces
        face_recognizer.config.min_face_size = 20
        result_low = face_recognizer.detect_faces(gallery_crop)

        print(f"Faces detected with min_face_size=20: {result_low.count}")
        for i, face in enumerate(result_low.faces):
            print(f"  Face {i}: size={face.width:.0f}x{face.height:.0f}")

        assert result_low.count >= 2, (
            f"Expected at least 2 faces with min_face_size=20, "
            f"but detected {result_low.count}"
        )

    def test_multi_face_detection_with_lower_min_size(self, face_recognizer):
        """Test if lowering min_face_size allows detecting both faces."""
        snapshot_path = Path("data/snapshots/reid_matches/20260124_083148_Judit_reid_0.91.jpg")

        if not snapshot_path.exists():
            pytest.skip(f"Test image not found: {snapshot_path}")

        snapshot = cv2.imread(str(snapshot_path))
        gallery_crop, _ = extract_crops_from_snapshot(snapshot)

        # Test with various min_face_size values
        test_sizes = [80, 60, 40, 20]

        print("\nTesting different min_face_size values:")
        for min_size in test_sizes:
            config = FaceRecognitionConfig(enabled=True, min_face_size=min_size)
            recognizer = FaceRecognizer(config)
            result = recognizer.detect_faces(gallery_crop)

            print(f"  min_face_size={min_size}: detected {result.count} faces")
            for face in result.faces:
                print(f"    bbox={face.bbox}, size={face.width:.0f}x{face.height:.0f}")


class TestReIDGalleryMultiFaceFilter:
    """Test the multi-face filter in ReIDGalleryManager."""

    @pytest.fixture
    def face_recognizer(self):
        """Create face recognizer for testing."""
        config = FaceRecognitionConfig(enabled=True, min_face_size=80)
        return FaceRecognizer(config)

    def test_has_multiple_faces_method(self, face_recognizer):
        """Test _has_multiple_faces method with real image.

        This test verifies that the multi-face filter correctly detects
        images with 2+ people, even when faces are small (< 80px).
        """
        from src.recognition.reid_gallery import ReIDGalleryManager
        from src.recognition.reid_extractor import ReIDExtractor
        from src.config import ReIDConfig

        reid_config = ReIDConfig(enabled=True)
        reid_extractor = ReIDExtractor(reid_config)

        manager = ReIDGalleryManager(
            reid_extractor=reid_extractor,
            face_recognizer=face_recognizer,
        )

        snapshot_path = Path("data/snapshots/reid_matches/20260124_083148_Judit_reid_0.91.jpg")

        if not snapshot_path.exists():
            pytest.skip(f"Test image not found: {snapshot_path}")

        snapshot = cv2.imread(str(snapshot_path))
        gallery_crop, _ = extract_crops_from_snapshot(snapshot)

        has_multiple = manager._has_multiple_faces(gallery_crop)

        print(f"\n_has_multiple_faces returned: {has_multiple}")

        # The filter should detect multiple faces even though they're small
        assert has_multiple, (
            f"Expected _has_multiple_faces to return True for image with 2 people. "
            f"The filter should use a lower min_face_size internally."
        )

    def test_has_multiple_faces_uses_config_threshold(self, face_recognizer):
        """Test that _has_multiple_faces uses multi_face_min_size from config."""
        from src.recognition.reid_gallery import ReIDGalleryManager
        from src.recognition.reid_extractor import ReIDExtractor
        from src.config import ReIDConfig

        reid_config = ReIDConfig(enabled=True)
        reid_extractor = ReIDExtractor(reid_config)

        manager = ReIDGalleryManager(
            reid_extractor=reid_extractor,
            face_recognizer=face_recognizer,
        )

        # Verify config has the multi_face_min_size parameter
        assert hasattr(face_recognizer.config, 'multi_face_min_size'), (
            "FaceRecognitionConfig should have multi_face_min_size attribute"
        )
        assert face_recognizer.config.multi_face_min_size == 20, (
            f"Default multi_face_min_size should be 20, "
            f"got {face_recognizer.config.multi_face_min_size}"
        )

        # Verify min_face_size (for recognition) is separate and unchanged
        assert face_recognizer.config.min_face_size == 80, (
            f"min_face_size should remain 80, got {face_recognizer.config.min_face_size}"
        )


def extract_crops_from_snapshot(snapshot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract gallery and current crops from a Re-ID match snapshot.

    The snapshot format from DebugImageSaver.save_reid_match() is:
    [gallery_crop | 5px white separator | match_crop]

    Returns:
        Tuple of (gallery_crop, match_crop)
    """
    h, w = snapshot.shape[:2]

    # Find white separator columns (5 pixels wide, RGB > 240)
    # Check middle row for separator
    mid_row = snapshot[h // 2, :, :]

    separator_start = None
    separator_end = None

    for x in range(w):
        is_white = np.all(mid_row[x] > 240)
        if is_white and separator_start is None:
            separator_start = x
        elif not is_white and separator_start is not None and separator_end is None:
            separator_end = x
            break

    if separator_start is None:
        # Fallback: split in half
        separator_start = w // 2
        separator_end = w // 2

    gallery_crop = snapshot[:, :separator_start]
    match_crop = snapshot[:, separator_end:]

    return gallery_crop, match_crop


if __name__ == "__main__":
    # Quick test without pytest
    print("=" * 60)
    print("Multi-face detection test")
    print("=" * 60)

    snapshot_path = Path("data/snapshots/reid_matches/20260124_083148_Judit_reid_0.91.jpg")
    if not snapshot_path.exists():
        print(f"Test image not found: {snapshot_path}")
        exit(1)

    # Load snapshot
    snapshot = cv2.imread(str(snapshot_path))
    print(f"Snapshot shape: {snapshot.shape}")

    # Extract gallery crop
    gallery_crop, match_crop = extract_crops_from_snapshot(snapshot)
    print(f"Gallery crop shape: {gallery_crop.shape}")
    print(f"Match crop shape: {match_crop.shape}")

    # Save extracted crops for inspection
    Path("data/debug").mkdir(parents=True, exist_ok=True)
    cv2.imwrite("data/debug/test_gallery_crop.jpg", gallery_crop)
    cv2.imwrite("data/debug/test_match_crop.jpg", match_crop)
    print("Saved crops to data/debug/")

    # Run face detection with default config
    config = FaceRecognitionConfig(enabled=True, min_face_size=80)
    recognizer = FaceRecognizer(config)

    print("\n--- Face detection on gallery crop (min_face_size=80) ---")
    result = recognizer.detect_faces(gallery_crop)
    print(f"Faces detected: {result.count}")
    for i, face in enumerate(result.faces):
        print(f"  Face {i}: bbox={[int(b) for b in face.bbox]}, "
              f"confidence={face.confidence:.2f}, "
              f"size={face.width:.0f}x{face.height:.0f}")

    # Test with different min_face_size values
    print("\n--- Testing different min_face_size values ---")
    for min_size in [80, 60, 40, 20, 10]:
        config = FaceRecognitionConfig(enabled=True, min_face_size=min_size)
        recognizer = FaceRecognizer(config)
        result = recognizer.detect_faces(gallery_crop)
        sizes = [f"{f.width:.0f}x{f.height:.0f}" for f in result.faces]
        print(f"  min_face_size={min_size:2d}: {result.count} faces detected ({', '.join(sizes) if sizes else 'none'})")

    # Draw detected faces on gallery crop
    gallery_with_faces = gallery_crop.copy()
    result = recognizer.detect_faces(gallery_crop)
    for face in result.faces:
        x1, y1, x2, y2 = [int(b) for b in face.bbox]
        cv2.rectangle(gallery_with_faces, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(gallery_with_faces, f"{face.confidence:.2f}",
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite("data/debug/test_gallery_with_faces.jpg", gallery_with_faces)
    print("\nSaved gallery with detected faces to data/debug/test_gallery_with_faces.jpg")

    # Conclusion
    print("\n" + "=" * 60)
    if result.count >= 2:
        print("RESULT: Face detector CAN detect 2 faces.")
        print("=> The filter should have caught this. Check if face_recognizer was None.")
    else:
        print(f"RESULT: Face detector only finds {result.count} face(s).")
        print("=> Need to lower min_face_size or accept this limitation.")
