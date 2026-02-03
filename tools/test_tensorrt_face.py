#!/usr/bin/env python3
"""Test script for TensorRT face detection and recognition.

Usage:
    python tools/test_tensorrt_face.py --det-model models/det_10g.engine --rec-model models/w600k_r50.engine
    python tools/test_tensorrt_face.py --show  # Display result

First, convert ONNX models to TensorRT:
    python tools/convert_insightface_to_trt.py
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.recognition.tensorrt_face import (
    TensorRTFaceDetector,
    TensorRTFaceEmbedder,
    TensorRTFaceRecognizer,
    align_face,
)

TEST_IMAGE = "tests/test-images/sample-person-image.jpg"


def test_detector(detector, image_path: str, num_runs: int = 10, show: bool = False):
    """Test face detector."""
    print(f"\n=== Testing Face Detector ===")

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"ERROR: Could not load image: {image_path}")
        return None

    print(f"Image shape: {frame.shape}")

    # Warmup
    print("Warming up...")
    detector.warmup()

    # Timed runs
    print(f"Running {num_runs} iterations...")
    times = []
    faces = None
    for i in range(num_runs):
        start = time.perf_counter()
        faces = detector.detect(frame)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg_time = np.mean(times)
    print(f"Average: {avg_time:.1f}ms, Detected: {len(faces)} faces")

    for i, face in enumerate(faces):
        print(f"  Face {i}: bbox={face.bbox}, conf={face.confidence:.3f}")

    # Always save visualization for debugging
    vis = frame.copy()
    for i, face in enumerate(faces):
        x1, y1, x2, y2 = map(int, face.bbox)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{i}:{face.confidence:.2f}"
        cv2.putText(vis, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        # Draw landmarks
        for lm in face.landmarks:
            cv2.circle(vis, (int(lm[0]), int(lm[1])), 2, (0, 0, 255), -1)

    output_path = "data/debug/face_detection_result.jpg"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, vis)
    print(f"\nSaved visualization to: {output_path}")

    return faces, frame


def test_embedder(embedder, frame: np.ndarray, faces: list, num_runs: int = 10):
    """Test face embedder."""
    print(f"\n=== Testing Face Embedder ===")

    if not faces:
        print("No faces to test")
        return

    # Get first face
    face = faces[0]
    aligned = align_face(frame, face.landmarks)

    print(f"Aligned face shape: {aligned.shape}")

    # Warmup
    print("Warming up...")
    embedder.warmup()

    # Timed runs
    print(f"Running {num_runs} iterations...")
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        embedding = embedder.extract(aligned)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg_time = np.mean(times)
    print(f"Average: {avg_time:.1f}ms")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
    print(f"Embedding sample: {embedding[:5]}")


def test_recognizer(recognizer, image_path: str, num_runs: int = 10, show: bool = False):
    """Test combined recognizer."""
    print(f"\n=== Testing Combined Recognizer ===")
    print(f"Min face size filter: {recognizer.min_face_size}px")

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"ERROR: Could not load image: {image_path}")
        return

    print(f"Image shape: {frame.shape}")

    # Warmup
    print("Warming up...")
    recognizer.warmup()

    # Timed runs
    print(f"Running {num_runs} iterations...")
    times = []
    faces = None
    for i in range(num_runs):
        start = time.perf_counter()
        faces = recognizer.detect_and_embed(frame)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg_time = np.mean(times)
    print(f"Average: {avg_time:.1f}ms (detection + embedding)")
    print(f"Detected: {len(faces)} faces with embeddings")

    for i, face in enumerate(faces):
        print(f"  Face {i}: conf={face.confidence:.3f}, embedding_norm={np.linalg.norm(face.embedding):.4f}")

    if show and faces:
        vis = frame.copy()
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = map(int, face.bbox)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{i}:{face.confidence:.2f}"
            cv2.putText(vis, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            # Draw landmarks if available
            if face.landmarks is not None:
                for lm in face.landmarks:
                    cv2.circle(vis, (int(lm[0]), int(lm[1])), 2, (0, 0, 255), -1)

        output_path = "data/debug/face_recognition_result.jpg"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, vis)
        print(f"\nSaved visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test TensorRT face recognition")
    parser.add_argument("--det-model", type=str, default="models/det_10g.engine",
                        help="Path to detection engine")
    parser.add_argument("--rec-model", type=str, default="models/w600k_r50.engine",
                        help="Path to recognition engine")
    parser.add_argument("--image", type=str, default=TEST_IMAGE, help="Test image")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs")
    parser.add_argument("--show", action="store_true", help="Show results")
    parser.add_argument("--det-only", action="store_true", help="Test detection only")
    parser.add_argument("--emb-only", action="store_true", help="Test embedding only")
    parser.add_argument("--min-face-size", type=int, default=40, help="Min face size in pixels (default 40)")
    parser.add_argument("--det-size", type=int, default=640, help="Detection input size (must match model)")
    args = parser.parse_args()

    # Check files
    if not Path(args.image).exists():
        print(f"ERROR: Image not found: {args.image}")
        return 1

    if args.det_only:
        if not Path(args.det_model).exists():
            print(f"ERROR: Detection model not found: {args.det_model}")
            print("\nConvert with: trtexec --onnx=det_10g.onnx --saveEngine=det_10g.engine --fp16")
            return 1
        detector = TensorRTFaceDetector(args.det_model, input_size=args.det_size)
        test_detector(detector, args.image, args.runs, args.show)
        return 0

    if args.emb_only:
        if not Path(args.rec_model).exists():
            print(f"ERROR: Recognition model not found: {args.rec_model}")
            print("\nConvert with: trtexec --onnx=w600k_r50.onnx --saveEngine=w600k_r50.engine --fp16")
            return 1
        # Need detector first to get faces
        if not Path(args.det_model).exists():
            print(f"ERROR: Need detection model for faces: {args.det_model}")
            return 1
        detector = TensorRTFaceDetector(args.det_model, input_size=args.det_size)
        embedder = TensorRTFaceEmbedder(args.rec_model)
        faces, frame = test_detector(detector, args.image, 3, False)
        if faces:
            test_embedder(embedder, frame, faces, args.runs)
        return 0

    # Full test
    if not Path(args.det_model).exists() or not Path(args.rec_model).exists():
        print(f"ERROR: Models not found")
        print(f"  Detection: {args.det_model} - {'OK' if Path(args.det_model).exists() else 'MISSING'}")
        print(f"  Recognition: {args.rec_model} - {'OK' if Path(args.rec_model).exists() else 'MISSING'}")
        print("\nConvert InsightFace models to TensorRT:")
        print("  python tools/convert_insightface_to_trt.py")
        return 1

    recognizer = TensorRTFaceRecognizer(
        det_model_path=args.det_model,
        rec_model_path=args.rec_model,
        det_size=args.det_size,
        min_face_size=args.min_face_size,
    )
    test_recognizer(recognizer, args.image, args.runs, args.show)

    return 0


if __name__ == "__main__":
    sys.exit(main())
