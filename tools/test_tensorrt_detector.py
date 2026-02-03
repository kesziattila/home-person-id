#!/usr/bin/env python3
"""Test script for TensorRT YOLO detector.

Run on Jetson to verify TensorRT model loading and inference.

Usage:
    python tools/test_tensorrt_detector.py
    python tools/test_tensorrt_detector.py --model models/yolo26s.engine
    python tools/test_tensorrt_detector.py --show  # Display result with OpenCV
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.detection.person_detector import TensorRTDetector, PersonDetector

TEST_IMAGE = "tests/test-images/sample-person-image.jpg"


def test_detector(detector, image_path: str, num_warmup: int = 3, num_runs: int = 10, show: bool = False):
    """Test detector with timing."""
    print(f"\nLoading test image: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"ERROR: Could not load image: {image_path}")
        return False

    print(f"Image shape: {frame.shape}")

    # Warmup
    print(f"\nWarmup ({num_warmup} runs)...")
    for i in range(num_warmup):
        result = detector.detect(frame)
        print(f"  Run {i+1}: {result.count} detections")

    # Timed runs
    print(f"\nTimed runs ({num_runs} runs)...")
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        result = detector.detect(frame)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.1f}ms, {result.count} detections")

    # Stats
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    std_time = np.std(times)

    print(f"\n--- Results ---")
    print(f"Average: {avg_time:.1f}ms")
    print(f"Min: {min_time:.1f}ms")
    print(f"Max: {max_time:.1f}ms")
    print(f"Std: {std_time:.1f}ms")
    print(f"FPS: {1000/avg_time:.1f}")

    # Show detections
    print(f"\nDetections:")
    for i, det in enumerate(result.detections):
        print(f"  [{i}] bbox={det.bbox}, conf={det.confidence:.3f}")

    if show and result.count > 0:
        # Draw detections
        for det in result.detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"person {det.confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("TensorRT Detection Test", frame)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return True


def main():
    parser = argparse.ArgumentParser(description="Test TensorRT YOLO detector")
    parser.add_argument("--model", type=str, help="Path to model file (overrides config)")
    parser.add_argument("--image", type=str, default=TEST_IMAGE, help="Path to test image")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=10, help="Number of timed runs")
    parser.add_argument("--show", action="store_true", help="Show detection result with OpenCV")
    parser.add_argument("--ultralytics", action="store_true", help="Test Ultralytics detector instead")
    args = parser.parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    try:
        config = load_config(args.config)
        model_path = args.model or config.detection.model
        confidence = config.detection.confidence_threshold
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        model_path = args.model or "models/yolo11s.engine"
        confidence = 0.5

    print(f"Model: {model_path}")
    print(f"Confidence threshold: {confidence}")

    # Check if model exists
    if not Path(model_path).exists():
        print(f"ERROR: Model file not found: {model_path}")
        print("\nTo export a TensorRT engine with NMS baked in:")
        print("  yolo export model=yolo11s.pt format=engine nms=True half=True")
        return 1

    # Check if test image exists
    if not Path(args.image).exists():
        print(f"ERROR: Test image not found: {args.image}")
        return 1

    # Create detector
    if args.ultralytics:
        print("\n=== Testing Ultralytics/PyTorch Detector ===")
        detector = PersonDetector(
            model_path=model_path,
            confidence_threshold=confidence,
        )
    else:
        print("\n=== Testing TensorRT Native Detector ===")

        if not model_path.endswith((".engine", ".trt")):
            print(f"ERROR: TensorRT detector requires .engine or .trt file, got: {model_path}")
            print("Use --ultralytics flag to test with Ultralytics backend")
            return 1

        detector = TensorRTDetector(
            model_path=model_path,
            confidence_threshold=confidence,
        )

    # Run test
    try:
        success = test_detector(
            detector,
            args.image,
            num_warmup=args.warmup,
            num_runs=args.runs,
            show=args.show,
        )
        return 0 if success else 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
