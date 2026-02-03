#!/usr/bin/env python3
"""Test InsightFace baseline for comparison with TensorRT implementation.

Usage:
    python tools/test_insightface_baseline.py
    python tools/test_insightface_baseline.py --image path/to/image.jpg
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

TEST_IMAGE = "tests/test-images/sample-person-image.jpg"


def main():
    parser = argparse.ArgumentParser(description="Test InsightFace baseline")
    parser.add_argument("--image", type=str, default=TEST_IMAGE, help="Test image")
    parser.add_argument("--det-size", type=int, default=640, help="Detection size")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs")
    args = parser.parse_args()

    from insightface.app import FaceAnalysis

    print(f"Loading InsightFace...")
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],
    )
    app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"ERROR: Could not load image: {args.image}")
        return 1

    print(f"Image shape: {frame.shape}")

    # Warmup
    print("Warming up...")
    app.get(frame)

    # Timed runs
    print(f"Running {args.runs} iterations...")
    times = []
    for i in range(args.runs):
        start = time.perf_counter()
        faces = app.get(frame)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg_time = np.mean(times)
    print(f"\nAverage: {avg_time:.1f}ms")
    print(f"Detected: {len(faces)} faces")

    for i, f in enumerate(faces):
        bbox = f.bbox.tolist()
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        print(f"  Face {i}: bbox={[round(x, 1) for x in bbox]}, "
              f"size={width:.0f}x{height:.0f}, score={f.det_score:.3f}")
        if hasattr(f, 'embedding') and f.embedding is not None:
            print(f"           embedding_norm={np.linalg.norm(f.embedding):.4f}")

    # Save visualization
    vis = frame.copy()
    for i, f in enumerate(faces):
        x1, y1, x2, y2 = map(int, f.bbox)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f"{i}:{f.det_score:.2f}"
        cv2.putText(vis, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        if hasattr(f, 'kps') and f.kps is not None:
            for lm in f.kps:
                cv2.circle(vis, (int(lm[0]), int(lm[1])), 2, (0, 255, 0), -1)

    output_path = "data/debug/insightface_baseline_result.jpg"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, vis)
    print(f"\nSaved visualization to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
