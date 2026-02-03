#!/usr/bin/env python3
"""Compare face embeddings between InsightFace and TensorRT implementations.

This script verifies that the TensorRT face embedder produces embeddings
that are similar to InsightFace's embeddings.

Usage:
    python tools/compare_embeddings.py
    python tools/compare_embeddings.py --image path/to/image.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.recognition.tensorrt_face import (
    TensorRTFaceDetector,
    TensorRTFaceEmbedder,
    align_face,
)

TEST_IMAGE = "tests/test-images/sample-person-image.jpg"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    parser = argparse.ArgumentParser(description="Compare InsightFace vs TensorRT embeddings")
    parser.add_argument("--image", type=str, default=TEST_IMAGE, help="Test image")
    parser.add_argument("--det-model", type=str, default="models/det_10g.engine",
                        help="TensorRT detection model")
    parser.add_argument("--rec-model", type=str, default="models/w600k_r50.engine",
                        help="TensorRT recognition model")
    parser.add_argument("--det-size", type=int, default=640,
                        help="Detection input size (must match model)")
    args = parser.parse_args()

    # Check files
    if not Path(args.image).exists():
        print(f"ERROR: Image not found: {args.image}")
        return 1

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"ERROR: Could not load image: {args.image}")
        return 1

    print(f"Image: {args.image}, shape: {frame.shape}")

    # === InsightFace Baseline ===
    print("\n=== InsightFace Baseline ===")
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    insightface_faces = app.get(frame)
    print(f"InsightFace detected {len(insightface_faces)} face(s)")

    if not insightface_faces:
        print("No faces detected by InsightFace")
        return 1

    # Get first face
    if_face = insightface_faces[0]
    if_bbox = if_face.bbox.tolist()
    if_landmarks = if_face.kps
    if_embedding = if_face.embedding
    if_embedding_norm = np.linalg.norm(if_embedding)

    print(f"  Bbox: {[round(x, 1) for x in if_bbox]}")
    print(f"  Landmarks shape: {if_landmarks.shape}")
    print(f"  Embedding shape: {if_embedding.shape}")
    print(f"  Embedding norm: {if_embedding_norm:.4f}")
    print(f"  Embedding sample: {if_embedding[:5]}")

    # === TensorRT Implementation ===
    print("\n=== TensorRT Implementation ===")

    if not Path(args.det_model).exists():
        print(f"ERROR: Detection model not found: {args.det_model}")
        return 1
    if not Path(args.rec_model).exists():
        print(f"ERROR: Recognition model not found: {args.rec_model}")
        return 1

    detector = TensorRTFaceDetector(args.det_model, input_size=args.det_size)
    embedder = TensorRTFaceEmbedder(args.rec_model)

    trt_faces = detector.detect(frame)
    print(f"TensorRT detected {len(trt_faces)} face(s)")

    if not trt_faces:
        print("No faces detected by TensorRT")
        return 1

    # Get first face
    trt_face = trt_faces[0]
    trt_bbox = list(trt_face.bbox)
    trt_landmarks = trt_face.landmarks

    print(f"  Bbox: {[round(x, 1) for x in trt_bbox]}")
    print(f"  Landmarks shape: {trt_landmarks.shape}")

    # Align face and extract embedding
    aligned = align_face(frame, trt_landmarks)
    trt_embedding = embedder.extract(aligned)
    trt_embedding_norm = np.linalg.norm(trt_embedding)

    print(f"  Embedding shape: {trt_embedding.shape}")
    print(f"  Embedding norm: {trt_embedding_norm:.4f}")
    print(f"  Embedding sample: {trt_embedding[:5]}")

    # === Comparison ===
    print("\n=== Comparison ===")

    # Bbox comparison
    bbox_diff = np.array(trt_bbox) - np.array(if_bbox)
    print(f"Bbox difference (TRT - IF): {[round(x, 1) for x in bbox_diff]}")
    print(f"Bbox max diff: {np.abs(bbox_diff).max():.1f} pixels")

    # Landmarks comparison
    lm_diff = trt_landmarks - if_landmarks
    print(f"Landmarks max diff: {np.abs(lm_diff).max():.1f} pixels")

    # Embedding comparison
    cos_sim = cosine_similarity(trt_embedding, if_embedding)
    l2_dist = np.linalg.norm(trt_embedding - if_embedding)

    print(f"\nEmbedding cosine similarity: {cos_sim:.6f}")
    print(f"Embedding L2 distance: {l2_dist:.6f}")

    # Interpretation
    print("\n=== Interpretation ===")
    if cos_sim >= 0.99:
        print("EXCELLENT: Embeddings are nearly identical (cos_sim >= 0.99)")
    elif cos_sim >= 0.95:
        print("GOOD: Embeddings are very similar (cos_sim >= 0.95)")
    elif cos_sim >= 0.90:
        print("ACCEPTABLE: Embeddings are similar enough (cos_sim >= 0.90)")
    elif cos_sim >= 0.80:
        print("WARNING: Embeddings differ somewhat (cos_sim >= 0.80)")
    else:
        print("ERROR: Embeddings are too different (cos_sim < 0.80)")

    # Save aligned face for visual inspection
    output_dir = Path("data/debug")
    output_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_dir / "aligned_face_trt.jpg"), aligned)
    print(f"\nSaved aligned face to: {output_dir / 'aligned_face_trt.jpg'}")

    # Also save the InsightFace aligned face for comparison
    # InsightFace uses the same alignment internally
    if_aligned = align_face(frame, if_landmarks)
    cv2.imwrite(str(output_dir / "aligned_face_insightface.jpg"), if_aligned)
    print(f"Saved InsightFace aligned face to: {output_dir / 'aligned_face_insightface.jpg'}")

    return 0 if cos_sim >= 0.90 else 1


if __name__ == "__main__":
    sys.exit(main())
