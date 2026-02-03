#!/usr/bin/env python3
"""Test TensorRT native Re-ID embedder.

Usage:
    python tools/test_tensorrt_reid.py
    python tools/test_tensorrt_reid.py --model models/osnet_ain_x1_0.engine
    python tools/test_tensorrt_reid.py --compare  # Compare with PyTorch
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

TEST_IMAGE = "tests/test-images/sample-person-image.jpg"


def test_embedder(model_path: str, image_path: str = None, num_runs: int = 20):
    """Test TensorRT Re-ID embedder."""
    from src.recognition.tensorrt_reid import TensorRTReIDEmbedder

    print(f"\n=== Testing TensorRT Re-ID Embedder ===")
    print(f"Model: {model_path}")

    if not Path(model_path).exists():
        print(f"ERROR: Model not found: {model_path}")
        print("\nConvert model with: python tools/convert_reid_to_trt.py")
        return 1

    # Load image or create dummy
    if image_path and Path(image_path).exists():
        frame = cv2.imread(image_path)
        crop = frame
        print(f"Image: {image_path}, shape: {crop.shape}")
    else:
        crop = np.zeros((256, 128, 3), dtype=np.uint8)
        print(f"Using dummy input: shape {crop.shape}")

    # Create embedder
    embedder = TensorRTReIDEmbedder(model_path)

    # Warmup
    print("Warming up...")
    embedder.warmup()

    # Timed runs
    print(f"Running {num_runs} iterations...")
    times = []
    embedding = None
    for i in range(num_runs):
        start = time.perf_counter()
        embedding, quality = embedder.extract(crop, return_quality=True)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg_time = np.mean(times)
    min_time = np.min(times)

    print(f"\n=== Results ===")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
    print(f"Quality score: {quality:.4f}")
    print(f"Average time: {avg_time:.2f}ms")
    print(f"Min time: {min_time:.2f}ms")
    print(f"Throughput: {1000/avg_time:.1f} embeddings/sec")
    print(f"Embedding sample: {embedding[:5]}")

    return 0


def compare_with_pytorch(
    trt_model: str,
    pth_model: str,
    image_path: str,
    arch: str = "osnet_ain_x1_0",
):
    """Compare TensorRT with PyTorch embeddings.

    IMPORTANT: PyTorch must run FIRST before TensorRT to avoid CUDA context conflicts.
    pycuda and PyTorch cannot share CUDA contexts in the same process.
    """
    print(f"\n=== Comparing TensorRT vs PyTorch ===")

    if not Path(trt_model).exists():
        print(f"ERROR: TensorRT model not found: {trt_model}")
        return 1
    if not Path(pth_model).exists():
        print(f"ERROR: PyTorch model not found: {pth_model}")
        return 1

    # Load image
    if Path(image_path).exists():
        crop = cv2.imread(image_path)
    else:
        crop = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)

    print(f"Image shape: {crop.shape}")

    # ========================================
    # PyTorch embedding FIRST (before TensorRT takes CUDA context)
    # ========================================
    print("\nPyTorch (reference):")
    torch_emb = None
    torch_times = []
    try:
        import torch
        import torchvision.transforms as T
        from torchreid import models

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {device}")

        # Build and load model
        model = models.build_model(
            name=arch,
            num_classes=1,
            loss="softmax",
            pretrained=False,
        )

        state_dict = torch.load(pth_model, map_location=device)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier")}
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()

        # Preprocessing
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0).to(device)

        # Warmup
        with torch.no_grad():
            _ = model(tensor)

        # Timed runs
        for _ in range(10):
            start = time.perf_counter()
            with torch.no_grad():
                features = model(tensor)
                if isinstance(features, tuple):
                    features = features[0]
                features = torch.nn.functional.normalize(features, p=2, dim=1)
                torch_emb = features.cpu().numpy().flatten()
            torch_times.append((time.perf_counter() - start) * 1000)

        print(f"  Shape: {torch_emb.shape}, Norm: {np.linalg.norm(torch_emb):.4f}")
        print(f"  Avg time: {np.mean(torch_times):.2f}ms")
        print(f"  Sample: {torch_emb[:5]}")

        # Clean up PyTorch CUDA context before TensorRT
        del model, tensor, features
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    except Exception as e:
        print(f"  Error loading PyTorch model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ========================================
    # TensorRT embedding (after PyTorch is done)
    # ========================================
    print("\nTensorRT:")
    from src.recognition.tensorrt_reid import TensorRTReIDEmbedder

    trt_embedder = TensorRTReIDEmbedder(trt_model)
    trt_embedder.warmup()

    trt_times = []
    trt_emb = None
    for _ in range(10):
        start = time.perf_counter()
        trt_emb = trt_embedder.extract(crop)
        trt_times.append((time.perf_counter() - start) * 1000)

    print(f"  Shape: {trt_emb.shape}, Norm: {np.linalg.norm(trt_emb):.4f}")
    print(f"  Avg time: {np.mean(trt_times):.2f}ms")
    print(f"  Sample: {trt_emb[:5]}")

    # ========================================
    # Compare embeddings
    # ========================================
    cos_sim = np.dot(trt_emb, torch_emb) / (np.linalg.norm(trt_emb) * np.linalg.norm(torch_emb) + 1e-8)
    l2_dist = np.linalg.norm(trt_emb - torch_emb)

    print(f"\n=== Comparison ===")
    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"L2 distance: {l2_dist:.6f}")
    print(f"TensorRT speedup: {np.mean(torch_times)/np.mean(trt_times):.2f}x")

    if cos_sim >= 0.99:
        print("\nEXCELLENT: Embeddings are nearly identical (cos_sim >= 0.99)")
    elif cos_sim >= 0.95:
        print("\nGOOD: Embeddings are very similar (cos_sim >= 0.95)")
    elif cos_sim >= 0.90:
        print("\nACCEPTABLE: Embeddings are similar enough (cos_sim >= 0.90)")
    else:
        print("\nWARNING: Embeddings differ significantly (cos_sim < 0.90)")
        return 1

    return 0


def main():
    parser = argparse.ArgumentParser(description="Test TensorRT Re-ID embedder")
    parser.add_argument("--model", type=str, default="models/osnet_ain_x1_0.engine",
                        help="TensorRT engine path")
    parser.add_argument("--image", type=str, default=TEST_IMAGE,
                        help="Test image path")
    parser.add_argument("--runs", type=int, default=20,
                        help="Number of timing runs")
    parser.add_argument("--compare", action="store_true",
                        help="Compare with PyTorch (reference)")
    parser.add_argument("--pth", type=str,
                        default="models/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth",
                        help="PyTorch model for comparison")
    parser.add_argument("--arch", type=str, default="osnet_ain_x1_0",
                        help="Model architecture")
    args = parser.parse_args()

    if args.compare:
        # Run comparison only (no separate test_embedder call to avoid CUDA conflict)
        return compare_with_pytorch(args.model, args.pth, args.image, args.arch)
    else:
        return test_embedder(args.model, args.image, args.runs)


if __name__ == "__main__":
    sys.exit(main())
