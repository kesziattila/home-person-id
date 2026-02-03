#!/usr/bin/env python3
"""Convert InsightFace ONNX models to TensorRT engines.

This script converts the detection (RetinaFace) and recognition (ArcFace)
models from InsightFace to TensorRT engines for native TensorRT inference.

Usage:
    python tools/convert_insightface_to_trt.py
    python tools/convert_insightface_to_trt.py --model buffalo_l
    python tools/convert_insightface_to_trt.py --output-dir models/
    python tools/convert_insightface_to_trt.py --det-size 640  # or 480, 320

Requirements:
    - TensorRT with trtexec command available
    - InsightFace models downloaded (run face recognition once first)
"""

import argparse
import subprocess
import sys
from pathlib import Path


def find_insightface_models(model_name: str = "buffalo_l") -> tuple[Path, Path]:
    """Find InsightFace ONNX models."""
    model_dir = Path.home() / ".insightface" / "models" / model_name

    if not model_dir.exists():
        raise FileNotFoundError(
            f"InsightFace model not found at {model_dir}\n"
            f"Run face recognition once to download models, or download manually."
        )

    det_model = model_dir / "det_10g.onnx"
    rec_model = model_dir / "w600k_r50.onnx"

    if not det_model.exists():
        raise FileNotFoundError(f"Detection model not found: {det_model}")
    if not rec_model.exists():
        raise FileNotFoundError(f"Recognition model not found: {rec_model}")

    return det_model, rec_model


def convert_to_tensorrt(
    onnx_path: Path,
    engine_path: Path,
    fp16: bool = True,
    input_shape: tuple = None,
    dynamic: bool = False,
) -> bool:
    """Convert ONNX model to TensorRT engine using trtexec."""
    cmd = ["trtexec", f"--onnx={onnx_path}", f"--saveEngine={engine_path}"]

    if fp16:
        cmd.append("--fp16")

    if input_shape:
        # Format: name:NxCxHxW
        shape_str = "x".join(map(str, input_shape))
        cmd.append(f"--shapes=input.1:{shape_str}")

    if dynamic:
        # Allow dynamic batch size
        cmd.append("--minShapes=input.1:1x3x112x112")
        cmd.append("--optShapes=input.1:1x3x112x112")
        cmd.append("--maxShapes=input.1:8x3x112x112")

    print(f"\nConverting: {onnx_path.name} -> {engine_path.name}")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR: Conversion failed")
            print(result.stderr)
            return False
        print(f"SUCCESS: {engine_path}")
        return True
    except FileNotFoundError:
        print("ERROR: trtexec not found. Make sure TensorRT is installed.")
        print("On Jetson, it should be at /usr/src/tensorrt/bin/trtexec")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert InsightFace to TensorRT")
    parser.add_argument("--model", type=str, default="buffalo_l",
                        help="InsightFace model name (buffalo_l, buffalo_s, buffalo_sc)")
    parser.add_argument("--output-dir", type=str, default="models",
                        help="Output directory for engines")
    parser.add_argument("--det-size", type=int, default=640,
                        help="Detection input size (640, 480, or 320)")
    parser.add_argument("--no-fp16", action="store_true",
                        help="Disable FP16 (use FP32)")
    args = parser.parse_args()

    # Find ONNX models
    try:
        det_onnx, rec_onnx = find_insightface_models(args.model)
        print(f"Found InsightFace models:")
        print(f"  Detection: {det_onnx}")
        print(f"  Recognition: {rec_onnx}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fp16 = not args.no_fp16
    det_size = args.det_size

    # Convert detection model
    det_engine = output_dir / "det_10g.engine"
    det_input_shape = (1, 3, det_size, det_size)

    success = convert_to_tensorrt(
        det_onnx,
        det_engine,
        fp16=fp16,
        input_shape=det_input_shape,
    )
    if not success:
        return 1

    # Convert recognition model
    rec_engine = output_dir / "w600k_r50.engine"
    rec_input_shape = (1, 3, 112, 112)

    success = convert_to_tensorrt(
        rec_onnx,
        rec_engine,
        fp16=fp16,
        input_shape=rec_input_shape,
    )
    if not success:
        return 1

    print(f"\n=== Conversion Complete ===")
    print(f"Detection engine: {det_engine}")
    print(f"Recognition engine: {rec_engine}")
    print(f"\nTo test:")
    print(f"  python tools/test_tensorrt_face.py --det-model {det_engine} --rec-model {rec_engine}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
