#!/usr/bin/env python3
"""Convert Re-ID model (PyTorch .pth) to TensorRT engine.

This script handles the full conversion pipeline:
1. Load PyTorch model (.pth) using torchreid -> ONNX (via convert_reid_to_onnx)
2. Convert ONNX to TensorRT engine

Usage:
    # Convert using existing ONNX
    python tools/convert_reid_to_trt.py

    # Convert from PyTorch model
    python tools/convert_reid_to_trt.py --pth models/osnet_ain_x1_0.pth

    # Specify ONNX directly (skip PyTorch step)
    python tools/convert_reid_to_trt.py --onnx models/osnet.onnx

Requirements:
    - PyTorch and torchreid (for .pth to ONNX conversion)
    - TensorRT (for ONNX to engine conversion)
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Import ONNX conversion from existing script
from tools.convert_reid_to_onnx import convert_to_onnx

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def detect_architecture(model_path: str) -> str:
    """Detect OSNet architecture from model filename."""
    model_path = model_path.lower()
    if "osnet_ain_x1_0" in model_path:
        return "osnet_ain_x1_0"
    elif "osnet_x1_0" in model_path:
        return "osnet_x1_0"
    elif "osnet_x0_75" in model_path:
        return "osnet_x0_75"
    elif "osnet_x0_5" in model_path:
        return "osnet_x0_5"
    elif "osnet_x0_25" in model_path:
        return "osnet_x0_25"
    else:
        return "osnet_x1_0"  # Default


def convert_with_trtexec(onnx_path: str, engine_path: str, fp16: bool = True, workspace: int = 1024) -> bool:
    """Convert ONNX to TensorRT using trtexec."""
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={workspace}",
    ]
    if fp16:
        cmd.append("--fp16")

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            logger.info(f"TensorRT engine saved: {engine_path}")
            return True
        else:
            logger.error(f"trtexec failed:\n{result.stderr}")
            return False
    except FileNotFoundError:
        logger.warning("trtexec not found, trying Python TensorRT API...")
        return False
    except subprocess.TimeoutExpired:
        logger.error("trtexec timed out")
        return False


def convert_with_tensorrt_api(onnx_path: str, engine_path: str, fp16: bool = True) -> bool:
    """Convert ONNX to TensorRT using Python API."""
    try:
        import tensorrt as trt
    except ImportError:
        logger.error("TensorRT Python bindings not found")
        logger.error("Install with: sudo apt install python3-libnvinfer")
        return False

    logger.info("Converting with TensorRT Python API...")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    logger.info(f"Parsing ONNX model: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                logger.error(f"ONNX parse error: {parser.get_error(error)}")
            return False

    # Build engine
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    if fp16 and builder.platform_has_fast_fp16:
        logger.info("Enabling FP16 mode")
        config.set_flag(trt.BuilderFlag.FP16)

    # Set optimization profile for dynamic batch
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    # Min, optimal, max batch sizes
    profile.set_shape(input_name, (1, 3, 256, 128), (1, 3, 256, 128), (8, 3, 256, 128))
    config.add_optimization_profile(profile)

    logger.info("Building TensorRT engine (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        logger.error("Failed to build TensorRT engine")
        return False

    # Save engine
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    logger.info(f"TensorRT engine saved: {engine_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert Re-ID model to TensorRT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert existing ONNX model (auto-detect)
    python tools/convert_reid_to_trt.py

    # Convert PyTorch model (auto-detect architecture)
    python tools/convert_reid_to_trt.py --pth models/osnet_ain_x1_0.pth

    # Convert with explicit architecture
    python tools/convert_reid_to_trt.py --pth models/custom.pth --arch osnet_x1_0

    # Convert existing ONNX model
    python tools/convert_reid_to_trt.py --onnx models/osnet.onnx
        """,
    )
    parser.add_argument(
        "--pth",
        type=str,
        default=None,
        help="Input PyTorch model (.pth file)",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default=None,
        help="Input ONNX model (skip PyTorch conversion)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        choices=["osnet_x1_0", "osnet_x0_75", "osnet_x0_5", "osnet_x0_25", "osnet_ain_x1_0"],
        help="OSNet architecture (auto-detected from filename if not specified)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output TensorRT engine path",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use FP32 precision instead of FP16",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=1024,
        help="Workspace size in MB for trtexec (default: 1024)",
    )
    args = parser.parse_args()

    # Determine input paths
    pth_path = None
    if args.onnx:
        onnx_path = Path(args.onnx)
        if not onnx_path.exists():
            logger.error(f"ONNX model not found: {onnx_path}")
            return 1
    elif args.pth:
        pth_path = Path(args.pth)
        if not pth_path.exists():
            logger.error(f"PyTorch model not found: {pth_path}")
            return 1
        onnx_path = pth_path.with_suffix(".onnx")
    else:
        # Try to find a default model
        default_pth = Path("models/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth")
        default_onnx = Path("models/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.onnx")

        if default_onnx.exists():
            onnx_path = default_onnx
            logger.info(f"Using existing ONNX model: {onnx_path}")
        elif default_pth.exists():
            pth_path = default_pth
            onnx_path = default_onnx
            logger.info(f"Using PyTorch model: {pth_path}")
        else:
            logger.error("No model found. Specify --pth or --onnx")
            logger.error("Download a pre-trained model from:")
            logger.error("  https://github.com/KaiyangZhou/deep-person-reid#model-zoo")
            return 1

    # Determine output path
    if args.output:
        engine_path = Path(args.output)
    else:
        engine_path = Path("models/osnet_ain_x1_0.engine")

    # Create models directory if needed
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert PyTorch to ONNX if needed
    if pth_path and not Path(onnx_path).exists():
        arch = args.arch or detect_architecture(str(pth_path))
        logger.info(f"Converting PyTorch to ONNX: {pth_path} -> {onnx_path}")
        try:
            convert_to_onnx(str(pth_path), str(onnx_path), arch)
        except Exception as e:
            logger.error(f"Failed to convert to ONNX: {e}")
            return 1

    if not Path(onnx_path).exists():
        logger.error(f"ONNX model not found: {onnx_path}")
        return 1

    # Step 2: Convert ONNX to TensorRT
    fp16 = not args.fp32

    # Try trtexec first (faster), fall back to Python API
    success = convert_with_trtexec(str(onnx_path), str(engine_path), fp16, args.workspace)
    if not success:
        success = convert_with_tensorrt_api(str(onnx_path), str(engine_path), fp16)

    if not success:
        logger.error("Failed to convert to TensorRT")
        return 1

    # Print config instructions
    print()
    print("=" * 60)
    print("SUCCESS! To use this model, update your config.yaml:")
    print("=" * 60)
    print()
    print("reid:")
    print("  use_tensorrt_native: true")
    print(f"  trt_model: \"{engine_path}\"")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
