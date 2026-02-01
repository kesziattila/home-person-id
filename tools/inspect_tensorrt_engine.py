#!/usr/bin/env python3
"""Inspect TensorRT engine input/output tensor shapes.

Usage:
    python tools/inspect_tensorrt_engine.py path/to/model.engine
"""

import sys


def inspect_engine(engine_path: str):
    """Print input/output tensor information for a TensorRT engine."""
    try:
        import tensorrt as trt
    except ImportError:
        print("Error: tensorrt not installed")
        sys.exit(1)

    print(f"Loading engine: {engine_path}")

    trt_logger = trt.Logger(trt.Logger.WARNING)

    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(trt_logger)
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        print("Error: Failed to load engine")
        sys.exit(1)

    print(f"\nTensorRT version: {trt.__version__}")
    print(f"Number of I/O tensors: {engine.num_io_tensors}")
    print()

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        mode = engine.get_tensor_mode(name)

        mode_str = "INPUT" if mode == trt.TensorIOMode.INPUT else "OUTPUT"

        print(f"[{i}] {name}")
        print(f"    Mode:  {mode_str}")
        print(f"    Shape: {list(shape)}")
        print(f"    Dtype: {dtype}")
        print()

    # Additional info for YOLO models
    print("=" * 50)
    print("YOLO Format Guide:")
    print("  YOLOv8/11 raw:    (1, 84, 8400) = 4 bbox + 80 classes")
    print("  YOLOv8/11 w/NMS:  (1, N, 6) = x1,y1,x2,y2,conf,class")
    print("  Custom classes:   (1, 4+num_classes, predictions)")
    print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <engine_path>")
        sys.exit(1)

    inspect_engine(sys.argv[1])
