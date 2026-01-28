# Running on NVIDIA Jetson

This guide covers running Home Person ID on NVIDIA Jetson devices (tested on Jetson Orin Nano 8GB).

## Overview: Jetson vs Desktop

| Aspect | Desktop (x86) | Jetson (ARM) |
|--------|---------------|--------------|
| PyTorch | Standard pip install | JetPack pre-built wheels |
| ONNX Runtime | `onnxruntime` or `onnxruntime-gpu` | `onnxruntime-gpu` from NVIDIA |
| TensorRT | Optional optimization | Highly recommended (2-3x faster) |
| Memory | Separate CPU/GPU RAM | Unified memory (shared) |
| Models | Can use larger models | Prefer smaller models |

---

## Installation

### 1. Prerequisites

Ensure JetPack is installed (includes CUDA, cuDNN, TensorRT):

```bash
# Check JetPack version
cat /etc/nv_tegra_release

# Should show CUDA
nvcc --version
```

### 2. Create Virtual Environment

```bash
cd ~/home-person-id
python3 -m venv venv
source venv/bin/activate
```

### 3. Install PyTorch for Jetson

**Do NOT use standard `pip install torch`** - it installs x86 version. Jetson uses ARM architecture and needs NVIDIA's pre-built wheels:

```bash
# Install PyTorch based on your JetPack and CUDA version
# Example for JetPack 6.x and CUDA 12.6:
pip install torch torchvision --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

### 4. Install ONNX Runtime for Jetson

**IMPORTANT:** onnxruntime-gpu 1.23.0 requires numpy<2.0. Install numpy first:

```bash
# Force numpy<2.0 (required for onnxruntime-gpu compatibility)
pip install "numpy<2.0,>=1.24"

# Install ONNX Runtime GPU from NVIDIA's Jetson AI Lab
pip install onnxruntime-gpu --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

# Or download wheel directly from:
# https://pypi.jetson-ai-lab.io/jp6/cu126/onnxruntime-gpu/

# Verify CUDA provider is available:
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
# Should show: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

### 5. Install OpenCV with GStreamer (for NVDEC hardware decoding)

The pip `opencv-python` packages don't include GStreamer support. For NVDEC hardware video decoding, use the system OpenCV via symlink:

```bash
# Install system OpenCV (has GStreamer + CUDA)
sudo apt install python3-opencv

# Remove any pip opencv from venv
pip uninstall opencv-python opencv-python-headless -y

# Symlink system cv2 into venv
VENV_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
ln -s /usr/lib/python3/dist-packages/cv2.cpython-310-aarch64-linux-gnu.so $VENV_SITE/

# Verify GStreamer is available:
python -c "import cv2; print(cv2.getBuildInformation())" | grep -i gstreamer
# Should show: GStreamer: YES
```

### 6. Install Other Dependencies

```bash
# Install Jetson-specific requirements with constraints
# Constraints file locks numpy<2.0 and prevents opencv pip packages
pip install -c constraints-jetson.txt -r requirements-jetson.txt
```

### 7. Copy Jetson Config

```bash
cp config/config.jetson.yaml config/config.yaml
# Edit config.yaml with your camera URLs
```

---

## Requirements Files

### Desktop: `requirements.txt`
Standard pip packages for x86 systems.

### Jetson: `requirements-jetson.txt`
Excludes packages that need special Jetson builds:
- `torch` / `torchvision` - Install from NVIDIA wheels
- `onnxruntime-gpu` - Install from NVIDIA wheels

---

## Model Conversion (TensorRT)

TensorRT provides 2-3x faster inference on Jetson. Convert models before running.

### YOLO to TensorRT

TensorRT is required for `.engine` models.

```bash
# Activate your venv
source venv/bin/activate

# Export to TensorRT engine (FP16 for speed)
yolo export model=yolov8n.pt format=engine device=0 half=True

# This creates yolov8n.engine in the same directory
# Move it to a known location:
mv yolov8n.engine models/

# Update config.yaml:
# detection:
#   model: "models/yolov8n.engine"
```

If you get `ModuleNotFoundError: No module named 'tensorrt'`, follow these steps:

1. **Install TensorRT Python bindings:**
   ```bash
   sudo apt update
   sudo apt install python3-libnvinfer
   ```

2. **Symlink into virtual environment:**
   The `apt` package installs to system site-packages. You need to symlink it to your `venv`:
   ```bash
   VENV_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
   # Find where tensorrt is installed in system packages (usually /usr/lib/python3.x/dist-packages/)
   # For JetPack 6 (Ubuntu 22.04), it's usually:
   ln -s /usr/lib/python3/dist-packages/tensorrt* $VENV_SITE/
   ```

**Model size comparison:**

| Model | Size | Speed (Jetson Orin Nano) |
|-------|------|--------------------------|
| yolov8n.pt | 6MB | ~30 FPS |
| yolov8n.engine (FP16) | ~15MB | ~60-80 FPS |
| yolov8s.pt | 22MB | ~15 FPS |
| yolov8s.engine (FP16) | ~40MB | ~30-40 FPS |

### InsightFace (Optional)

InsightFace models use ONNX format which automatically uses CUDA on Jetson when `onnxruntime-gpu` is installed. To enable full TensorRT acceleration:

1. **Update config.yaml:**
   ```yaml
   face_recognition:
     use_tensorrt: true
     trt_max_workspace_size: 536870912  # 512MB
   ```

2. **Wait for engine build:**
   The first time face recognition is used, it will build the TensorRT engine. This may take a few minutes.

### Re-ID Models (Optional)

OSNet models from torchreid run on PyTorch. For TensorRT optimization:

1. **Convert the model to ONNX** using the provided utility script:
   ```bash
   # Usage: python tools/convert_reid_to_onnx.py <path_to_pth> --arch <architecture>
   python tools/convert_reid_to_onnx.py models/osnet_ain_x1_0.pth --arch osnet_ain_x1_0
   ```

2. **Update config.yaml** to use the ONNX model:
   ```yaml
   reid:
     model: "models/osnet_ain_x1_0.onnx"
   ```

**NOTE:** onnxruntime will automatically create and cache a `.engine` file the first time it's loaded. This initial load may take several minutes.

---

## Hardware Acceleration (nvJPEG)

### Install nvjpeg for hardware-accelerated JPEG encoding

`pynvjpeg` provides a Python wrapper for the NVIDIA nvJPEG library, which significantly speeds up JPEG encoding for snapshots and MJPEG streaming.

1. **Ensure `nvcc` is in your PATH:**
   The installer needs to compile C++ code using the CUDA compiler.
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   ```

2. **Install `pynvjpeg`:**
   ```bash
   source venv/bin/activate
   pip install pynvjpeg
   ```

---

## Configuration Differences

### Key Jetson-Optimized Settings

```yaml
detection:
  model: "yolov8n.engine"      # TensorRT engine (faster)
  # Or: "yolov8n.pt" if not using TensorRT
  frame_skip: 1                 # No need to skip frames on GPU

reid:
  model: "osnet_x0_25"          # Smallest model (~150MB)
  # Desktop can use: "osnet_x1_0" (~350MB)

face_recognition:
  model: "buffalo_sc"           # Smallest model (~150MB)
  # Desktop can use: "buffalo_l" (~750MB)
```

### Memory Usage Comparison

| Config | Desktop | Jetson 8GB |
|--------|---------|------------|
| YOLO | yolov8s (~400MB) | yolov8n (~300MB) |
| Face | buffalo_l (~750MB) | buffalo_sc (~150MB) |
| Re-ID | osnet_x1_0 (~350MB) | osnet_x0_25 (~150MB) |
| **Total** | ~1.5GB | ~600MB |

---

## Performance Expectations

### Jetson Orin Nano 8GB

| Cameras | FPS/cam | Notes |
|---------|---------|-------|
| 1 | 10-15 | All features enabled |
| 2 | 5-10 | All features enabled |
| 3-4 | 5 | May need to disable Re-ID |

### Optimization Tips

1. **Use TensorRT** - 2-3x faster inference
2. **Enable motion detection** - Skip frames without motion
3. **Reduce FPS** - 5 FPS is usually sufficient
4. **Disable Re-ID** - Saves ~300MB if not needed
5. **Use frame_skip** - Process every 2nd frame if needed

---

## Troubleshooting

### "Illegal instruction" error

PyTorch was installed from pip (x86 version). Reinstall from NVIDIA wheels.

### Out of memory

Jetson has unified memory - GPU and CPU share RAM. Solutions:
- Use smaller models (osnet_x0_25, buffalo_sc)
- Disable Re-ID (`reid.enabled: false`)
- Reduce `reid.gallery_size`
- Enable `motion.enabled: true`

### Slow inference

- Convert models to TensorRT
- Check GPU is being used: `tegrastats` should show GPU activity
- Ensure `onnxruntime-gpu` is installed (not `onnxruntime`)

### Model download fails

Some models are downloaded on first run. On headless Jetson:
```bash
# Pre-download YOLO model
yolo predict model=yolov8n.pt source=https://ultralytics.com/images/bus.jpg

# Pre-download InsightFace model
python -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_sc'); app.prepare(ctx_id=0)"
```

### TensorRT engine incompatibility

TensorRT engines are device-specific. If you get errors after JetPack update:
```bash
# Delete old engine and re-export
rm models/yolov8n.engine
yolo export model=yolov8n.pt format=engine device=0 half=True
mv yolov8n.engine models/
```

---

## Quick Start Checklist

1. [ ] JetPack installed and CUDA working
2. [ ] PyTorch installed from NVIDIA wheels (not pip)
3. [ ] onnxruntime-gpu installed from NVIDIA wheels
4. [ ] Other dependencies: `pip install -r requirements-jetson.txt`
5. [ ] Config copied: `cp config/config.jetson.yaml config/config.yaml`
6. [ ] Camera URLs configured in config.yaml
7. [ ] (Optional) YOLO exported to TensorRT: `yolo export ...`
8. [ ] Test: `python -m src.cli preview --camera <id>`
