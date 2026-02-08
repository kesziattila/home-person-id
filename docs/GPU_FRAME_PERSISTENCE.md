# GPU Frame Persistence - Research Findings

> **Status**: Deferred
> **Date**: January 2026
> **Decision**: Keep current implementation; revisit when scaling to 4+ cameras

## Summary

Research into zero-copy NVDEC→GpuMat direct path on Jetson reveals it's **possible but requires significant C++ integration work**.

## The Problem

```
Current: NVDEC (GPU) → nvvidconv → videoconvert → CPU → appsink → Frame.image (numpy)
                                      ↑                    ↓
                                  GPU→CPU copy      CPU→GPU copy (motion detector)
```

Each frame currently makes a round-trip between GPU and CPU memory, adding ~2-5ms latency per frame.

## The Solution: CUDA-EGL Interop

On Jetson, NVMM memory can be accessed as GpuMat via:

1. **NvBufSurfaceMapEglImage()** - Map NVMM buffer to EGL image
2. **cuGraphicsEGLRegisterImage()** - Register EGL image with CUDA
3. **cuGraphicsResourceGetMappedEglFrame()** - Get CUDA pointer
4. **cv::cuda::GpuMat()** - Wrap CUDA pointer as GpuMat

## Implementation Requirements

### GStreamer Pipeline Change

```bash
# Current (CPU output):
nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink

# Zero-copy (NVMM output):
nvv4l2decoder ! nvvidconv ! video/x-raw(memory:NVMM),format=RGBA ! appsink
```

### C++ Pad Probe Code Required

```cpp
static GstPadProbeReturn buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
    GstBuffer *buffer = (GstBuffer *)info->data;
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_WRITE);

    NvBufSurface* surf = (NvBufSurface*)map.data;
    NvBufSurfaceParams& params = surf->surfaceList[0];

    // Map to EGL
    NvBufSurfaceMapEglImage(surf, 0);
    EGLImageKHR egl_image = params.mappedAddr.eglImage;

    // Register with CUDA (cache registration for performance)
    CUgraphicsResource cu_res;
    cuGraphicsEGLRegisterImage(&cu_res, egl_image, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);

    // Get CUDA frame
    CUeglFrame eglFrame;
    cuGraphicsResourceGetMappedEglFrame(&eglFrame, cu_res, 0, 0);

    // Create GpuMat (zero-copy!)
    cv::cuda::GpuMat gpu_mat(eglFrame.height, params.pitch / 4, CV_8UC4,
                              (uchar*)eglFrame.frame.pPitch[0]);

    // ... process gpu_mat ...

    NvBufSurfaceUnMapEglImage(surf, 0);
    gst_buffer_unmap(buffer, &map);
    return GST_PAD_PROBE_OK;
}
```

## Performance Impact

Per the [NVIDIA forum discussion](https://forums.developer.nvidia.com/t/gstreamer-writing-to-cuda-memory-and-zero-copy-cv-gpumat-with-jetpack-5-1-2/338039):

- EGLMap: ~300-330µs per frame
- EGLUnmap: ~550µs per frame
- **Eliminates**: ~2-5ms CPU↔GPU copy per frame

## Complexity Assessment

| Aspect | Current (Python) | Zero-Copy (C++ Required) |
|--------|-----------------|--------------------------|
| GStreamer interaction | cv2.VideoCapture() | Custom pad probe in C++ |
| Frame handling | numpy array | GpuMat from CUDA-EGL interop |
| Integration | Pure Python | C++ extension via pybind11/Cython |
| Dependencies | opencv-python | OpenCV C++, CUDA, NvBufSurface API |
| JetPack requirement | Any | 5.1+ (tested), 6+ ideal |

## Implementation Options

### Option A: Full Zero-Copy (High Effort, High Reward)

**Scope**: Write C++ extension module with pybind11

1. Create `src/stream/nvmm_capture.cpp`:
   - Custom GStreamer pipeline with pad probe
   - CUDA-EGL interop for zero-copy GpuMat
   - Export Python bindings via pybind11

2. Modify `src/stream/rtsp_client.py`:
   - Use C++ module when `use_nvdec=True` and on Jetson
   - Fallback to current Python approach otherwise

3. Frame dataclass gains `gpu_image: GpuMat` populated directly

**Estimated effort**: 2-3 days
**Performance gain**: ~2-5ms per frame (eliminates GPU↔CPU round-trip)

### Option B: Hybrid Approach (Medium Effort, Medium Reward)

**Scope**: Use existing Python + optimize motion detector

1. Keep current Python GStreamer/appsink approach
2. Add `gpu_frame` parameter to motion detector (API prep)
3. Cache uploaded GpuMat per camera to avoid re-allocation (already done)

**Estimated effort**: 2-4 hours
**Performance gain**: Minimal (just avoids GpuMat re-allocation, which is already optimized)

### Option C: DeepStream SDK (Alternative)

**Scope**: Use NVIDIA DeepStream for the entire pipeline

DeepStream provides built-in zero-copy and would replace current GStreamer/OpenCV approach. However:
- Steeper learning curve
- Different programming model
- May be overkill for this use case

## Decision: Deferred

**Chosen**: Keep current implementation. The C++ integration complexity outweighs the ~2-5ms/frame benefit at this stage.

### Conditions to Revisit

- Adding 4+ cameras where cumulative savings become significant
- Performance profiling shows GPU↔CPU transfer as a bottleneck
- JetPack 6+ provides easier NVMM access patterns

## References

- [GStreamer NVMM to GpuMat Zero-Copy (NVIDIA Forum)](https://forums.developer.nvidia.com/t/gstreamer-writing-to-cuda-memory-and-zero-copy-cv-gpumat-with-jetpack-5-1-2/338039)
- [DeepStream Custom GStreamer Plugin (NVIDIA Docs)](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_sample_custom_gstream.html)
- [RidgeRun GstCUDA (Commercial)](https://developer.ridgerun.com/wiki/index.php?title=GstCUDA)
- [NvBuffer to GpuMat Mapping (NVIDIA Forum)](https://forums.developer.nvidia.com/t/nvbuffer-mapping-to-gpumat/224427)
