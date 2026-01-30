import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ==================== Crop Utilities ====================

def crop_with_margin(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin_ratio: float = 0.3,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop a region from an image with margin around the bounding box.

    Args:
        image: Source image (HxWxC or HxW)
        bbox: Bounding box as (x1, y1, x2, y2)
        margin_ratio: Margin as ratio of bbox size (0.3 = 30%)

    Returns:
        Tuple of (cropped_image, adjusted_bbox)
        adjusted_bbox contains the actual coordinates used after bounds checking
    """
    if image.size == 0:
        return np.array([]), bbox

    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]

    # Calculate margin based on bbox size
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    margin = int(min(bbox_w, bbox_h) * margin_ratio)

    # Apply margin with bounds checking
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    crop = image[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)


def crop_bbox(
    image: np.ndarray,
    bbox: Union[Tuple[float, float, float, float], Tuple[int, int, int, int]],
) -> np.ndarray:
    """Crop a bounding box region from an image.

    Args:
        image: Source image (HxWxC or HxW)
        bbox: Bounding box as (x1, y1, x2, y2)

    Returns:
        Cropped image region
    """
    if image.size == 0:
        return np.array([])

    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]

    # Bounds checking
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    return image[y1:y2, x1:x2]


def crop_face_region(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    height_ratio: float = 0.5,
) -> np.ndarray:
    """Crop the upper face region from a person bounding box.

    Typically used to extract face region from a full-body detection.

    Args:
        image: Source image (HxWxC or HxW)
        bbox: Person bounding box as (x1, y1, x2, y2)
        height_ratio: Ratio of bbox height to use (0.5 = upper half)

    Returns:
        Cropped face region
    """
    if image.size == 0:
        return np.array([])

    x1, y1, x2, y2 = map(int, bbox)
    h = y2 - y1

    # Take upper portion (typically contains the face)
    face_y2 = y1 + int(h * height_ratio)

    return crop_bbox(image, (x1, y1, x2, face_y2))

# Global instances for nvjpeg
_nvjpeg_instance = None
_nvjpeg_available = None

def get_nvjpeg():
    """Get or initialize nvJPEG instance."""
    global _nvjpeg_instance, _nvjpeg_available
    
    if _nvjpeg_available is False:
        return None
        
    if _nvjpeg_instance is None:
        try:
            from nvjpeg import NvJpeg
            _nvjpeg_instance = NvJpeg()
            _nvjpeg_available = True
            logger.info("nvJPEG initialized successfully for hardware-accelerated encoding")
        except ImportError:
            _nvjpeg_available = False
            logger.debug("pynvjpeg not installed, falling back to CPU encoding")
        except Exception as e:
            _nvjpeg_available = False
            logger.warning(f"Failed to initialize nvJPEG: {e}. Falling back to CPU encoding")
            
    return _nvjpeg_instance

def encode_jpeg(image: np.ndarray, quality: int = 85, use_nvjpeg: bool = False) -> bytes:
    """Encode image to JPEG bytes.
    
    Args:
        image: BGR image array
        quality: JPEG quality (1-100)
        use_nvjpeg: Whether to attempt hardware-accelerated encoding
        
    Returns:
        JPEG bytes
    """
    if use_nvjpeg:
        nj = get_nvjpeg()
        if nj:
            try:
                # pynvjpeg expects BGR by default if we want to match OpenCV behavior
                return nj.encode(image)
            except Exception as e:
                logger.error(f"nvJPEG encoding failed: {e}. Falling back to CPU")
                
    # Fallback to OpenCV (CPU)
    ret, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ret:
        raise RuntimeError("JPEG encoding failed")
    return buffer.tobytes()

def write_jpeg(path: str, image: np.ndarray, quality: int = 85, use_nvjpeg: bool = False):
    """Write image to JPEG file.
    
    Args:
        path: Output file path
        image: BGR image array
        quality: JPEG quality
        use_nvjpeg: Whether to attempt hardware-accelerated encoding
    """
    jpeg_bytes = encode_jpeg(image, quality, use_nvjpeg)
    with open(path, 'wb') as f:
        f.write(jpeg_bytes)
