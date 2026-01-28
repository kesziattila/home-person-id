import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

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
                # pynvjpeg uses RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return nj.encode(rgb_image)
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
