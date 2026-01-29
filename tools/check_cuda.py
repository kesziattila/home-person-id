#!/usr/bin/env python3
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_cuda():
    print("--- OpenCV CUDA Support Checker ---")
    
    # 1. Check if cv2 is available
    try:
        import cv2
    except ImportError:
        logger.error("OpenCV (cv2) is not installed in the current environment.")
        return False

    print(f"OpenCV Version: {cv2.__version__}")

    # 2. Check for CUDA module
    has_cuda_module = hasattr(cv2, 'cuda')
    if has_cuda_module:
        print("cv2.cuda module: FOUND")
    else:
        logger.error("cv2.cuda module: NOT FOUND")
        print("\nPossible reasons:")
        print("- OpenCV was not compiled with CUDA support (WITH_CUDA=OFF).")
        print("- You are using a pip version of opencv-python which usually doesn't include CUDA.")
        print("\nOn Jetson, try using the system OpenCV:")
        print("sudo apt install libopencv-python")
        return False

    # 3. Check for CUDA devices
    try:
        device_count = cv2.cuda.getCudaEnabledDeviceCount()
        print(f"CUDA Device Count: {device_count}")
        
        if device_count > 0:
            for i in range(device_count):
                cv2.cuda.printShortCudaDeviceInfo(i)
            print("CUDA hardware: READY")
        else:
            logger.warning("CUDA hardware: NOT FOUND or NOT ACCESSIBLE")
            return False
            
    except Exception as e:
        logger.error(f"Error checking CUDA devices: {e}")
        return False

    # 4. Check specific required functions
    try:
        # Try to create a MOG2 subtractor on GPU
        subtractor = cv2.cuda.createBackgroundSubtractorMOG2()
        print("CUDA MOG2 initialization: SUCCESS")
    except Exception as e:
        logger.error(f"CUDA MOG2 initialization: FAILED - {e}")
        return False

    print("\nConclusion: Your environment supports CUDA-accelerated motion detection!")
    return True

if __name__ == "__main__":
    success = check_cuda()
    sys.exit(0 if success else 1)
