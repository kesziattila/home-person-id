import unittest
from unittest.mock import MagicMock, patch
import time
import numpy as np
from src.stream.rtsp_client import RTSPClient, Frame

class TestRTSPClientNativeFPS(unittest.TestCase):
    def setUp(self):
        self.camera_id = "test_cam"
        self.rtsp_url = "rtsp://example.com/stream"

    @patch('cv2.VideoCapture')
    def test_native_fps_mode(self, mock_videocapture):
        # Setup mock VideoCapture
        mock_cap = MagicMock()
        mock_videocapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.grab.return_value = True
        
        # Mock retrieve to return a dummy image
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cap.retrieve.return_value = (True, dummy_image)
        
        # Initialize client with target_fps=None (native mode)
        client = RTSPClient(self.camera_id, self.rtsp_url, target_fps=None)
        client._cap = mock_cap  # Directly set the mock cap
        
        # Start the client (it runs in a background thread)
        with patch.object(client, '_connect', return_value=True):
            client._connected = True # Ensure it's marked as connected
            client.start()
            
            # Wait for a few frames to be captured
            frames = []
            start_time = time.time()
            while len(frames) < 5 and time.time() - start_time < 2.0:
                frame = client.get_frame(timeout=0.1)
                if frame:
                    frames.append(frame)
            
            client.stop()
            
        self.assertGreaterEqual(len(frames), 5)
        # In native mode, it should capture as fast as the mock allows
        # Since our mock is fast, we should get frames quickly.

    @patch('cv2.VideoCapture')
    def test_target_fps_mode(self, mock_videocapture):
        # Setup mock VideoCapture
        mock_cap = MagicMock()
        mock_videocapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.grab.return_value = True
        
        # Mock retrieve to return a dummy image
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cap.retrieve.return_value = (True, dummy_image)
        
        # Initialize client with target_fps=2 (low FPS to test interval)
        target_fps = 2
        client = RTSPClient(self.camera_id, self.rtsp_url, target_fps=target_fps)
        client._cap = mock_cap  # Directly set the mock cap
        
        # Start the client
        with patch.object(client, '_connect', return_value=True):
            client._connected = True # Ensure it's marked as connected
            client.start()
            
            # Capture frames for 1 second
            frames = []
            start_time = time.time()
            while time.time() - start_time < 1.1:
                frame = client.get_frame(timeout=0.1)
                if frame:
                    frames.append(frame)
            
            client.stop()
            
        # At 2 FPS, we should get about 2-3 frames in 1.1 seconds
        # Definitely fewer than if there was no rate limit
        self.assertLessEqual(len(frames), 4)
        self.assertGreaterEqual(len(frames), 1)

if __name__ == '__main__':
    unittest.main()
