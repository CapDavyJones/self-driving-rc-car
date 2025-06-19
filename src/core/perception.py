"""
Perception module for camera processing and image preprocessing
"""

import cv2
import numpy as np
import threading
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PerceptionModule:
    """Handles camera input and image preprocessing"""
    
    def __init__(self, camera_id: int = 0, target_size: Tuple[int, int] = (160, 120)):
        self.camera_id = camera_id
        self.target_size = target_size
        self.cap = None
        self.latest_frame = None
        self.running = False
        self._lock = threading.Lock()
        
    def start(self):
        """Start camera capture"""
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
        
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        logger.info("Perception module started")
    
    def _capture_loop(self):
        """Continuous capture loop"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self.latest_frame = frame
    
    def get_processed_frame(self) -> Optional[np.ndarray]:
        """Get preprocessed frame for model input"""
        with self._lock:
            if self.latest_frame is None:
                return None
            
            # Resize
            frame = cv2.resize(self.latest_frame, self.target_size)
            
            # Convert to RGB (OpenCV uses BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Normalize
            frame = frame.astype(np.float32) / 255.0
            
            return frame
    
    def get_display_frame(self) -> Optional[np.ndarray]:
        """Get frame for display (with annotations)"""
        with self._lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()
    
    def annotate_frame(self, frame: np.ndarray, steering: float, speed: float):
        """Add visual annotations to frame"""
        h, w = frame.shape[:2]
        
        # Draw steering indicator
        center_x = w // 2
        bottom_y = h - 30
        
        # Steering line
        steer_x = int(center_x + steering * 100)
        cv2.arrowedLine(frame, (center_x, bottom_y), 
                       (steer_x, bottom_y - 30), 
                       (0, 255, 0), 3)
        
        # Speed indicator
        speed_text = f"Speed: {speed:.2f}"
        cv2.putText(frame, speed_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Steering text
        steer_text = f"Steering: {steering:.2f}"
        cv2.putText(frame, steer_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame
    
    def stop(self):
        """Stop camera capture"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        logger.info("Perception module stopped")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def test_camera():
    """Test camera functionality"""
    print("Testing camera... Press 'q' to quit")
    
    with PerceptionModule() as perception:
        while True:
            frame = perception.get_display_frame()
            if frame is not None:
                # Add test annotations
                frame = perception.annotate_frame(frame, 0.5, 0.7)
                cv2.imshow('Camera Test', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    cv2.destroyAllWindows()
    print("Camera test completed")


if __name__ == "__main__":
    test_camera()
