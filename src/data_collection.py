"""
Data collection script for recording training data
"""

import cv2
import os
import csv
import argparse
import time
from datetime import datetime
import yaml
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.motor_control import MotorController
from src.core.perception import PerceptionModule
from src.core.safety_monitor import SafetyMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCollector:
    """Collect driving data with manual control"""
    
    def __init__(self, output_dir: str, config_path: str = "config/config.yaml"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup directories
        self.image_dir = self.output_dir / "images"
        self.image_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.motor_controller = MotorController()
        self.perception = PerceptionModule()
        self.safety_monitor = SafetyMonitor()
        
        # Setup logging
        self.log_file = self.output_dir / "driving_log.csv"
        self.csv_file = open(self.log_file, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['timestamp', 'image_file', 'steering', 'throttle', 'speed'])
        
        # Control state
        self.steering = 0.0
        self.throttle = 0.0
        self.recording = False
        
    def run(self):
        """Main data collection loop"""
        self.perception.start()
        logger.info("Starting data collection...")
        logger.info("Controls:")
        logger.info("  W/S - Forward/Backward")
        logger.info("  A/D - Left/Right")
        logger.info("  SPACE - Stop")
        logger.info("  R - Toggle recording")
        logger.info("  Q - Quit")
        
        frame_count = 0
        
        try:
            while True:
                # Get camera frame
                frame = self.perception.get_display_frame()
                if frame is None:
                    continue
                
                # Process keyboard input
                key = cv2.waitKey(1) & 0xFF
                self._handle_input(key)
                
                # Apply safety checks
                safe_throttle, safe_steering = self.safety_monitor.check_command(
                    self.throttle, self.steering
                )
                
                # Apply motor commands
                self.motor_controller.set_speed_steering(safe_throttle, safe_steering)
                
                # Annotate frame
                frame = self._annotate_frame(frame)
                
                # Display
                cv2.imshow('Data Collection', frame)
                
                # Save data if recording
                if self.recording and (self.throttle != 0 or self.steering != 0):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    image_file = f"{timestamp}.jpg"
                    image_path = self.image_dir / image_file
                    
                    # Save image
                    cv2.imwrite(str(image_path), frame)
                    
                    # Log data
                    speed = abs(self.throttle)  # Simplified speed estimate
                    self.csv_writer.writerow([
                        timestamp, image_file, self.steering, self.throttle, speed
                    ])
                    
                    frame_count += 1
                    if frame_count % 100 == 0:
                        logger.info(f"Recorded {frame_count} frames")
                        self.csv_file.flush()  # Ensure data is written
                
                # Check for quit
                if key == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            logger.info("Data collection interrupted")
        finally:
            self.cleanup()
    
    def _handle_input(self, key):
        """Handle keyboard input"""
        # Smooth control changes
        ACCEL_RATE = 0.1
        STEER_RATE = 0.1
        
        if key == ord('w'):  # Forward
            self.throttle = min(1.0, self.throttle + ACCEL_RATE)
        elif key == ord('s'):  # Backward/Brake
            self.throttle = max(-1.0, self.throttle - ACCEL_RATE)
        elif key == ord('a'):  # Left
            self.steering = max(-1.0, self.steering - STEER_RATE)
        elif key == ord('d'):  # Right
            self.steering = min(1.0, self.steering + STEER_RATE)
        elif key == ord(' '):  # Stop
            self.throttle = 0.0
            self.steering = 0.0
        elif key == ord('r'):  # Toggle recording
            self.recording = not self.recording
            status = "STARTED" if self.recording else "STOPPED"
            logger.info(f"Recording {status}")
        
        # Decay steering when not turning
        if key not in [ord('a'), ord('d')]:
            self.steering *= 0.9
            if abs(self.steering) < 0.05:
                self.steering = 0.0
        
        # Decay throttle when not accelerating
        if key not in [ord('w'), ord('s')]:
            self.throttle *= 0.95
            if abs(self.throttle) < 0.05:
                self.throttle = 0.0
    
    def _annotate_frame(self, frame):
        """Add information overlay to frame"""
        h, w = frame.shape[:2]
        
        # Recording indicator
        if self.recording:
            cv2.circle(frame, (w - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (w - 80, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Control indicators
        cv2.putText(frame, f"Throttle: {self.throttle:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Steering: {self.steering:.2f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw steering visualization
        center_x = w // 2
        bottom_y = h - 50
        steer_x = int(center_x + self.steering * 100)
        cv2.arrowedLine(frame, (center_x, bottom_y), 
                       (steer_x, bottom_y - 30), 
                       (255, 255, 0), 3)
        
        # Safety warnings
        warnings = self.safety_monitor.warnings
        for i, warning in enumerate(warnings[:3]):  # Show max 3 warnings
            cv2.putText(frame, warning, (10, h - 20 - i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up...")
        self.motor_controller.stop()
        self.motor_controller.cleanup()
        self.perception.stop()
        self.csv_file.close()
        cv2.destroyAllWindows()
        
        # Summary
        total_images = len(list(self.image_dir.glob("*.jpg")))
        logger.info(f"Data collection complete!")
        logger.info(f"Total images collected: {total_images}")
        logger.info(f"Data saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Collect training data for RC car")
    parser.add_argument('--output', '-o', default='data/session_new', 
                        help='Output directory for data')
    parser.add_argument('--config', '-c', default='config/config.yaml',
                        help='Configuration file path')
    
    args = parser.parse_args()
    
    collector = DataCollector(args.output, args.config)
    collector.run()


if __name__ == '__main__':
    main()
