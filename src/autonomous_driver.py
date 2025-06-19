"""
Autonomous driving module using trained CNN model
"""

import cv2
import numpy as np
import tensorflow as tf
import time
import logging
import argparse
from pathlib import Path
import yaml
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.motor_control import MotorController
from src.core.perception import PerceptionModule
from src.core.safety_monitor import SafetyMonitor
from src.core.sensor_fusion import SensorFusion
from src.utils.data_augmentation import preprocess_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutonomousDriver:
    """
    Autonomous driving system using trained neural network
    """
    
    def __init__(self, model_path: str, config_path: str = "config/config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully")
        
        # Initialize components
        self.motor_controller = MotorController()
        self.perception = PerceptionModule()
        self.safety_monitor = SafetyMonitor()
        self.sensor_fusion = SensorFusion()
        
        # Control parameters
        self.max_speed = self.config['autonomous']['max_speed']
        self.speed_factor = self.config['autonomous']['speed_factor']
        self.steering_smoothing = self.config['autonomous']['steering_smoothing']
        
        # State variables
        self.is_running = False
        self.manual_override = False
        self.current_speed = 0.0
        self.current_steering = 0.0
        self.prev_steering = 0.0
        
        # Performance monitoring
        self.fps = 0
        self.inference_time = 0
        self.frame_count = 0
        self.start_time = time.time()
    
    def run(self):
        """Main autonomous driving loop"""
        self.perception.start()
        self.is_running = True
        
        logger.info("Starting autonomous driving...")
        logger.info("Press 'SPACE' for emergency stop, 'M' for manual override, 'Q' to quit")
        
        try:
            while self.is_running:
                # Get camera frame
                frame = self.perception.get_display_frame()
                if frame is None:
                    continue
                
                # Process frame for inference
                processed_frame = self.preprocess_frame(frame)
                
                # Run inference
                t_start = time.time()
                steering_pred = self.predict_steering(processed_frame)
                self.inference_time = (time.time() - t_start) * 1000  # ms
                
                # Smooth steering
                steering_command = self.smooth_steering(steering_pred)
                
                # Calculate speed based on steering
                speed_command = self.calculate_speed(steering_command)
                
                # Safety check
                sensor_data = self.get_sensor_data()
                if self.safety_monitor.check_sensors(sensor_data):
                    safe_speed, safe_steering = self.safety_monitor.check_command(
                        speed_command, steering_command
                    )
                else:
                    safe_speed, safe_steering = 0.0, 0.0
                
                # Apply commands if not in manual override
                if not self.manual_override:
                    self.motor_controller.set_speed_steering(safe_speed, safe_steering)
                    self.current_speed = safe_speed
                    self.current_steering = safe_steering
                
                # Handle keyboard input
                self.handle_input(cv2.waitKey(1) & 0xFF)
                
                # Annotate and display frame
                annotated_frame = self.annotate_frame(frame)
                cv2.imshow('Autonomous Driving', annotated_frame)
                
                # Update FPS
                self.update_fps()
                
        except KeyboardInterrupt:
            logger.info("Autonomous driving interrupted")
        finally:
            self.cleanup()
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model input"""
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Preprocess (crop and resize)
        processed = preprocess_image(
            rgb_frame, 
            target_size=(120, 160),
            crop=True
        )
        
        # Add batch dimension
        return np.expand_dims(processed, axis=0)
    
    def predict_steering(self, processed_frame):
        """Predict steering angle from processed frame"""
        # Run inference
        predictions = self.model.predict(processed_frame, verbose=0)
        
        # Handle different model outputs
        if isinstance(predictions, list):
            # Multi-output model (steering, throttle)
            steering = predictions[0][0][0]
        else:
            # Single output model
            steering = predictions[0][0]
        
        # Clip to valid range
        steering = np.clip(steering, -1.0, 1.0)
        
        return steering
    
    def smooth_steering(self, steering):
        """Apply exponential smoothing to steering commands"""
        alpha = self.steering_smoothing
        smoothed = alpha * steering + (1 - alpha) * self.prev_steering
        self.prev_steering = smoothed
        return smoothed
    
    def calculate_speed(self, steering_angle):
        """Calculate speed based on steering angle"""
        # Reduce speed when turning
        speed_reduction = 1.0 - min(abs(steering_angle), 0.5)
        speed = self.max_speed * self.speed_factor * speed_reduction
        
        return speed
    
    def get_sensor_data(self):
        """Get current sensor readings"""
        # Get perception data
        obstacle_distance = self.perception.get_obstacle_distance()
        
        # Get fusion data
        fusion_state = self.sensor_fusion.get_state()
        
        return {
            'ultrasonic_distance': obstacle_distance,
            'speed': fusion_state['speed'],
            'position': fusion_state['position'],
            'heading': fusion_state['heading']
        }
    
    def handle_input(self, key):
        """Handle keyboard input"""
        if key == ord(' '):  # Emergency stop
            logger.warning("EMERGENCY STOP!")
            self.motor_controller.stop()
            self.manual_override = True
        elif key == ord('m'):  # Toggle manual override
            self.manual_override = not self.manual_override
            if self.manual_override:
                logger.info("Manual override ENABLED")
                self.motor_controller.stop()
            else:
                logger.info("Manual override DISABLED")
        elif key == ord('q'):  # Quit
            self.is_running = False
        elif self.manual_override:
            # Manual control
            if key == ord('w'):
                self.motor_controller.set_speed_steering(0.5, 0)
            elif key == ord('s'):
                self.motor_controller.set_speed_steering(-0.5, 0)
            elif key == ord('a'):
                self.motor_controller.set_speed_steering(0.3, -0.8)
            elif key == ord('d'):
                self.motor_controller.set_speed_steering(0.3, 0.8)
            else:
                self.motor_controller.stop()
    
    def annotate_frame(self, frame):
        """Add information overlay to frame"""
        h, w = frame.shape[:2]
        annotated = frame.copy()
        
        # Draw steering visualization
        center_x = w // 2
        bottom_y = h - 50
        steer_x = int(center_x + self.current_steering * 100)
        cv2.arrowedLine(annotated, (center_x, bottom_y), 
                       (steer_x, bottom_y - 30), 
                       (0, 255, 0), 3)
        
        # Add status text
        status = "MANUAL" if self.manual_override else "AUTONOMOUS"
        cv2.putText(annotated, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add metrics
        cv2.putText(annotated, f"FPS: {self.fps}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(annotated, f"Inference: {self.inference_time:.1f}ms", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(annotated, f"Speed: {self.current_speed:.2f}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(annotated, f"Steering: {self.current_steering:.2f}", (10, 135),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw predicted path
        self.draw_predicted_path(annotated)
        
        # Show safety warnings
        warnings = self.safety_monitor.warnings
        for i, warning in enumerate(warnings[:3]):
            cv2.putText(annotated, warning, (10, h - 20 - i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return annotated
    
    def draw_predicted_path(self, frame):
        """Draw predicted driving path based on steering angle"""
        h, w = frame.shape[:2]
        
        # Path parameters
        path_length = 100
        num_points = 20
        
        # Calculate path points
        points = []
        for i in range(num_points):
            t = i / float(num_points - 1)
            
            # Simple arc based on steering angle
            if abs(self.current_steering) < 0.01:
                # Straight line
                x = w // 2
                y = h - 50 - int(t * path_length)
            else:
                # Curved path
                radius = 200 / abs(self.current_steering)
                angle = t * abs(self.current_steering)
                
                if self.current_steering > 0:
                    x = int(w // 2 + radius * np.sin(angle))
                else:
                    x = int(w // 2 - radius * np.sin(angle))
                
                y = h - 50 - int(radius * (1 - np.cos(angle)))
            
            points.append((x, y))
        
        # Draw path
        for i in range(1, len(points)):
            cv2.line(frame, points[i-1], points[i], (255, 255, 0), 2)
    
    def update_fps(self):
        """Update FPS calculation"""
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.start_time
            self.fps = int(self.frame_count / elapsed)
            self.frame_count = 0
            self.start_time = time.time()
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Shutting down autonomous driver...")
        self.motor_controller.stop()
        self.motor_controller.cleanup()
        self.perception.stop()
        cv2.destroyAllWindows()
        logger.info("Cleanup complete")


def main():
    parser = argparse.ArgumentParser(description="Run autonomous driving")
    parser.add_argument('--model', '-m', required=True,
                        help='Path to trained model')
    parser.add_argument('--config', '-c', default='config/config.yaml',
                        help='Configuration file')
    parser.add_argument('--max-speed', type=float, default=0.5,
                        help='Maximum speed (0-1)')
    
    args = parser.parse_args()
    
    # Create driver
    driver = AutonomousDriver(args.model, args.config)
    
    # Override max speed if specified
    if args.max_speed:
        driver.max_speed = args.max_speed
    
    # Run
    driver.run()


if __name__ == '__main__':
    main()
