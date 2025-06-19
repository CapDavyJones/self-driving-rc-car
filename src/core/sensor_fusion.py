"""
Sensor Fusion Module
Combines data from multiple sensors for robust state estimation
"""

import numpy as np
import time
import threading
from typing import Dict, Optional, Tuple
import logging
from collections import deque

logger = logging.getLogger(__name__)


class KalmanFilter:
    """Simple Kalman filter for position estimation"""
    
    def __init__(self):
        # State: [x, y, vx, vy]
        self.state = np.zeros(4)
        self.P = np.eye(4) * 100  # Covariance matrix
        
        # Process noise
        self.Q = np.eye(4) * 0.1
        
        # Measurement noise
        self.R = np.eye(2) * 1.0
        
        # State transition matrix
        self.F = np.eye(4)
        
        # Measurement matrix
        self.H = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0]])
    
    def predict(self, dt: float):
        """Predict next state"""
        # Update state transition matrix with dt
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        
        # Predict state
        self.state = self.F @ self.state
        
        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, measurement: np.ndarray):
        """Update with measurement"""
        # Innovation
        y = measurement - self.H @ self.state
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P


class SensorFusion:
    """
    Fuses data from multiple sensors for robust state estimation
    """
    
    def __init__(self):
        self.kalman = KalmanFilter()
        self.last_update = time.time()
        
        # Sensor data buffers
        self.imu_buffer = deque(maxlen=10)
        self.gps_buffer = deque(maxlen=5)
        self.encoder_buffer = deque(maxlen=20)
        
        # Current estimates
        self.position = np.zeros(2)  # x, y
        self.velocity = np.zeros(2)  # vx, vy
        self.heading = 0.0  # radians
        self.angular_velocity = 0.0
        
        self._lock = threading.Lock()
        logger.info("Sensor fusion initialized")
    
    def update_imu(self, accel: Tuple[float, float, float], 
                   gyro: Tuple[float, float, float]):
        """Update with IMU data"""
        with self._lock:
            timestamp = time.time()
            self.imu_buffer.append({
                'timestamp': timestamp,
                'accel': accel,
                'gyro': gyro
            })
            
            # Update angular velocity (z-axis)
            self.angular_velocity = gyro[2]
            
            # Integrate to update heading
            if len(self.imu_buffer) > 1:
                dt = timestamp - self.imu_buffer[-2]['timestamp']
                self.heading += self.angular_velocity * dt
                self.heading = self._normalize_angle(self.heading)
    
    def update_gps(self, lat: float, lon: float):
        """Update with GPS data (for outdoor use)"""
        with self._lock:
            # Convert lat/lon to local coordinates
            # This is simplified - real implementation would use proper projection
            x = lon * 111320.0  # meters (approximate)
            y = lat * 110540.0  # meters (approximate)
            
            timestamp = time.time()
            self.gps_buffer.append({
                'timestamp': timestamp,
                'position': np.array([x, y])
            })
            
            # Update Kalman filter
            if len(self.gps_buffer) > 0:
                self.kalman.update(np.array([x, y]))
    
    def update_encoders(self, left_ticks: int, right_ticks: int, 
                       wheel_base: float = 0.2, wheel_radius: float = 0.05):
        """Update with wheel encoder data"""
        with self._lock:
            timestamp = time.time()
            self.encoder_buffer.append({
                'timestamp': timestamp,
                'left': left_ticks,
                'right': right_ticks
            })
            
            if len(self.encoder_buffer) > 1:
                # Calculate odometry
                prev = self.encoder_buffer[-2]
                dt = timestamp - prev['timestamp']
                
                # Ticks to distance (simplified)
                ticks_per_rev = 20  # Example value
                left_dist = (left_ticks - prev['left']) * 2 * np.pi * wheel_radius / ticks_per_rev
                right_dist = (right_ticks - prev['right']) * 2 * np.pi * wheel_radius / ticks_per_rev
                
                # Calculate movement
                distance = (left_dist + right_dist) / 2
                delta_heading = (right_dist - left_dist) / wheel_base
                
                # Update position estimate
                self.heading += delta_heading
                self.heading = self._normalize_angle(self.heading)
                
                dx = distance * np.cos(self.heading)
                dy = distance * np.sin(self.heading)
                
                self.position[0] += dx
                self.position[1] += dy
                
                # Update velocity
                if dt > 0:
                    self.velocity[0] = dx / dt
                    self.velocity[1] = dy / dt
    
    def get_state(self) -> Dict:
        """Get current fused state estimate"""
        with self._lock:
            # Update Kalman filter
            current_time = time.time()
            dt = current_time - self.last_update
            
            if dt > 0:
                self.kalman.predict(dt)
                
                # Get Kalman estimate
                kalman_pos = self.kalman.state[:2]
                kalman_vel = self.kalman.state[2:]
                
                # Fuse with odometry (simple weighted average)
                alpha = 0.7  # Weight for Kalman estimate
                self.position = alpha * kalman_pos + (1 - alpha) * self.position
                self.velocity = alpha * kalman_vel + (1 - alpha) * self.velocity
            
            self.last_update = current_time
            
            return {
                'position': self.position.copy(),
                'velocity': self.velocity.copy(),
                'speed': np.linalg.norm(self.velocity),
                'heading': self.heading,
                'angular_velocity': self.angular_velocity,
                'timestamp': current_time
            }
    
    def reset(self):
        """Reset all estimates to zero"""
        with self._lock:
            self.kalman = KalmanFilter()
            self.position = np.zeros(2)
            self.velocity = np.zeros(2)
            self.heading = 0.0
            self.angular_velocity = 0.0
            self.imu_buffer.clear()
            self.gps_buffer.clear()
            self.encoder_buffer.clear()
            logger.info("Sensor fusion reset")
    
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


if __name__ == "__main__":
    # Test sensor fusion
    fusion = SensorFusion()
    
    # Simulate some sensor data
    print("Testing sensor fusion...")
    
    # Simulate IMU updates
    for i in range(10):
        fusion.update_imu(
            accel=(0.1, 0.0, 9.8),
            gyro=(0.0, 0.0, 0.1)
        )
        time.sleep(0.1)
    
    # Simulate encoder updates
    for i in range(10):
        fusion.update_encoders(i * 10, i * 10)
        time.sleep(0.1)
    
    # Get fused state
    state = fusion.get_state()
    print(f"Fused state: {state}")
