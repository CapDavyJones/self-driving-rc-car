"""
Safety Monitor Module
Ensures safe operation with emergency stops and boundary checking
"""

import logging
import time
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SafetyLimits:
    """Safety configuration limits"""
    max_speed: float = 0.8
    max_steering: float = 0.9
    min_obstacle_distance: float = 30.0  # cm
    max_acceleration: float = 0.5
    emergency_stop_distance: float = 15.0  # cm
    max_tilt_angle: float = 45.0  # degrees


class SafetyMonitor:
    """
    Monitors system safety and can override controls
    """
    
    def __init__(self, limits: SafetyLimits = None):
        self.limits = limits or SafetyLimits()
        self.emergency_stop = False
        self.warnings = []
        self.last_speed = 0.0
        self.last_time = time.time()
        
        logger.info("Safety monitor initialized")
    
    def check_command(self, speed: float, steering: float) -> Tuple[float, float]:
        """
        Validate and potentially modify control commands
        
        Returns: (safe_speed, safe_steering)
        """
        # Clear previous warnings
        self.warnings = []
        
        # Check emergency stop
        if self.emergency_stop:
            self.warnings.append("EMERGENCY STOP ACTIVE")
            return 0.0, 0.0
        
        # Limit speed
        if abs(speed) > self.limits.max_speed:
            self.warnings.append(f"Speed limited from {speed:.2f} to {self.limits.max_speed:.2f}")
            speed = self.limits.max_speed if speed > 0 else -self.limits.max_speed
        
        # Limit steering
        if abs(steering) > self.limits.max_steering:
            self.warnings.append(f"Steering limited from {steering:.2f} to {self.limits.max_steering:.2f}")
            steering = self.limits.max_steering if steering > 0 else -self.limits.max_steering
        
        # Check acceleration
        current_time = time.time()
        dt = current_time - self.last_time
        if dt > 0:
            acceleration = (speed - self.last_speed) / dt
            if abs(acceleration) > self.limits.max_acceleration:
                # Limit acceleration
                max_delta = self.limits.max_acceleration * dt
                if acceleration > 0:
                    speed = min(speed, self.last_speed + max_delta)
                else:
                    speed = max(speed, self.last_speed - max_delta)
                self.warnings.append(f"Acceleration limited to {self.limits.max_acceleration:.2f}")
        
        self.last_speed = speed
        self.last_time = current_time
        
        return speed, steering
    
    def check_sensors(self, sensor_data: Dict) -> bool:
        """
        Check sensor data for safety violations
        
        Returns: True if safe to continue, False if should stop
        """
        # Check ultrasonic distance
        if 'ultrasonic_distance' in sensor_data:
            distance = sensor_data['ultrasonic_distance']
            
            if distance < self.limits.emergency_stop_distance:
                self.warnings.append(f"OBSTACLE TOO CLOSE: {distance:.1f}cm")
                self.trigger_emergency_stop()
                return False
            
            elif distance < self.limits.min_obstacle_distance:
                self.warnings.append(f"Obstacle detected at {distance:.1f}cm")
        
        # Check IMU tilt
        if 'imu_tilt' in sensor_data:
            tilt = abs(sensor_data['imu_tilt'])
            if tilt > self.limits.max_tilt_angle:
                self.warnings.append(f"EXCESSIVE TILT: {tilt:.1f}°")
                self.trigger_emergency_stop()
                return False
        
        # Check battery voltage
        if 'battery_voltage' in sensor_data:
            voltage = sensor_data['battery_voltage']
            if voltage < 6.0:  # Low battery
                self.warnings.append(f"LOW BATTERY: {voltage:.1f}V")
        
        return not self.emergency_stop
    
    def trigger_emergency_stop(self):
        """Activate emergency stop"""
        self.emergency_stop = True
        logger.error("EMERGENCY STOP TRIGGERED")
    
    def reset_emergency_stop(self):
        """Reset emergency stop (requires manual intervention)"""
        self.emergency_stop = False
        self.warnings = []
        logger.info("Emergency stop reset")
    
    def get_status(self) -> Dict:
        """Get current safety status"""
        return {
            'emergency_stop': self.emergency_stop,
            'warnings': self.warnings.copy(),
            'limits': {
                'max_speed': self.limits.max_speed,
                'max_steering': self.limits.max_steering,
                'min_obstacle_distance': self.limits.min_obstacle_distance
            }
        }
    
    def log_warnings(self):
        """Log any active warnings"""
        for warning in self.warnings:
            logger.warning(warning)


class SafetyOverride:
    """
    Context manager for temporary safety overrides (testing only!)
    """
    def __init__(self, safety_monitor: SafetyMonitor):
        self.monitor = safety_monitor
        self.original_limits = None
    
    def __enter__(self):
        logger.warning("SAFETY OVERRIDE ACTIVE - USE WITH CAUTION")
        self.original_limits = self.monitor.limits
        # Set permissive limits for testing
        self.monitor.limits = SafetyLimits(
            max_speed=1.0,
            max_steering=1.0,
            min_obstacle_distance=5.0,
            emergency_stop_distance=5.0
        )
        return self.monitor
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.limits = self.original_limits
        logger.info("Safety override ended")


if __name__ == "__main__":
    # Test safety monitor
    monitor = SafetyMonitor()
    
    # Test speed limiting
    safe_speed, safe_steering = monitor.check_command(1.5, 0.5)
    print(f"Command check: speed={safe_speed}, steering={safe_steering}")
    print(f"Warnings: {monitor.warnings}")
    
    # Test sensor checking
    sensor_data = {
        'ultrasonic_distance': 25.0,
        'battery_voltage': 7.2
    }
    is_safe = monitor.check_sensors(sensor_data)
    print(f"Sensor check: safe={is_safe}")
    print(f"Warnings: {monitor.warnings}")
