"""
Motor Control Module for Self-Driving RC Car
Handles low-level motor commands with safety features
"""

import RPi.GPIO as GPIO
import time
import threading
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MotorController:
    """
    Motor controller with PWM speed control and safety features
    """
    
    def __init__(self):
        # GPIO pins (BCM numbering)
        self.LEFT_FORWARD = 17
        self.LEFT_BACKWARD = 18
        self.RIGHT_FORWARD = 22
        self.RIGHT_BACKWARD = 23
        
        # PWM settings
        self.PWM_FREQ = 1000
        self.current_speed = 0
        self.current_steering = 0
        
        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Setup pins
        pins = [self.LEFT_FORWARD, self.LEFT_BACKWARD, 
                self.RIGHT_FORWARD, self.RIGHT_BACKWARD]
        for pin in pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
        
        # Setup PWM
        self.left_pwm = GPIO.PWM(self.LEFT_FORWARD, self.PWM_FREQ)
        self.right_pwm = GPIO.PWM(self.RIGHT_FORWARD, self.PWM_FREQ)
        self.left_pwm.start(0)
        self.right_pwm.start(0)
        
        logger.info("Motor controller initialized")
    
    def set_speed_steering(self, speed: float, steering: float):
        """
        Set speed and steering
        speed: -1 to 1 (negative = backward)
        steering: -1 to 1 (negative = left)
        """
        # Clamp values
        speed = max(-1, min(1, speed))
        steering = max(-1, min(1, steering))
        
        # Calculate motor speeds
        left_speed = speed * (1 - steering)
        right_speed = speed * (1 + steering)
        
        # Apply to motors
        self._set_motor('left', left_speed)
        self._set_motor('right', right_speed)
        
        self.current_speed = speed
        self.current_steering = steering
    
    def _set_motor(self, side: str, speed: float):
        """Set individual motor speed"""
        if side == 'left':
            forward_pin = self.LEFT_FORWARD
            backward_pin = self.LEFT_BACKWARD
            pwm = self.left_pwm
        else:
            forward_pin = self.RIGHT_FORWARD
            backward_pin = self.RIGHT_BACKWARD
            pwm = self.right_pwm
        
        # Convert to PWM duty cycle
        duty = abs(speed) * 100
        duty = max(0, min(100, duty))
        
        if speed > 0.05:  # Forward
            GPIO.output(backward_pin, GPIO.LOW)
            pwm.ChangeDutyCycle(duty)
        elif speed < -0.05:  # Backward
            GPIO.output(forward_pin, GPIO.LOW)
            GPIO.output(backward_pin, GPIO.HIGH)
        else:  # Stop
            GPIO.output(forward_pin, GPIO.LOW)
            GPIO.output(backward_pin, GPIO.LOW)
            pwm.ChangeDutyCycle(0)
    
    def stop(self):
        """Stop all motors"""
        self.set_speed_steering(0, 0)
    
    def cleanup(self):
        """Clean up GPIO resources"""
        self.stop()
        self.left_pwm.stop()
        self.right_pwm.stop()
        GPIO.cleanup()
        logger.info("Motor controller cleaned up")


# Simple interface functions
def move_forward(duration=0.5):
    """Move forward for duration seconds"""
    controller = MotorController()
    controller.set_speed_steering(0.5, 0)
    time.sleep(duration)
    controller.cleanup()


def move_left(duration=0.5):
    """Turn left for duration seconds"""
    controller = MotorController()
    controller.set_speed_steering(0.5, -0.5)
    time.sleep(duration)
    controller.cleanup()


def move_right(duration=0.5):
    """Turn right for duration seconds"""
    controller = MotorController()
    controller.set_speed_steering(0.5, 0.5)
    time.sleep(duration)
    controller.cleanup()


def stop():
    """Stop motors"""
    controller = MotorController()
    controller.stop()
    controller.cleanup()
