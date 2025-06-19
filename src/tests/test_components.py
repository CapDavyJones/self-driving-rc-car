"""
Test suite for verifying all components are working correctly
"""

import sys
import os
import time
import numpy as np
import cv2
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.motor_control import MotorController
from src.core.perception import PerceptionModule
from src.core.safety_monitor import SafetyMonitor, SafetyLimits
from src.core.sensor_fusion import SensorFusion


class ComponentTester:
    """Test individual components of the self-driving system"""
    
    def __init__(self):
        self.passed_tests = []
        self.failed_tests = []
    
    def run_all_tests(self):
        """Run all component tests"""
        print("=" * 50)
        print("SELF-DRIVING RC CAR COMPONENT TESTS")
        print("=" * 50)
        
        # Test each component
        self.test_gpio_setup()
        self.test_motor_control()
        self.test_camera()
        self.test_safety_monitor()
        self.test_sensor_fusion()
        
        # Summary
        self.print_summary()
    
    def test_gpio_setup(self):
        """Test GPIO availability"""
        print("\n1. Testing GPIO Setup...")
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            print("✓ GPIO module loaded successfully")
            self.passed_tests.append("GPIO Setup")
        except Exception as e:
            print(f"✗ GPIO test failed: {e}")
            print("  Note: This is expected if not running on Raspberry Pi")
            self.failed_tests.append("GPIO Setup")
    
    def test_motor_control(self):
        """Test motor control system"""
        print("\n2. Testing Motor Control...")
        try:
            motor = MotorController()
            
            # Test initialization
            print("✓ Motor controller initialized")
            
            # Test speed/steering commands
            print("  Testing forward motion...")
            motor.set_speed_steering(0.3, 0.0)
            time.sleep(0.5)
            motor.stop()
            print("✓ Forward motion test complete")
            
            print("  Testing steering...")
            motor.set_speed_steering(0.0, 0.5)
            time.sleep(0.5)
            motor.set_speed_steering(0.0, -0.5)
            time.sleep(0.5)
            motor.stop()
            print("✓ Steering test complete")
            
            motor.cleanup()
            print("✓ Motor control tests passed")
            self.passed_tests.append("Motor Control")
            
        except Exception as e:
            print(f"✗ Motor control test failed: {e}")
            self.failed_tests.append("Motor Control")
    
    def test_camera(self):
        """Test camera system"""
        print("\n3. Testing Camera...")
        try:
            perception = PerceptionModule()
            perception.start()
            
            # Wait for camera to initialize
            time.sleep(2)
            
            # Try to get frames
            success_count = 0
            for i in range(5):
                frame = perception.get_frame()
                if frame is not None:
                    success_count += 1
                    h, w, c = frame.shape
                    print(f"  Frame {i+1}: {w}x{h}, {c} channels")
                time.sleep(0.2)
            
            perception.stop()
            
            if success_count > 0:
                print(f"✓ Camera test passed ({success_count}/5 frames captured)")
                self.passed_tests.append("Camera")
            else:
                print("✗ No frames captured")
                self.failed_tests.append("Camera")
                
        except Exception as e:
            print(f"✗ Camera test failed: {e}")
            self.failed_tests.append("Camera")
    
    def test_safety_monitor(self):
        """Test safety monitoring system"""
        print("\n4. Testing Safety Monitor...")
        try:
            # Create safety monitor with test limits
            limits = SafetyLimits(
                max_speed=0.8,
                max_steering=0.9,
                min_obstacle_distance=30.0
            )
            safety = SafetyMonitor(limits)
            
            # Test 1: Speed limiting
            safe_speed, safe_steering = safety.check_command(1.5, 0.5)
            assert safe_speed == 0.8, f"Speed limiting failed: {safe_speed}"
            print("✓ Speed limiting working")
            
            # Test 2: Obstacle detection
            sensor_data = {'ultrasonic_distance': 10.0}
            is_safe = safety.check_sensors(sensor_data)
            assert not is_safe, "Emergency stop should trigger"
            assert safety.emergency_stop, "Emergency stop not activated"
            print("✓ Obstacle detection working")
            
            # Test 3: Reset
            safety.reset_emergency_stop()
            assert not safety.emergency_stop, "Emergency stop not reset"
            print("✓ Emergency stop reset working")
            
            print("✓ Safety monitor tests passed")
            self.passed_tests.append("Safety Monitor")
            
        except Exception as e:
            print(f"✗ Safety monitor test failed: {e}")
            self.failed_tests.append("Safety Monitor")
    
    def test_sensor_fusion(self):
        """Test sensor fusion system"""
        print("\n5. Testing Sensor Fusion...")
        try:
            fusion = SensorFusion()
            
            # Test IMU update
            fusion.update_imu(
                accel=(0.1, 0.0, 9.8),
                gyro=(0.0, 0.0, 0.1)
            )
            
            # Test encoder update
            fusion.update_encoders(100, 100)
            time.sleep(0.1)
            fusion.update_encoders(110, 110)
            
            # Get state
            state = fusion.get_state()
            
            # Verify state structure
            required_keys = ['position', 'velocity', 'heading', 'angular_velocity']
            for key in required_keys:
                assert key in state, f"Missing state key: {key}"
            
            print(f"✓ Fusion state: pos={state['position']}, heading={state['heading']:.2f}")
            print("✓ Sensor fusion tests passed")
            self.passed_tests.append("Sensor Fusion")
            
        except Exception as e:
            print(f"✗ Sensor fusion test failed: {e}")
            self.failed_tests.append("Sensor Fusion")
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.passed_tests) + len(self.failed_tests)
        
        print(f"\nPassed: {len(self.passed_tests)}/{total_tests}")
        for test in self.passed_tests:
            print(f"  ✓ {test}")
        
        if self.failed_tests:
            print(f"\nFailed: {len(self.failed_tests)}/{total_tests}")
            for test in self.failed_tests:
                print(f"  ✗ {test}")
        
        print("\n" + "=" * 50)
        
        if not self.failed_tests:
            print("All tests passed! System is ready.")
        else:
            print("Some tests failed. Please check the components.")


def test_model_loading():
    """Test model loading and inference"""
    print("\n6. Testing Model Loading...")
    try:
        import tensorflow as tf
        
        # Check if model exists
        model_path = Path("models/best_model.h5")
        if not model_path.exists():
            print("  No trained model found. Train a model first.")
            return False
        
        # Load model
        model = tf.keras.models.load_model(str(model_path))
        print("✓ Model loaded successfully")
        
        # Test inference
        dummy_input = np.random.rand(1, 120, 160, 3).astype(np.float32)
        prediction = model.predict(dummy_input, verbose=0)
        
        print(f"✓ Model inference working. Output shape: {prediction.shape}")
        return True
        
    except Exception as e:
        print(f"✗ Model loading test failed: {e}")
        return False


def check_dependencies():
    """Check if all required dependencies are installed"""
    print("\nChecking Dependencies...")
    
    dependencies = {
        'numpy': 'NumPy',
        'cv2': 'OpenCV',
        'tensorflow': 'TensorFlow',
        'pandas': 'Pandas',
        'yaml': 'PyYAML',
        'sklearn': 'Scikit-learn'
    }
    
    missing = []
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - Not installed")
            missing.append(name)
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True


def main():
    """Main test runner"""
    print("Starting component tests...\n")
    
    # Check dependencies first
    if not check_dependencies():
        print("\nPlease install missing dependencies before running tests.")
        return
    
    # Run component tests
    tester = ComponentTester()
    tester.run_all_tests()
    
    # Test model if available
    test_model_loading()
    
    print("\nTesting complete!")


if __name__ == '__main__':
    main()
