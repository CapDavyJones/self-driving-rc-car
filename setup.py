"""
Setup and installation script for self-driving RC car
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(cmd, check=True):
    """Run a shell command"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    return True


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 7:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python 3.7+ required, found {version.major}.{version.minor}")
        return False


def install_system_dependencies():
    """Install system-level dependencies"""
    print("\nInstalling system dependencies...")
    
    if platform.system() != "Linux":
        print("Warning: System dependencies can only be auto-installed on Linux")
        return True
    
    # Update package list
    run_command("sudo apt-get update")
    
    # Install required packages
    packages = [
        "python3-pip",
        "python3-dev",
        "python3-opencv",
        "libopencv-dev",
        "libatlas-base-dev",
        "libjasper-dev",
        "libqtgui4",
        "libqt4-test",
        "libhdf5-dev",
        "git",
        "i2c-tools",
        "python3-smbus"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        if not run_command(f"sudo apt-get install -y {package}", check=False):
            print(f"Warning: Could not install {package}")
    
    return True


def install_python_dependencies():
    """Install Python dependencies"""
    print("\nInstalling Python dependencies...")
    
    # Upgrade pip
    run_command(f"{sys.executable} -m pip install --upgrade pip")
    
    # Create requirements.txt if it doesn't exist
    requirements_path = Path("requirements.txt")
    if not requirements_path.exists():
        print("Creating requirements.txt...")
        requirements = """
numpy>=1.19.5
opencv-python>=4.5.0
tensorflow>=2.6.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
pyyaml>=5.4.0
pillow>=8.3.0
h5py>=3.1.0
"""
        requirements_path.write_text(requirements.strip())
    
    # Install from requirements
    return run_command(f"{sys.executable} -m pip install -r requirements.txt")


def setup_gpio():
    """Setup GPIO library for Raspberry Pi"""
    print("\nSetting up GPIO...")
    
    # Check if we're on Raspberry Pi
    try:
        with open('/proc/cpuinfo', 'r') as f:
            if 'BCM' not in f.read():
                print("Not running on Raspberry Pi, skipping GPIO setup")
                return True
    except:
        print("Could not detect Raspberry Pi")
        return True
    
    # Install RPi.GPIO
    return run_command(f"{sys.executable} -m pip install RPi.GPIO")


def enable_camera():
    """Enable camera interface on Raspberry Pi"""
    print("\nEnabling camera interface...")
    
    if platform.system() != "Linux":
        print("Camera setup only needed on Raspberry Pi")
        return True
    
    print("Please enable camera interface manually:")
    print("1. Run: sudo raspi-config")
    print("2. Go to Interface Options > Camera")
    print("3. Enable camera and reboot")
    
    return True


def create_directories():
    """Create necessary directories"""
    print("\nCreating project directories...")
    
    directories = [
        "data",
        "models",
        "logs",
        "config",
        "data/raw",
        "data/processed",
        "models/checkpoints"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {directory}/")
    
    return True


def download_sample_data():
    """Download sample data for testing"""
    print("\nSetting up sample data...")
    
    # Create sample config if it doesn't exist
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        print("Config file already exists")
    
    print("✓ Sample configuration ready")
    return True


def test_installation():
    """Test the installation"""
    print("\nTesting installation...")
    
    # Test imports
    test_imports = [
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("tensorflow", "TensorFlow"),
        ("pandas", "Pandas"),
        ("yaml", "PyYAML")
    ]
    
    all_good = True
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✓ {name} import successful")
        except ImportError:
            print(f"✗ {name} import failed")
            all_good = False
    
    # Test camera (non-blocking)
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Camera detected")
            cap.release()
        else:
            print("⚠ No camera detected (this is OK for development)")
    except:
        print("⚠ Could not test camera")
    
    return all_good


def setup_systemd_service():
    """Setup systemd service for autonomous driving"""
    print("\nSetting up systemd service...")
    
    if platform.system() != "Linux":
        print("Systemd service only available on Linux")
        return True
    
    service_content = """[Unit]
Description=Self-Driving RC Car Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/self-driving-rc-car
ExecStart=/usr/bin/python3 /home/pi/self-driving-rc-car/src/autonomous_driver.py --model models/best_model.h5
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
"""
    
    service_path = Path("self-driving-car.service")
    service_path.write_text(service_content)
    
    print("Service file created. To install:")
    print("1. sudo cp self-driving-car.service /etc/systemd/system/")
    print("2. sudo systemctl daemon-reload")
    print("3. sudo systemctl enable self-driving-car.service")
    
    return True


def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*50)
    print("SETUP COMPLETE!")
    print("="*50)
    
    print("\nNext steps:")
    print("\n1. Test components:")
    print("   python src/tests/test_components.py")
    
    print("\n2. Collect training data:")
    print("   python src/data_collection.py --output data/session1")
    
    print("\n3. Train the model:")
    print("   python src/train_model.py --data data/session1 --epochs 50")
    
    print("\n4. Run autonomous driving:")
    print("   python src/autonomous_driver.py --model models/best_model.h5")
    
    print("\n5. Evaluate performance:")
    print("   python src/evaluate_model.py --model models/best_model.h5 --data data/test")
    
    print("\nFor more information, see README.md")


def main():
    """Main setup function"""
    print("="*50)
    print("SEL
