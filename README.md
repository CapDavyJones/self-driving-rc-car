# Self-Driving RC Car

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Build your own autonomous RC car using computer vision and deep learning!
By Amer Blboheath

## Project Overview

This project implements a fully autonomous RC car system with:
- Computer Vision based navigation using CNNs
- Sensor Fusion for robust state estimation
- Path Planning with obstacle avoidance
- Real-time Control with safety monitoring

## Hardware Requirements

- Raspberry Pi 4B (4GB+)
- RC Car Chassis (1:10 scale)
- Pi Camera v2 or USB Webcam
- L298N Motor Driver
- Power supplies (5V for Pi, 7.4V for motors)

## Installation

1. Clone the repository
2. Install dependencies: pip install -r requirements.txt

## Quick Start

1. Collect training data: python src/data_collection.py
2. Train the model: python src/train_model.py
3. Run autonomous mode: python src/autonomous_drive.py

## License

This project is licensed under the MIT License - see the LICENSE file for details.
Sure! Let me create a comprehensive README in a single, easy-to-copy format:

```markdown
# Self-Driving RC Car

An autonomous RC car powered by computer vision and deep learning, built with Raspberry Pi and TensorFlow.

## 🚗 Overview

This project transforms a standard RC car into an autonomous vehicle capable of:
- Lane following and navigation
- Obstacle detection and avoidance
- Real-time decision making using CNN
- Manual override for safety

## 📋 Features

- **Deep Learning**: NVIDIA-inspired CNN architecture for end-to-end learning
- **Real-time Processing**: 30+ FPS inference on Raspberry Pi
- **Safety First**: Multiple safety layers including emergency stop
- **Data Collection**: Easy-to-use manual driving mode for collecting training data
- **Modular Design**: Clean, extensible architecture

## 🛠️ Hardware Requirements

- Raspberry Pi 4 (4GB+ recommended)
- RC Car chassis with DC motors
- Raspberry Pi Camera Module v2
- L298N Motor Driver
- Ultrasonic sensor (HC-SR04)
- Power supply (7.4V LiPo battery recommended)
- SD Card (32GB+ recommended)

### Wiring Diagram

```
Raspberry Pi GPIO → L298N Motor Driver
- GPIO 17 → IN1 (Motor A)
- GPIO 27 → IN2 (Motor A)
- GPIO 22 → IN3 (Motor B)
- GPIO 23 → IN4 (Motor B)
- GPIO 24 → ENA (PWM Speed A)
- GPIO 25 → ENB (PWM Speed B)

Ultrasonic Sensor → Raspberry Pi
- VCC → 5V
- GND → GND
- Trig → GPIO 18
- Echo → GPIO 15 (through voltage divider)

Camera → CSI Port
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/self-driving-rc-car.git
cd self-driving-rc-car

# Run setup script
python setup.py
```

### 2. Test Components

```bash
python src/tests/test_components.py
```

### 3. Collect Training Data

```bash
# Start data collection with manual control
python src/data_collection.py --output data/session1

# Controls:
# W/S - Forward/Backward
# A/D - Left/Right
# SPACE - Stop
# R - Toggle recording
# Q - Quit
```

### 4. Train the Model

```bash
# Train with collected data
python src/train_model.py --data data/session1 --epochs 50

# Train with multiple sessions
python src/train_model.py --data data/session1 data/session2 --epochs 100
```

### 5. Run Autonomous Mode

```bash
# Run with trained model
python src/autonomous_driver.py --model models/best_model.h5

# With custom max speed
python src/autonomous_driver.py --model models/best_model.h5 --max-speed 0.7
```

## 📁 Project Structure

```
self-driving-rc-car/
├── config/
│   └── config.yaml          # Configuration file
├── src/
│   ├── core/               # Core modules
│   │   ├── motor_control.py
│   │   ├── perception.py
│   │   ├── safety_monitor.py
│   │   └── sensor_fusion.py
│   ├── models/             # Neural network models
│   │   └── cnn_model.py
│   ├── utils/              # Utility functions
│   │   └── data_augmentation.py
│   ├── tests/              # Test scripts
│   ├── data_collection.py  # Data collection script
│   ├── train_model.py      # Training script
│   ├── autonomous_driver.py # Autonomous driving
│   └── evaluate_model.py   # Model evaluation
├── data/                   # Training data
├── models/                 # Saved models
├── setup.py               # Installation script
└── README.md
```

## 🧠 Model Architecture

The CNN architecture is inspired by NVIDIA's end-to-end learning approach:

1. **Input**: 160x120x3 RGB images
2. **Normalization**: Pixel values normalized to [-0.5, 0.5]
3. **Convolutional Layers**: 5 layers with feature extraction
4. **Fully Connected Layers**: 4 layers for decision making
5. **Output**: Steering angle (-1 to 1)

## 📊 Training

### Data Augmentation

The training pipeline includes several augmentation techniques:
- Horizontal flipping
- Brightness adjustment
- Shadow simulation
- Random translation
- Gaussian noise

### Model Evaluation

```bash
# Evaluate on test data
python src/evaluate_model.py --model models/best_model.h5 --data data/test --visualize
```

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

```yaml
# Motor control settings
motor:
  max_pwm: 100
  min_pwm: 20
  
# Camera settings
camera:
  resolution: [640, 480]
  fps: 30
  
# Autonomous driving settings
autonomous:
  max_speed: 0.6
  steering_smoothing: 0.7
```

## 🛡️ Safety Features

1. **Emergency Stop**: Press SPACE to immediately stop
2. **Obstacle Detection**: Automatic stop when obstacles detected
3. **Speed Limiting**: Configurable maximum speed
4. **Manual Override**: Press 'M' to take manual control
5. **Boundary Checking**: Prevents unsafe commands

## 🔧 Troubleshooting

### Camera not detected
```bash
# Enable camera interface
sudo raspi-config
# Navigate to Interface Options > Camera > Enable
```

### GPIO permission errors
```bash
# Add user to gpio group
sudo usermod -a -G gpio $USER
# Logout and login again
```

### Low FPS
- Reduce camera resolution in config
- Use lighter model architecture
- Ensure adequate cooling for Pi

## 📈 Performance Tips

1. **Data Collection**:
   - Collect diverse data (different lighting, surfaces)
   - Aim for 10,000+ images
   - Balance turning vs straight driving

2. **Training**:
   - Use data augmentation
   - Monitor validation loss
   - Experiment with learning rates

3. **Deployment**:
   - Use TensorFlow Lite for faster inference
   - Enable GPU acceleration if available
   - Optimize camera settings

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- NVIDIA for the end-to-end learning approach
- TensorFlow team for the excellent framework
- Raspberry Pi Foundation for the hardware platform

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Read the documentation thoroughly

---

**Happy Autonomous Driving! 🚗🤖**
```
