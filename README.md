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
