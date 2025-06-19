#!/usr/bin/env python3
"""
Utility script for common self-driving car tasks
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import shutil
import yaml
import cv2
import numpy as np


def create_video_from_session(session_dir: str, output_path: str = None):
    """Create a video from collected session data"""
    session_path = Path(session_dir)
    log_file = session_path / "driving_log.csv"
    
    if not log_file.exists():
        print(f"Error: No driving log found in {session_dir}")
        return
    
    # Read log file
    import pandas as pd
    df = pd.read_csv(log_file)
    
    # Setup video writer
    if output_path is None:
        output_path = f"{session_path.name}_video.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))
    
    print(f"Creating video from {len(df)} frames...")
    
    for idx, row in df.iterrows():
        img_path = session_path / "images" / row['image_file']
        if img_path.exists():
            img = cv2.imread(str(img_path))
            
            # Add steering visualization
            h, w = img.shape[:2]
            center_x = w // 2
            steering_x = int(center_x + row['steering'] * 200)
            
            # Draw steering line
            cv2.line(img, (center_x, h-50), (steering_x, h-100), (0, 255, 0), 3)
            
            # Add text
            cv2.putText(img, f"Steering: {row['steering']:.3f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"Speed: {row['speed']:.3f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            out.write(img)
        
        if idx % 100 == 0:
            print(f"  Processed {idx}/{len(df)} frames...")
    
    out.release()
    print(f"Video saved to: {output_path}")


def backup_model(model_path: str, backup_name: str = None):
    """Backup a trained model"""
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"Error: Model not found: {model_path}")
        return
    
    # Create backup directory
    backup_dir = Path("models/backups")
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate backup name
    if backup_name is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{model_file.stem}_{timestamp}{model_file.suffix}"
    
    backup_path = backup_dir / backup_name
    shutil.copy2(model_file, backup_path)
    print(f"Model backed up to: {backup_path}")


def merge_datasets(dataset_dirs: list, output_dir: str):
    """Merge multiple training datasets"""
    import pandas as pd
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create images directory
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)
    
    all_data = []
    image_counter = 0
    
    for dataset_dir in dataset_dirs:
        dataset_path = Path(dataset_dir)
        log_file = dataset_path / "driving_log.csv"
        
        if not log_file.exists():
            print(f"Warning: No log file in {dataset_dir}")
            continue
        
        df = pd.read_csv(log_file)
        print(f"Processing {dataset_dir}: {len(df)} samples")
        
        # Copy images and update paths
        for idx, row in df.iterrows():
            old_image_path = dataset_path / "images" / row['image_file']
            if old_image_path.exists():
                # Create new filename
                new_filename = f"img_{image_counter:06d}.jpg"
                new_image_path = images_dir / new_filename
                
                # Copy image
                shutil.copy2(old_image_path, new_image_path)
                
                # Update row
                row['image_file'] = new_filename
                all_data.append(row)
                image_counter += 1
    
    # Create merged dataframe
    merged_df = pd.DataFrame(all_data)
    merged_df.to_csv(output_path / "driving_log.csv", index=False)
    
    print(f"\nMerged {len(all_data)} samples into {output_dir}")
    print(f"Total images: {image_counter}")


def clean_dataset(dataset_dir: str, min_steering: float = 0.02):
    """Remove low-steering samples to balance dataset"""
    import pandas as pd
    
    dataset_path = Path(dataset_dir)
    log_file = dataset_path / "driving_log.csv"
    
    df = pd.read_csv(log_file)
    original_count = len(df)
    
    # Separate straight and turning samples
    straight_mask = np.abs(df['steering']) < min_steering
    straight_samples = df[straight_mask]
    turning_samples = df[~straight_mask]
    
    print(f"Original dataset: {original_count} samples")
    print(f"Straight: {len(straight_samples)}, Turning: {len(turning_samples)}")
    
    # Keep only 50% of straight samples
    keep_straight = straight_samples.sample(frac=0.5)
    
    # Combine
    balanced_df = pd.concat([keep_straight, turning_samples])
    balanced_df = balanced_df.sort_index()
    
    # Save backup
    backup_file = log_file.with_suffix('.csv.backup')
    shutil.copy2(log_file, backup_file)
    
    # Save cleaned dataset
    balanced_df.to_csv(log_file, index=False)
    
    print(f"Cleaned dataset: {len(balanced_df)} samples")
    print(f"Removed {original_count - len(balanced_df)} samples")
    print(f"Backup saved to: {backup_file}")


def visualize_model_architecture(model_path: str):
    """Visualize model architecture"""
    import tensorflow as tf
    from tensorflow.keras.utils import plot_model
    
    model = tf.keras.models.load_model(model_path)
    output_path = Path(model_path).stem + "_architecture.png"
    
    plot_model(model, to_file=output_path, show_shapes=True, 
               show_layer_names=True, dpi=96)
    
    print(f"Model architecture saved to: {output_path}")
    
    # Also print summary
    print("\nModel Summary:")
    model.summary()


def calibrate_steering(port: str = '/dev/ttyUSB0'):
    """Interactive steering calibration tool"""
    print("Steering Calibration Tool")
    print("=" * 30)
    print("Instructions:")
    print("- Use LEFT/RIGHT arrows to adjust steering")
    print("- Press SPACE when wheels are straight")
    print("- Press Q to quit")
    
    try:
        from src.core.motor_control import MotorController
        motor = MotorController()
        
        current_steering = 0.0
        step = 0.05
        
        import cv2
        
        while True:
            # Create visualization
            img = np.ones((200, 400, 3), dtype=np.uint8) * 255
            cv2.putText(img, f"Steering: {current_steering:.2f}", 
                       (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            # Draw steering indicator
            center_x = 200
            indicator_x = int(center_x + current_steering * 150)
            cv2.line(img, (center_x, 150), (indicator_x, 120), (0, 255, 0), 3)
            
            cv2.imshow('Steering Calibration', img)
            
            # Apply steering
            motor.set_speed_steering(0, current_steering)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 81:  # Left arrow
                current_steering = max(-1.0, current_steering - step)
            elif key == 83:  # Right arrow
                current_steering = min(1.0, current_steering + step)
            elif key == ord(' '):
                print(f"Center steering offset: {current_steering}")
                # Save to config
                save_calibration({'steering_offset': current_steering})
                break
        
        motor.stop()
        motor.cleanup()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {e}")


def save_calibration(calibration_data: dict):
    """Save calibration data to config"""
    config_path = Path("config/config.yaml")
    
    # Load existing config
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Update calibration
    if 'calibration' not in config:
        config['calibration'] = {}
    
    config['calibration'].update(calibration_data)
    
    # Save
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Calibration saved to {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Self-driving car utilities")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create video command
    video_parser = subparsers.add_parser('create-video', 
                                        help='Create video from session data')
    video_parser.add_argument('session', help='Session directory')
    video_parser.add_argument('--output', '-o', help='Output video path')
    
    # Backup model command
    backup_parser = subparsers.add_parser('backup-model', 
                                         help='Backup a trained model')
    backup_parser.add_argument('model', help='Model file path')
    backup_parser.add_argument('--name', '-n', help='Backup name')
    
    # Merge datasets command
    merge_parser = subparsers.add_parser('merge-data', 
                                        help='Merge multiple datasets')
    merge_parser.add_argument('datasets', nargs='+', help='Dataset directories')
    merge_parser.add_argument('--output', '-o', required=True, 
                             help='Output directory')
    
    # Clean dataset command
    clean_parser = subparsers.add_parser('clean-data', 
                                        help='Balance dataset by removing straight samples')
    clean_parser.add_argument('dataset', help='Dataset directory')
    clean_parser.add_argument('--min-steering', type=float, default=0.02,
                             help='Minimum steering threshold')
    
    # Visualize model command
    viz_parser = subparsers.add_parser('visualize-model', 
                                      help='Visualize model architecture')
    viz_parser.add_argument('model', help='Model file path')
    
    # Calibrate steering command
    cal_parser = subparsers.add_parser('calibrate', 
                                      help='Calibrate steering center')
    
    args = parser.parse_args()
    
    if args.command == 'create-video':
        create_video_from_session(args.session, args.output)
    elif args.command == 'backup-model':
        backup_model(args.model, args.name)
    elif args.command == 'merge-data':
        merge_datasets(args.datasets, args.output)
    elif args.command == 'clean-data':
        clean_dataset(args.dataset, args.min_steering)
    elif args.command == 'visualize-model':
        visualize_model_architecture(args.model)
    elif args.command == 'calibrate':
        calibrate_steering()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
