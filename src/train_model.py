"""
Model training script for self-driving car
"""

import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
import json
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cnn_model import build_nvidia_model, build_advanced_model, get_callbacks
from src.utils.data_augmentation import augment_image


class DataGenerator(tf.keras.utils.Sequence):
    """Generates batches of data for training"""
    
    def __init__(self, df, image_dir, batch_size=32, image_size=(120, 160), 
                 augment=True, shuffle=True):
        self.df = df
        self.image_dir = Path(image_dir)
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.iloc[indexes]
        
        X, y = self.__data_generation(batch_df)
        return X, y
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, batch_df):
        X = np.empty((self.batch_size, *self.image_size, 3))
        y = np.empty((self.batch_size,))
        
        for i, (_, row) in enumerate(batch_df.iterrows()):
            # Load image
            img_path = self.image_dir / row['image_file']
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get steering angle
            steering = float(row['steering'])
            
            # Augment if enabled
            if self.augment and np.random.random() > 0.5:
                img, steering = augment_image(img, steering)
            
            # Resize
            img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
            
            X[i] = img
            y[i] = steering
        
        return X, y


def load_data(data_dirs):
    """Load data from multiple directories"""
    all_data = []
    
    for data_dir in data_dirs:
        data_dir = Path(data_dir)
        log_file = data_dir / "driving_log.csv"
        
        if not log_file.exists():
            print(f"Warning: No driving_log.csv found in {data_dir}")
            continue
        
        # Read CSV
        df = pd.read_csv(log_file)
        
        # Add image directory path
        df['image_dir'] = str(data_dir / "images")
        
        all_data.append(df)
        print(f"Loaded {len(df)} samples from {data_dir}")
    
    if not all_data:
        raise ValueError("No data found in specified directories")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total samples: {len(combined_df)}")
    
    return combined_df


def analyze_data(df):
    """Analyze and visualize data distribution"""
    print("\nData Analysis:")
    print(f"Total samples: {len(df)}")
    print(f"Steering range: [{df['steering'].min():.3f}, {df['steering'].max():.3f}]")
    print(f"Steering mean: {df['steering'].mean():.3f}")
    print(f"Steering std: {df['steering'].std():.3f}")
    
    # Plot steering distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['steering'], bins=50, edgecolor='black')
    plt.xlabel('Steering Angle')
    plt.ylabel('Count')
    plt.title('Steering Angle Distribution')
    plt.grid(True, alpha=0.3)
    plt.savefig('steering_distribution.png')
    plt.close()
    
    # Check for imbalanced data
    zero_steering = len(df[abs(df['steering']) < 0.05])
    print(f"Samples with near-zero steering: {zero_steering} ({zero_steering/len(df)*100:.1f}%)")
    
    if zero_steering / len(df) > 0.5:
        print("WARNING: Data is imbalanced with too many straight driving samples!")


def balance_data(df, zero_threshold=0.05, keep_prob=0.3):
    """Balance dataset by reducing straight driving samples"""
    # Separate zero and non-zero steering
    zero_steering = df[abs(df['steering']) < zero_threshold]
    non_zero_steering = df[abs(df['steering']) >= zero_threshold]
    
    # Randomly sample zero steering data
    n_keep = int(len(zero_steering) * keep_prob)
    zero_steering_balanced = zero_steering.sample(n=n_keep, random_state=42)
    
    # Combine
    balanced_df = pd.concat([zero_steering_balanced, non_zero_steering])
    balanced_df = shuffle(balanced_df, random_state=42)
    
    print(f"Balanced dataset: {len(balanced_df)} samples (removed {len(df) - len(balanced_df)} samples)")
    
    return balanced_df


def train_model(data_dirs, model_type='nvidia', epochs=50, batch_size=32, 
                validation_split=0.2, augment=True, balance=True):
    """Train the self-driving model"""
    
    # Load data
    df = load_data(data_dirs)
    
    # Analyze data
    analyze_data(df)
    
    # Balance data if requested
    if balance:
        df = balance_data(df)
    
    # Split data by sessions to avoid leakage
    train_df, val_df = train_test_split(df, test_size=validation_split, random_state=42)
    
    print(f"\nTraining samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Create data generators
    # Use the first data directory's image folder
    image_dir = Path(data_dirs[0]) / "images"
    
    train_gen = DataGenerator(
        train_df, image_dir, batch_size=batch_size, 
        augment=augment, shuffle=True
    )
    
    val_gen = DataGenerator(
        val_df, image_dir, batch_size=batch_size, 
        augment=False, shuffle=False
    )
    
    # Build model
    if model_type == 'nvidia':
        model = build_nvidia_model()
    else:
        model = build_advanced_model()
    
    print(f"\nUsing {model_type} model architecture")
    
    # Get callbacks
    callbacks = get_callbacks('models/best_model.h5')
    
    # Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save('models/final_model.h5')
    
    # Save training history
    with open('models/training_history.json', 'w') as f:
        json.dump(history.history, f)
    
    # Plot training history
    plot_training_history(history)
    
    return model, history


def plot_training_history(history):
    """Plot and save training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Model Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MAE
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title('Model MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    plt.close()
    
    print("Training history saved to models/training_history.png")


def evaluate_model(model_path, test_data_dir):
    """Evaluate trained model on test data"""
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Load test data
    test_df = load_data([test_data_dir])
    
    # Create test generator
    image_dir = Path(test_data_dir) / "images"
    test_gen = DataGenerator(
        test_df, image_dir, batch_size=32, 
        augment=False, shuffle=False
    )
    
    # Evaluate
    results = model.evaluate(test_gen, verbose=1)
    
    print(f"\nTest Results:")
    print(f"Loss: {results[0]:.4f}")
    print(f"MAE: {results[1]:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train self-driving car model")
    parser.add_argument('--data', nargs='+', required=True,
                        help='Training data directories')
    parser.add_argument('--model', choices=['nvidia', 'advanced'], 
                        default='nvidia', help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--no-balance', action='store_true',
                        help='Disable data balancing')
    parser.add_argument('--evaluate', type=str,
                        help='Evaluate model on test data')
    
    args = parser.parse_args()
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    if args.evaluate:
        # Evaluation mode
        evaluate_model('models/best_model.h5', args.evaluate)
    else:
        # Training mode
        train_model(
            args.data,
            model_type=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            augment=not args.no_augment,
            balance=not args.no_balance
        )


if __name__ == '__main__':
    main()
