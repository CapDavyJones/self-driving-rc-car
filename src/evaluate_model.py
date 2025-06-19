"""
Model evaluation and analysis script
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
import argparse
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_augmentation import preprocess_image
from src.train_model import DataGenerator


class ModelEvaluator:
    """Comprehensive model evaluation and analysis"""
    
    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)
        self.results = {}
        
    def evaluate_on_dataset(self, data_dir: str, batch_size: int = 32):
        """Evaluate model on a dataset"""
        # Load data
        data_dir = Path(data_dir)
        log_file = data_dir / "driving_log.csv"
        df = pd.read_csv(log_file)
        
        # Create data generator
        image_dir = data_dir / "images"
        test_gen = DataGenerator(
            df, image_dir, batch_size=batch_size,
            augment=False, shuffle=False
        )
        
        # Get predictions
        print("Running predictions...")
        predictions = []
        ground_truth = []
        
        for i in range(len(test_gen)):
            X_batch, y_batch = test_gen[i]
            pred_batch = self.model.predict(X_batch, verbose=0)
            
            if isinstance(pred_batch, list):
                # Multi-output model
                pred_batch = pred_batch[0]
            
            predictions.extend(pred_batch.flatten())
            ground_truth.extend(y_batch.flatten())
        
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        # Calculate metrics
        self.results['mse'] = mean_squared_error(ground_truth, predictions)
        self.results['mae'] = mean_absolute_error(ground_truth, predictions)
        self.results['rmse'] = np.sqrt(self.results['mse'])
        self.results['r2'] = r2_score(ground_truth, predictions)
        
        # Store for visualization
        self.predictions = predictions
        self.ground_truth = ground_truth
        
        return self.results
    
    def analyze_errors(self):
        """Analyze prediction errors"""
        errors = self.predictions - self.ground_truth
        
        # Error statistics
        error_stats = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(np.abs(errors)),
            'percentile_95': np.percentile(np.abs(errors), 95)
        }
        
        self.results['error_stats'] = error_stats
        
        # Create error distribution plot
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Error histogram
        plt.subplot(2, 2, 1)
        plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(x=0, color='red', linestyle='--', label='Zero error')
        plt.xlabel('Prediction Error')
        plt.ylabel('Count')
        plt.title('Error Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Predictions vs Ground Truth
        plt.subplot(2, 2, 2)
        plt.scatter(self.ground_truth, self.predictions, alpha=0.5, s=10)
        plt.plot([-1, 1], [-1, 1], 'r--', label='Perfect prediction')
        plt.xlabel('Ground Truth Steering')
        plt.ylabel('Predicted Steering')
        plt.title('Predictions vs Ground Truth')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Subplot 3: Error vs Steering Magnitude
        plt.subplot(2, 2, 3)
        steering_magnitude = np.abs(self.ground_truth)
        plt.scatter(steering_magnitude, np.abs(errors), alpha=0.5, s=10)
        plt.xlabel('Steering Magnitude')
        plt.ylabel('Absolute Error')
        plt.title('Error vs Steering Magnitude')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Cumulative error distribution
        plt.subplot(2, 2, 4)
        sorted_errors = np.sort(np.abs(errors))
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        plt.plot(sorted_errors, cumulative)
        plt.xlabel('Absolute Error')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Error Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('error_analysis.png', dpi=150)
        plt.close()
        
        return error_stats
    
    def visualize_predictions(self, data_dir: str, num_samples: int = 20):
        """Visualize model predictions on sample images"""
        # Load data
        data_dir = Path(data_dir)
        log_file = data_dir / "driving_log.csv"
        df = pd.read_csv(log_file)
        
        # Sample random images
        sample_indices = np.random.choice(len(df), num_samples, replace=False)
        
        # Create visualization
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        axes = axes.ravel()
        
        for idx, sample_idx in enumerate(sample_indices):
            row = df.iloc[sample_idx]
            
            # Load and preprocess image
            img_path = data_dir / "images" / row['image_file']
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Preprocess for model
            processed = preprocess_image(img_rgb, target_size=(120, 160), crop=True)
            img_batch = np.expand_dims(processed, axis=0)
            
            # Get prediction
            pred = self.model.predict(img_batch, verbose=0)
            if isinstance(pred, list):
                pred = pred[0]
            pred_steering = pred[0][0]
            
            # Plot
            axes[idx].imshow(img_rgb)
            axes[idx].set_title(f'GT: {row["steering"]:.3f}\nPred: {pred_steering:.3f}')
            axes[idx].axis('off')
            
            # Draw steering visualization
            h, w = img_rgb.shape[:2]
            center_x = w // 2
            
            # Ground truth (green)
            gt_x = int(center_x + row['steering'] * 100)
            axes[idx].arrow(center_x, h-10, gt_x-center_x, -20, 
                           head_width=10, head_length=5, fc='green', ec='green')
            
            # Prediction (red)
            pred_x = int(center_x + pred_steering * 100)
            axes[idx].arrow(center_x, h-15, pred_x-center_x, -20,
                           head_width=10, head_length=5, fc='red', ec='red')
        
        plt.suptitle('Model Predictions (Green: GT, Red: Prediction)', fontsize=16)
        plt.tight_layout()
        plt.savefig('prediction_samples.png', dpi=150)
        plt.close()
    
    def analyze_model_behavior(self):
        """Analyze model behavior patterns"""
        # Steering distribution comparison
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(self.ground_truth, bins=50, alpha=0.5, label='Ground Truth', density=True)
        plt.hist(self.predictions, bins=50, alpha=0.5, label='Predictions', density=True)
        plt.xlabel('Steering Angle')
        plt.ylabel('Density')
        plt.title('Steering Angle Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Analyze turns vs straight
        plt.subplot(1, 2, 2)
        straight_mask = np.abs(self.ground_truth) < 0.1
        turn_mask = ~straight_mask
        
        straight_error = mean_absolute_error(
            self.ground_truth[straight_mask], 
            self.predictions[straight_mask]
        )
        turn_error = mean_absolute_error(
            self.ground_truth[turn_mask],
            self.predictions[turn_mask]
        )
        
        categories = ['Straight', 'Turning']
        errors = [straight_error, turn_error]
        
        plt.bar(categories, errors)
        plt.ylabel('Mean Absolute Error')
        plt.title('Error by Driving Scenario')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('behavior_analysis.png', dpi=150)
        plt.close()
        
        return {
            'straight_driving_error': straight_error,
            'turning_error': turn_error
        }
    
    def test_model_robustness(self, test_image_path: str):
        """Test model robustness to various conditions"""
        # Load test image
        img = cv2.imread(test_image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Test different conditions
        conditions = []
        
        # Original
        processed = preprocess_image(img_rgb, target_size=(120, 160), crop=True)
        pred = self.model.predict(np.expand_dims(processed, axis=0), verbose=0)
        if isinstance(pred, list):
            pred = pred[0]
        conditions.append(('Original', processed, pred[0][0]))
        
        # Brightness variations
        for brightness in [0.5, 1.5]:
            bright_img = cv2.convertScaleAbs(img_rgb, alpha=brightness, beta=0)
            processed = preprocess_image(bright_img, target_size=(120, 160), crop=True)
            pred = self.model.predict(np.expand_dims(processed, axis=0), verbose=0)
            if isinstance(pred, list):
                pred = pred[0]
            conditions.append((f'Brightness {brightness}x', processed, pred[0][0]))
        
        # Blur
        blurred = cv2.GaussianBlur(img_rgb, (5, 5), 0)
        processed = preprocess_image(blurred, target_size=(120, 160), crop=True)
        pred = self.model.predict(np.expand_dims(processed, axis=0), verbose=0)
        if isinstance(pred, list):
            pred = pred[0]
        conditions.append(('Blurred', processed, pred[0][0]))
        
        # Noise
        noise = np.random.normal(0, 20, img_rgb.shape).astype(np.uint8)
        noisy = cv2.add(img_rgb, noise)
        processed = preprocess_image(noisy, target_size=(120, 160), crop=True)
        pred = self.model.predict(np.expand_dims(processed, axis=0), verbose=0)
        if isinstance(pred, list):
            pred = pred[0]
        conditions.append(('Noisy', processed, pred[0][0]))
        
        # Visualize
        fig, axes = plt.subplots(1, len(conditions), figsize=(20, 4))
        
        for idx, (name, img, pred) in enumerate(conditions):
            axes[idx].imshow(img)
            axes[idx].set_title(f'{name}\nPred: {pred:.3f}')
            axes[idx].axis('off')
        
        plt.suptitle('Model Robustness Test', fontsize=16)
        plt.tight_layout()
        plt.savefig('robustness_test.png', dpi=150)
        plt.close()
        
        # Calculate variance
        predictions = [c[2] for c in conditions]
        robustness_score = 1.0 / (1.0 + np.std(predictions))
        
        return {
            'robustness_score': robustness_score,
            'prediction_std': np.std(predictions),
            'predictions': {c[0]: c[2] for c in conditions}
        }
    
    def generate_report(self, output_path: str = 'evaluation_report.txt'):
        """Generate comprehensive evaluation report"""
        with open(output_path, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Model metrics
            f.write("Performance Metrics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"MSE:  {self.results.get('mse', 'N/A'):.6f}\n")
            f.write(f"MAE:  {self.results.get('mae', 'N/A'):.6f}\n")
            f.write(f"RMSE: {self.results.get('rmse', 'N/A'):.6f}\n")
            f.write(f"R²:   {self.results.get('r2', 'N/A'):.6f}\n\n")
            
            # Error statistics
            if 'error_stats' in self.results:
                f.write("Error Statistics:\n")
                f.write("-" * 20 + "\n")
                stats = self.results['error_stats']
                f.write(f"Mean Error:     {stats['mean_error']:.6f}\n")
                f.write(f"Std Error:      {stats['std_error']:.6f}\n")
                f.write(f"Max Error:      {stats['max_error']:.6f}\n")
                f.write(f"95% Percentile: {stats['percentile_95']:.6f}\n\n")
            
            # Save as JSON for programmatic access
            json_results = {k: float(v) if isinstance(v, np.number) else v 
                          for k, v in self.results.items()}
            with open('evaluation_results.json', 'w') as json_f:
                json.dump(json_results, json_f, indent=2)
        
        print(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument('--model', '-m', required=True,
                        help='Path to trained model')
    parser.add_argument('--data', '-d', required=True,
                        help='Test data directory')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--test-image', type=str,
                        help='Test image for robustness analysis')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(args.model)
    
    # Run evaluation
    print("Evaluating model...")
    results = evaluator.evaluate_on_dataset(args.data, args.batch_size)
    
    print("\nResults:")
    print(f"MSE:  {results['mse']:.6f}")
    print(f"MAE:  {results['mae']:.6f}")
    print(f"RMSE: {results['rmse']:.6f}")
    print(f"R²:   {results['r2']:.6f}")
    
    # Analyze errors
    print("\nAnalyzing errors...")
    error_stats = evaluator.analyze_errors()
    
    # Generate visualizations if requested
    if args.visualize:
        print("\nGenerating visualizations...")
        evaluator.visualize_predictions(args.data)
        evaluator.analyze_model_behavior()
        
        if args.test_image:
            print("Testing model robustness...")
            robustness = evaluator.test_model_robustness(args.test_image)
            print(f"Robustness score: {robustness['robustness_score']:.3f}")
    
    # Generate report
    evaluator.generate_report()
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
