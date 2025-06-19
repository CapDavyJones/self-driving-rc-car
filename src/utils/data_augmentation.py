"""
Data augmentation utilities for training robust models
"""

import cv2
import numpy as np
import random


def augment_image(image, steering_angle):
    """
    Apply random augmentation to image and steering angle
    
    Args:
        image: Input image
        steering_angle: Corresponding steering angle
    
    Returns:
        Augmented image and adjusted steering angle
    """
    # Randomly apply augmentations
    if random.random() > 0.5:
        image, steering_angle = flip_image(image, steering_angle)
    
    if random.random() > 0.7:
        image = adjust_brightness(image)
    
    if random.random() > 0.8:
        image = add_shadow(image)
    
    if random.random() > 0.9:
        image, steering_angle = translate_image(image, steering_angle)
    
    if random.random() > 0.8:
        image = add_noise(image)
    
    return image, steering_angle


def flip_image(image, steering_angle):
    """
    Flip image horizontally and negate steering angle
    
    Args:
        image: Input image
        steering_angle: Steering angle
    
    Returns:
        Flipped image and negated steering angle
    """
    flipped_image = cv2.flip(image, 1)
    flipped_steering = -steering_angle
    
    return flipped_image, flipped_steering


def adjust_brightness(image):
    """
    Randomly adjust image brightness
    
    Args:
        image: Input image
    
    Returns:
        Brightness-adjusted image
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Random brightness adjustment
    brightness = random.uniform(0.3, 1.3)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness, 0, 255).astype(np.uint8)
    
    # Convert back to RGB
    adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return adjusted


def add_shadow(image):
    """
    Add random shadow to simulate different lighting conditions
    
    Args:
        image: Input image
    
    Returns:
        Image with shadow
    """
    h, w = image.shape[:2]
    
    # Random shadow coordinates
    x1 = random.randint(0, w)
    y1 = 0
    x2 = random.randint(0, w)
    y2 = h
    
    # Create mask
    mask = np.zeros_like(image)
    vertices = np.array([[(x1, y1), (x2, y2), (0, h), (0, 0)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, (1, 1, 1))
    
    # Apply shadow
    shadow_strength = random.uniform(0.3, 0.7)
    shadowed = image.copy()
    shadowed[mask[:, :, 0] == 1] = (shadowed[mask[:, :, 0] == 1] * shadow_strength).astype(np.uint8)
    
    return shadowed


def translate_image(image, steering_angle, max_shift=20):
    """
    Translate image horizontally and adjust steering angle
    
    Args:
        image: Input image
        steering_angle: Steering angle
        max_shift: Maximum horizontal shift in pixels
    
    Returns:
        Translated image and adjusted steering angle
    """
    h, w = image.shape[:2]
    
    # Random translation
    tx = random.randint(-max_shift, max_shift)
    ty = 0
    
    # Translation matrix
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    
    # Apply translation
    translated = cv2.warpAffine(image, M, (w, h))
    
    # Adjust steering angle (approximate)
    # More shift right = turn left (negative steering)
    steering_adjustment = -tx / max_shift * 0.1
    adjusted_steering = steering_angle + steering_adjustment
    
    return translated, adjusted_steering


def add_noise(image):
    """
    Add random noise to image
    
    Args:
        image: Input image
    
    Returns:
        Noisy image
    """
    # Gaussian noise
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    noisy = cv2.add(image, noise)
    
    return noisy


def crop_image(image, top_crop=0.3, bottom_crop=0.1):
    """
    Crop image to remove sky and car hood
    
    Args:
        image: Input image
        top_crop: Fraction to crop from top
        bottom_crop: Fraction to crop from bottom
    
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    
    top = int(h * top_crop)
    bottom = int(h * (1 - bottom_crop))
    
    cropped = image[top:bottom, :]
    
    return cropped


def preprocess_image(image, target_size=(120, 160), crop=True):
    """
    Preprocess image for model input
    
    Args:
        image: Input image
        target_size: Target size (height, width)
        crop: Whether to crop image
    
    Returns:
        Preprocessed image
    """
    # Crop if requested
    if crop:
        image = crop_image(image)
    
    # Resize
    resized = cv2.resize(image, (target_size[1], target_size[0]))
    
    return resized


def visualize_augmentations(image_path, steering_angle=0.0):
    """
    Visualize different augmentations on a single image
    
    Args:
        image_path: Path to test image
        steering_angle: Original steering angle
    """
    import matplotlib.pyplot as plt
    
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create augmentations
    augmentations = [
        ("Original", image, steering_angle),
        ("Flipped", *flip_image(image.copy(), steering_angle)),
        ("Brightness", adjust_brightness(image.copy()), steering_angle),
        ("Shadow", add_shadow(image.copy()), steering_angle),
        ("Translated", *translate_image(image.copy(), steering_angle)),
        ("Noise", add_noise(image.copy()), steering_angle),
        ("Cropped", crop_image(image.copy()), steering_angle),
        ("Combined", *augment_image(image.copy(), steering_angle))
    ]
    
    # Plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    for i, (title, img, angle) in enumerate(augmentations):
        axes[i].imshow(img)
        axes[i].set_title(f"{title}\nSteering: {angle:.3f}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_examples.png', dpi=150)
    plt.show()


def batch_augment(images, steering_angles):
    """
    Apply augmentation to a batch of images
    
    Args:
        images: Batch of images
        steering_angles: Corresponding steering angles
    
    Returns:
        Augmented images and steering angles
    """
    augmented_images = []
    augmented_angles = []
    
    for img, angle in zip(images, steering_angles):
        aug_img, aug_angle = augment_image(img, angle)
        augmented_images.append(aug_img)
        augmented_angles.append(aug_angle)
    
    return np.array(augmented_images), np.array(augmented_angles)


if __name__ == "__main__":
    # Test augmentations
    print("Testing data augmentation...")
    
    # Create a test image
    test_image = np.ones((240, 320, 3), dtype=np.uint8) * 128
    cv2.rectangle(test_image, (100, 100), (220, 140), (255, 0, 0), -1)
    cv2.circle(test_image, (160, 120), 30, (0, 255, 0), -1)
    
    # Test each augmentation
    print("Testing flip...")
    flipped, angle = flip_image(test_image, 0.5)
    print(f"Original angle: 0.5, Flipped angle: {angle}")
    
    print("Testing brightness...")
    bright = adjust_brightness(test_image)
    
    print("Testing shadow...")
    shadow = add_shadow(test_image)
    
    print("Testing translation...")
    translated, angle = translate_image(test_image, 0.5)
    print(f"Original angle: 0.5, Translated angle: {angle}")
    
    print("Testing noise...")
    noisy = add_noise(test_image)
    
    print("Testing crop...")
    cropped = crop_image(test_image)
    print(f"Original shape: {test_image.shape}, Cropped shape: {cropped.shape}")
    
    print("\nAll augmentations working correctly!")
