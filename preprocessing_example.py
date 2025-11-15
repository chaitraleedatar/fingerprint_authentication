"""
Example script demonstrating fingerprint image preprocessing.

This script shows how to use the preprocessing module to enhance fingerprint images.
"""

import os
import cv2
import matplotlib.pyplot as plt
from preprocessing import (
    load_image, 
    preprocess_pipeline, 
    preprocess_directory,
    save_image
)


def visualize_preprocessing(image_path: str, output_dir: str = "preprocessed_output") -> None:
    """
    Visualize the preprocessing steps on a sample image.
    
    Args:
        image_path: Path to input fingerprint image
        output_dir: Directory to save preprocessed images
    """
    # Load original image
    original = load_image(image_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get filename without extension
    filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Different preprocessing configurations
    configs = {
        'basic': {
            'normalize': True,
            'denoise': True,
            'enhance_contrast': True,
            'enhance_ridges': False,
            'binarize': False
        },
        'enhanced': {
            'normalize': True,
            'denoise': True,
            'denoise_method': 'median',
            'denoise_kernel_size': 5,
            'enhance_contrast': True,
            'clahe_clip_limit': 2.0,
            'clahe_tile_size': (8, 8),
            'enhance_ridges': False,
            'binarize': False
        },
        'binarized': {
            'normalize': True,
            'denoise': True,
            'enhance_contrast': True,
            'enhance_ridges': False,
            'binarize': True,
            'binarize_method': 'adaptive',
            'binarize_block_size': 11,
            'binarize_c': 2
        }
    }
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    # Original image
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Processed images
    for idx, (config_name, config) in enumerate(configs.items(), 1):
        processed = preprocess_pipeline(original, **config)
        
        axes[idx].imshow(processed, cmap='gray')
        axes[idx].set_title(f'{config_name.capitalize()} Preprocessing')
        axes[idx].axis('off')
        
        # Save processed image
        output_path = os.path.join(output_dir, f"{filename}_{config_name}.bmp")
        save_image(processed, output_path)
        print(f"Saved: {output_path}")
    
    # Save visualization
    plt.tight_layout()
    viz_path = os.path.join(output_dir, f"{filename}_comparison.png")
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization: {viz_path}")
    plt.close()


def preprocess_train_data():
    """
    Preprocess all training images.
    """
    input_dir = "project-data/Project-Data/train"
    output_dir = "preprocessed_output/train"
    
    # Standard preprocessing configuration
    config = {
        'normalize': True,
        'denoise': True,
        'denoise_method': 'median',
        'denoise_kernel_size': 5,
        'enhance_contrast': True,
        'clahe_clip_limit': 2.0,
        'clahe_tile_size': (8, 8),
        'enhance_ridges': False,
        'binarize': False
    }
    
    preprocess_directory(input_dir, output_dir, **config)


def preprocess_validate_data():
    """
    Preprocess all validation images.
    """
    input_dir = "project-data/Project-Data/validate"
    output_dir = "preprocessed_output/validate"
    
    # Standard preprocessing configuration
    config = {
        'normalize': True,
        'denoise': True,
        'denoise_method': 'median',
        'denoise_kernel_size': 5,
        'enhance_contrast': True,
        'clahe_clip_limit': 2.0,
        'clahe_tile_size': (8, 8),
        'enhance_ridges': False,
        'binarize': False
    }
    
    preprocess_directory(input_dir, output_dir, **config)


if __name__ == "__main__":
    # Example: Visualize preprocessing on a sample image
    sample_image = "project-data/Project-Data/train/000_R0_0.bmp"
    
    if os.path.exists(sample_image):
        print("Visualizing preprocessing on sample image...")
        visualize_preprocessing(sample_image)
        print("\n" + "="*50 + "\n")
    
    # Example: Preprocess training data
    print("Preprocessing training data...")
    if os.path.exists("project-data/Project-Data/train"):
        preprocess_train_data()
        print("\n" + "="*50 + "\n")
    
    # Example: Preprocess validation data
    print("Preprocessing validation data...")
    if os.path.exists("project-data/Project-Data/validate"):
        preprocess_validate_data()
    
    print("\nPreprocessing examples complete!")



