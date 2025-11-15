"""
Interactive demonstration of fingerprint preprocessing steps.

This script shows what each preprocessing step does to a fingerprint image.
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from preprocessing import (
    load_image,
    normalize_image,
    reduce_noise_gaussian,
    reduce_noise_median,
    enhance_contrast_clahe,
    enhance_histogram_equalization,
    preprocess_pipeline
)


def demonstrate_preprocessing_steps(image_path: str, output_dir: str = "preprocessed_output/demo"):
    """
    Demonstrate each preprocessing step individually.
    
    Args:
        image_path: Path to input fingerprint image
        output_dir: Directory to save demonstration images
    """
    print("=" * 70)
    print("FINGERPRINT PREPROCESSING DEMONSTRATION")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original image
    print(f"\n1. Loading image: {image_path}")
    original = load_image(image_path)
    print(f"   - Image shape: {original.shape}")
    print(f"   - Image range: [{original.min()}, {original.max()}]")
    print(f"   - Mean intensity: {original.mean():.2f}")
    print(f"   - Std deviation: {original.std():.2f}")
    
    # Get filename for saving
    filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    # Plot 0: Original
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('0. Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    print("\n2. Original image displayed")
    
    # Step 1: Normalization
    print("\n3. Applying NORMALIZATION...")
    print("   Purpose: Standardize image intensity to have consistent mean and std")
    print("   Effect: Makes images comparable regardless of lighting conditions")
    normalized = normalize_image(original)
    axes[1].imshow(normalized, cmap='gray')
    axes[1].set_title('1. Normalized\n(Mean=127.5, Std=127.5)', fontsize=11)
    axes[1].axis('off')
    print(f"   - New mean: {normalized.mean():.2f}")
    print(f"   - New std: {normalized.std():.2f}")
    
    # Step 2: Gaussian Denoising
    print("\n4. Applying GAUSSIAN DENOISING...")
    print("   Purpose: Reduce high-frequency noise using Gaussian blur")
    print("   Effect: Smooths the image while preserving main features")
    gaussian_denoised = reduce_noise_gaussian(normalized, kernel_size=5, sigma=1.0)
    axes[2].imshow(gaussian_denoised, cmap='gray')
    axes[2].set_title('2. Gaussian Denoised\n(kernel=5, σ=1.0)', fontsize=11)
    axes[2].axis('off')
    
    # Step 3: Median Denoising
    print("\n5. Applying MEDIAN DENOISING...")
    print("   Purpose: Remove salt-and-pepper noise while preserving edges")
    print("   Effect: Better at preserving ridge structures than Gaussian")
    median_denoised = reduce_noise_median(normalized, kernel_size=5)
    axes[3].imshow(median_denoised, cmap='gray')
    axes[3].set_title('3. Median Denoised\n(kernel=5)', fontsize=11)
    axes[3].axis('off')
    
    # Step 4: Histogram Equalization
    print("\n6. Applying HISTOGRAM EQUALIZATION...")
    print("   Purpose: Enhance contrast globally across the entire image")
    print("   Effect: Improves visibility of ridges and valleys")
    hist_eq = enhance_histogram_equalization(normalized)
    axes[4].imshow(hist_eq, cmap='gray')
    axes[4].set_title('4. Histogram Equalized', fontsize=11)
    axes[4].axis('off')
    
    # Step 5: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    print("\n7. Applying CLAHE (Contrast Enhancement)...")
    print("   Purpose: Enhance contrast adaptively in small regions")
    print("   Effect: Better than global histogram eq - preserves local details")
    clahe_enhanced = enhance_contrast_clahe(normalized, clip_limit=2.0, tile_grid_size=(8, 8))
    axes[5].imshow(clahe_enhanced, cmap='gray')
    axes[5].set_title('5. CLAHE Enhanced\n(clip=2.0, tile=8x8)', fontsize=11)
    axes[5].axis('off')
    
    # Step 6: Complete Pipeline (Normalize + Median + CLAHE)
    print("\n8. Applying COMPLETE PIPELINE...")
    print("   Steps: Normalize -> Median Denoise -> CLAHE")
    print("   This is the recommended preprocessing for fingerprints")
    complete = preprocess_pipeline(
        original,
        normalize=True,
        denoise=True,
        denoise_method='median',
        denoise_kernel_size=5,
        enhance_contrast=True,
        clahe_clip_limit=2.0,
        clahe_tile_size=(8, 8),
        enhance_ridges=False,
        binarize=False
    )
    axes[6].imshow(complete, cmap='gray')
    axes[6].set_title('6. Complete Pipeline\n(Norm + Median + CLAHE)', fontsize=11)
    axes[6].axis('off')
    
    # Step 7: Show difference
    diff = cv2.absdiff(original.astype(np.int16), complete.astype(np.int16))
    axes[7].imshow(diff, cmap='hot')
    axes[7].set_title('7. Difference Map\n(Original vs Processed)', fontsize=11)
    axes[7].axis('off')
    
    # Step 8: Histogram comparison
    axes[8].hist(original.flatten(), bins=50, alpha=0.5, label='Original', color='blue')
    axes[8].hist(complete.flatten(), bins=50, alpha=0.5, label='Processed', color='red')
    axes[8].set_title('8. Intensity Histogram\nComparison', fontsize=11)
    axes[8].set_xlabel('Pixel Intensity')
    axes[8].set_ylabel('Frequency')
    axes[8].legend()
    axes[8].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f"{filename}_preprocessing_demo.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n9. Saved demonstration figure to: {output_path}")
    
    # Also save the final processed image
    final_output = os.path.join(output_dir, f"{filename}_processed.bmp")
    cv2.imwrite(final_output, complete)
    print(f"10. Saved processed image to: {final_output}")
    
    plt.close()
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  - Normalization: Makes images comparable")
    print("  - Median Denoising: Removes noise while preserving edges")
    print("  - CLAHE: Enhances contrast locally for better ridge visibility")
    print("  - Complete Pipeline: Combines all steps for optimal results")
    print("\nThe processed image should have:")
    print("  - Better contrast between ridges and valleys")
    print("  - Reduced noise")
    print("  - Standardized intensity values")
    print("  - Enhanced ridge patterns for feature extraction")


if __name__ == "__main__":
    # Use a sample image from the training set
    sample_image = "project-data/Project-Data/train/000_R0_0.bmp"
    
    if not os.path.exists(sample_image):
        print(f"Error: Sample image not found at {sample_image}")
        print("Please make sure the project-data folder is in the correct location.")
    else:
        demonstrate_preprocessing_steps(sample_image)

