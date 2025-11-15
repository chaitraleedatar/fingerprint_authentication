"""
Test script for feature extraction module.
"""

import os
import numpy as np
from preprocessing import load_image, preprocess_pipeline
from feature_extraction import extract_features, visualize_features


def test_feature_extraction():
    """Test feature extraction on a sample image."""
    print("=" * 70)
    print("FEATURE EXTRACTION TEST")
    print("=" * 70)
    
    # Load and preprocess a sample image
    sample_image = "project-data/Project-Data/train/000_R0_0.bmp"
    
    if not os.path.exists(sample_image):
        print(f"Error: Sample image not found at {sample_image}")
        return
    
    print(f"\n1. Loading and preprocessing image: {sample_image}")
    original = load_image(sample_image)
    
    # Preprocess
    processed = preprocess_pipeline(
        original,
        normalize=True,
        denoise=True,
        enhance_contrast=True,
        enhance_ridges=False,
        binarize=False
    )
    print(f"   - Image shape: {processed.shape}")
    
    # Extract features
    print("\n2. Extracting features...")
    print("   - Estimating orientation field...")
    print("   - Estimating ridge frequency...")
    print("   - Enhancing ridges with Gabor filters...")
    print("   - Binarizing image...")
    print("   - Skeletonizing...")
    print("   - Extracting minutiae...")
    
    features = extract_features(processed, block_size=16, min_distance=10)
    
    # Display results
    print("\n3. Feature Extraction Results:")
    print(f"   - Minutiae points found: {len(features['minutiae'])}")
    
    endings = [m for m in features['minutiae'] if m.type == 'ending']
    bifurcations = [m for m in features['minutiae'] if m.type == 'bifurcation']
    
    print(f"     * Ridge endings: {len(endings)}")
    print(f"     * Bifurcations: {len(bifurcations)}")
    
    # Show first few minutiae
    if features['minutiae']:
        print("\n   First 5 minutiae points:")
        for i, m in enumerate(features['minutiae'][:5]):
            print(f"     {i+1}. {m}")
    
    # Visualize
    print("\n4. Creating visualization...")
    output_dir = "feature_output"
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.splitext(os.path.basename(sample_image))[0]
    viz_path = os.path.join(output_dir, f"{filename}_features.png")
    
    visualize_features(processed, features, viz_path)
    
    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION TEST COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    test_feature_extraction()

