"""
Comprehensive test script for preprocessing and feature extraction.

This script tests the complete pipeline on multiple images and provides
detailed statistics and comparisons.
"""

import os
import numpy as np
from preprocessing import load_image, preprocess_pipeline
from feature_extraction import extract_features, minutiae_to_features
from utils import get_images_by_person, parse_filename


def test_single_image(image_path: str, person_id: str = None):
    """
    Test preprocessing and feature extraction on a single image.
    
    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*70}")
    print(f"Testing: {os.path.basename(image_path)}")
    if person_id:
        print(f"Person ID: {person_id}")
    print(f"{'='*70}")
    
    try:
        # Load image
        original = load_image(image_path)
        print(f"\n[1/6] Image loaded successfully")
        print(f"      Shape: {original.shape}")
        print(f"      Intensity range: [{original.min()}, {original.max()}]")
        print(f"      Mean: {original.mean():.2f}, Std: {original.std():.2f}")
        
        # Preprocess
        processed = preprocess_pipeline(
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
        print(f"\n[2/6] Preprocessing complete")
        print(f"      Processed range: [{processed.min()}, {processed.max()}]")
        print(f"      Processed mean: {processed.mean():.2f}, Std: {processed.std():.2f}")
        
        # Extract features
        features = extract_features(processed, block_size=16, min_distance=10)
        print(f"\n[3/6] Feature extraction complete")
        
        # Analyze minutiae
        minutiae = features['minutiae']
        endings = [m for m in minutiae if m.type == 'ending']
        bifurcations = [m for m in minutiae if m.type == 'bifurcation']
        
        print(f"\n[4/6] Minutiae analysis:")
        print(f"      Total minutiae: {len(minutiae)}")
        print(f"      Ridge endings: {len(endings)} ({len(endings)/len(minutiae)*100:.1f}%)")
        print(f"      Bifurcations: {len(bifurcations)} ({len(bifurcations)/len(minutiae)*100:.1f}%)")
        
        # Check feature quality
        if len(minutiae) < 20:
            print(f"      WARNING: Very few minutiae detected (< 20)")
        elif len(minutiae) > 300:
            print(f"      WARNING: Many minutiae detected (> 300), may include noise")
        else:
            print(f"      OK: Minutiae count in normal range")
        
        # Feature statistics
        if minutiae:
            orientations = [m.orientation for m in minutiae]
            print(f"\n[5/6] Orientation statistics:")
            print(f"      Mean: {np.mean(orientations):.2f} rad")
            print(f"      Std: {np.std(orientations):.2f} rad")
            print(f"      Range: [{np.min(orientations):.2f}, {np.max(orientations):.2f}] rad")
        
        # Convert to feature vector
        feature_vector = minutiae_to_features(minutiae)
        print(f"\n[6/6] Feature vector created")
        print(f"      Shape: {feature_vector.shape}")
        print(f"      Ready for matching: {len(minutiae) > 0}")
        
        return {
            'success': True,
            'image_path': image_path,
            'person_id': person_id,
            'image_shape': original.shape,
            'num_minutiae': len(minutiae),
            'num_endings': len(endings),
            'num_bifurcations': len(bifurcations),
            'feature_vector': feature_vector,
            'minutiae': minutiae
        }
        
    except Exception as e:
        print(f"\n[ERROR] Failed to process image: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'image_path': image_path,
            'error': str(e)
        }


def test_multiple_images(image_paths: list, max_images: int = 5):
    """
    Test preprocessing and feature extraction on multiple images.
    
    Args:
        image_paths: List of image paths to test
        max_images: Maximum number of images to test
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE TEST: Multiple Images")
    print("="*70)
    
    results = []
    test_images = image_paths[:max_images]
    
    for image_path in test_images:
        person_id, _ = parse_filename(os.path.basename(image_path))
        result = test_single_image(image_path, person_id)
        results.append(result)
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    print(f"\nTotal images tested: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        num_minutiae_list = [r['num_minutiae'] for r in successful]
        num_endings_list = [r['num_endings'] for r in successful]
        num_bifurcations_list = [r['num_bifurcations'] for r in successful]
        
        print(f"\nMinutiae Statistics:")
        print(f"  Mean minutiae per image: {np.mean(num_minutiae_list):.1f}")
        print(f"  Std deviation: {np.std(num_minutiae_list):.1f}")
        print(f"  Range: [{np.min(num_minutiae_list)}, {np.max(num_minutiae_list)}]")
        
        print(f"\nRidge Endings:")
        print(f"  Mean per image: {np.mean(num_endings_list):.1f}")
        print(f"  Range: [{np.min(num_endings_list)}, {np.max(num_endings_list)}]")
        
        print(f"\nBifurcations:")
        print(f"  Mean per image: {np.mean(num_bifurcations_list):.1f}")
        print(f"  Range: [{np.min(num_bifurcations_list)}, {np.max(num_bifurcations_list)}]")
        
        # Check consistency
        print(f"\nConsistency Check:")
        if np.std(num_minutiae_list) / np.mean(num_minutiae_list) < 0.3:
            print(f"  OK: Minutiae counts are relatively consistent")
        else:
            print(f"  WARNING: High variance in minutiae counts across images")
    
    if failed:
        print(f"\nFailed Images:")
        for r in failed:
            print(f"  - {os.path.basename(r['image_path'])}: {r.get('error', 'Unknown error')}")


def test_same_person_images(person_id: str = "000", train_dir: str = "project-data/Project-Data/train"):
    """
    Test multiple images from the same person to check consistency.
    
    Args:
        person_id: Person ID to test
        train_dir: Training data directory
    """
    print("\n" + "="*70)
    print(f"CONSISTENCY TEST: Same Person (ID: {person_id})")
    print("="*70)
    
    images_by_person = get_images_by_person(train_dir)
    
    if person_id not in images_by_person:
        print(f"Person ID {person_id} not found in training data")
        return
    
    image_files = images_by_person[person_id]
    image_paths = [os.path.join(train_dir, f) for f in image_files]
    
    print(f"\nFound {len(image_paths)} images for person {person_id}")
    
    results = []
    for image_path in image_paths:
        result = test_single_image(image_path, person_id)
        results.append(result)
    
    # Compare features from same person
    successful_results = [r for r in results if r.get('success', False)]
    
    if len(successful_results) >= 2:
        print(f"\n{'='*70}")
        print("SAME PERSON COMPARISON")
        print(f"{'='*70}")
        
        for i, r1 in enumerate(successful_results):
            for j, r2 in enumerate(successful_results[i+1:], i+1):
                m1 = r1['minutiae']
                m2 = r2['minutiae']
                
                print(f"\nComparing image {i+1} vs image {j+1}:")
                print(f"  Image {i+1}: {len(m1)} minutiae")
                print(f"  Image {j+1}: {len(m2)} minutiae")
                print(f"  Difference: {abs(len(m1) - len(m2))} minutiae")
                
                # Simple overlap check (would need matching algorithm for real comparison)
                print(f"  Note: Full matching requires enrollment/matching module")


def run_all_tests():
    """Run all comprehensive tests."""
    print("="*70)
    print("COMPREHENSIVE FINGERPRINT AUTHENTICATION SYSTEM TEST")
    print("="*70)
    
    # Test 1: Single image
    print("\n[TEST 1] Single Image Test")
    sample_image = "project-data/Project-Data/train/000_R0_0.bmp"
    if os.path.exists(sample_image):
        test_single_image(sample_image, "000")
    
    # Test 2: Multiple images from different people
    print("\n[TEST 2] Multiple Images Test")
    train_dir = "project-data/Project-Data/train"
    if os.path.exists(train_dir):
        images_by_person = get_images_by_person(train_dir)
        person_ids = sorted(list(images_by_person.keys()))[:5]  # First 5 people
        
        image_paths = []
        for pid in person_ids:
            first_image = images_by_person[pid][0]
            image_paths.append(os.path.join(train_dir, first_image))
        
        test_multiple_images(image_paths, max_images=5)
    
    # Test 3: Same person consistency
    print("\n[TEST 3] Same Person Consistency Test")
    if os.path.exists(train_dir):
        test_same_person_images("000", train_dir)
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)
    print("\nReview the results above to verify:")
    print("  - Preprocessing is working correctly")
    print("  - Feature extraction is consistent")
    print("  - Minutiae counts are reasonable")
    print("  - System is ready for enrollment and matching")


if __name__ == "__main__":
    run_all_tests()

