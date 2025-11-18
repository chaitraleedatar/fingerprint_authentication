"""
Test script for enrollment and matching module.

This script demonstrates:
1. Enrolling persons from training data
2. Identifying fingerprints from test/validation data
"""

import os
from enrollment_matching import (
    FingerprintDatabase,
    enroll_from_directory,
    identify_fingerprint,
    match_fingerprint,
    minutiae_to_features
)
from preprocessing import load_image, preprocess_pipeline, skeletonize
from feature_extraction import extract_features
from utils import get_images_by_person, get_all_person_ids


def test_enrollment():
    """Test enrollment functionality using thinned_output_from_preprocessed."""
    print("\n" + "="*70)
    print("TEST 1: ENROLLMENT (using thinned_output_from_preprocessed)")
    print("="*70)
    print("Using Crossing Number method for minutiae extraction")
    print("="*70)
    
    # Initialize database
    db = FingerprintDatabase(database_dir="database_thinned")
    
    # Enroll from thinned_output_from_preprocessed directory (already thinned images)
    train_dir = "thinned_output_from_preprocessed/train"
    
    if not os.path.exists(train_dir):
        print(f"Error: Thinned directory not found: {train_dir}")
        print("Please run batch_thinning_from_preprocessed.py first")
        return False
    
    # Enroll first 10 persons (for testing)
    all_person_ids = sorted(get_all_person_ids(train_dir), key=lambda x: int(x))
    person_ids = all_person_ids[:10]  # First 10
    print(f"\nFound {len(all_person_ids)} persons in thinned directory")
    print(f"Enrolling first 10 persons: {person_ids}")
    
    # Enroll with thinned images (already skeletonized)
    enroll_from_directory(train_dir, db, person_ids=person_ids, use_thinned=True, use_binarized=False)
    
    # Verify enrollment
    enrolled = db.list_enrolled_persons()
    print(f"\n✓ Enrolled persons: {enrolled}")
    
    return len(enrolled) > 0


def test_identification():
    """Test identification functionality using thinned_output_from_preprocessed."""
    print("\n" + "="*70)
    print("TEST 2: IDENTIFICATION (using thinned_output_from_preprocessed)")
    print("="*70)
    
    db = FingerprintDatabase(database_dir="database_thinned")
    enrolled = db.list_enrolled_persons()
    
    if len(enrolled) == 0:
        print("No enrolled persons found. Please run enrollment first.")
        return False
    
    print(f"Testing identification against {len(enrolled)} enrolled persons")
    
    # Test with thinned validation images
    validate_dir = "thinned_output_from_preprocessed/validate"
    
    if not os.path.exists(validate_dir):
        print(f"Error: Thinned validation directory not found: {validate_dir}")
        return False
    
    # Get images by person
    images_by_person = get_images_by_person(validate_dir)
    
    # Test identification for enrolled persons (limit to first 10)
    test_person_ids = [pid for pid in enrolled[:10] if pid in images_by_person]
    
    if len(test_person_ids) == 0:
        print("No matching persons found in validation directory")
        return False
    
    correct_matches = 0
    total_tests = 0
    
    for person_id in test_person_ids:
        image_files = images_by_person[person_id]
        if len(image_files) == 0:
            continue
        
        # Test with first validation image
        test_image = os.path.join(validate_dir, image_files[0])
        
        print(f"\nTesting identification for Person {person_id}...")
        result = identify_fingerprint(test_image, db, threshold=0.22, use_thinned=True, use_binarized=False)
        
        total_tests += 1
        
        if result and result['person_id'] == person_id:
            correct_matches += 1
            print(f"✓ Correctly identified Person {person_id} (score: {result['match_score']:.3f})")
        else:
            if result:
                print(f"✗ Misidentified as Person {result['person_id']} (expected {person_id}, score: {result['match_score']:.3f})")
            else:
                print(f"✗ Failed to identify Person {person_id}")
    
    print(f"\n{'='*70}")
    print(f"IDENTIFICATION RESULTS: {correct_matches}/{total_tests} correct")
    print(f"{'='*70}")
    
    return correct_matches > 0


def test_matching():
    """Test matching between two fingerprints using thinned images."""
    print("\n" + "="*70)
    print("TEST 3: DIRECT MATCHING (using thinned images)")
    print("="*70)
    
    train_dir = "thinned_output_from_preprocessed/train"
    
    if not os.path.exists(train_dir):
        print(f"Error: Thinned training directory not found: {train_dir}")
        return False
    
    # Get images for person 000
    images_by_person = get_images_by_person(train_dir)
    
    if "000" not in images_by_person or len(images_by_person["000"]) < 2:
        print("Need at least 2 images for person 000 to test matching")
        return False
    
    # Process two images from same person (already thinned)
    image1_path = os.path.join(train_dir, images_by_person["000"][0])
    image2_path = os.path.join(train_dir, images_by_person["000"][1])
    
    print(f"\nProcessing image 1: {os.path.basename(image1_path)}")
    image1 = load_image(image1_path)
    features1_dict = extract_features(image1, is_thinned=True)
    features1 = minutiae_to_features(features1_dict['minutiae'])
    
    print(f"Processing image 2: {os.path.basename(image2_path)}")
    image2 = load_image(image2_path)
    features2_dict = extract_features(image2, is_thinned=True)
    features2 = minutiae_to_features(features2_dict['minutiae'])
    
    print(f"\nImage 1: {len(features1)} minutiae")
    print(f"Image 2: {len(features2)} minutiae")
    
    # Match same person (using polar coordinate method)
    print("\nMatching same person (should match)...")
    is_match, score, info = match_fingerprint(
        features1, features2, 
        threshold=0.22,
        use_orb=False  # Use polar coordinate matching
    )
    
    print(f"Match result: {'✓ MATCH' if is_match else '✗ NO MATCH'}")
    print(f"Match score: {score:.3f}")
    print(f"Method: {info.get('method', 'Unknown')}")
    if 'rotation' in info:
        print(f"Rotation: {info['rotation']:.3f} rad")
        print(f"Translation: ({info['translation'][0]:.1f}, {info['translation'][1]:.1f})")
        print(f"Alignment score: {info.get('alignment_score', 'N/A')}")
    
    # Match different person (if available)
    if "001" in images_by_person:
        image3_path = os.path.join(train_dir, images_by_person["001"][0])
        print(f"\nProcessing different person image: {os.path.basename(image3_path)}")
        image3 = load_image(image3_path)
        features3_dict = extract_features(image3, is_thinned=True)
        features3 = minutiae_to_features(features3_dict['minutiae'])
        
        print("\nMatching different person (should NOT match)...")
        is_match2, score2, info2 = match_fingerprint(
            features1, features3, 
            threshold=0.22,
            use_orb=False  # Use polar coordinate matching
        )
        
        print(f"Match result: {'✗ MATCH (unexpected!)' if is_match2 else '✓ NO MATCH (correct)'}")
        print(f"Match score: {score2:.3f}")
    
    return True


def run_all_tests():
    """Run all enrollment and matching tests."""
    print("="*70)
    print("FINGERPRINT ENROLLMENT AND MATCHING TEST SUITE")
    print("="*70)
    
    # Test 1: Enrollment
    enrollment_success = test_enrollment()
    
    # Test 2: Matching
    matching_success = test_matching()
    
    # Test 3: Identification (requires enrollment)
    identification_success = False
    if enrollment_success:
        identification_success = test_identification()
    else:
        print("\nSkipping identification test (enrollment failed)")
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Enrollment: {'✓ PASS' if enrollment_success else '✗ FAIL'}")
    print(f"Matching: {'✓ PASS' if matching_success else '✗ FAIL'}")
    print(f"Identification: {'✓ PASS' if identification_success else '✗ FAIL'}")
    print("="*70)


if __name__ == "__main__":
    run_all_tests()

