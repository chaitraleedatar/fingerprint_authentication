"""
Simple test script to verify preprocessing functionality.
"""

import os
from preprocessing import load_image, preprocess_pipeline, save_image
from utils import parse_filename, get_images_by_person


def test_preprocessing():
    """Test preprocessing on a sample image."""
    # Test with a sample image from train directory
    train_dir = "project-data/Project-Data/train"
    
    if not os.path.exists(train_dir):
        print(f"Error: {train_dir} not found!")
        return
    
    # Get first available image
    images = [f for f in os.listdir(train_dir) if f.endswith('.bmp')]
    if not images:
        print("No images found in train directory!")
        return
    
    test_image = os.path.join(train_dir, images[0])
    print(f"Testing preprocessing on: {test_image}")
    
    # Test filename parsing
    person_id, image_index = parse_filename(images[0])
    print(f"Parsed filename - Person ID: {person_id}, Image Index: {image_index}")
    
    # Load and preprocess
    try:
        original = load_image(test_image)
        print(f"Original image shape: {original.shape}")
        print(f"Original image dtype: {original.dtype}")
        print(f"Original image range: [{original.min()}, {original.max()}]")
        
        # Test basic preprocessing
        processed = preprocess_pipeline(
            original,
            normalize=True,
            denoise=True,
            enhance_contrast=True,
            enhance_ridges=False,
            binarize=False
        )
        
        print(f"Processed image shape: {processed.shape}")
        print(f"Processed image dtype: {processed.dtype}")
        print(f"Processed image range: [{processed.min()}, {processed.max()}]")
        
        # Save test output
        output_dir = "preprocessed_output/test"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"test_{images[0]}")
        save_image(processed, output_path)
        print(f"Saved preprocessed image to: {output_path}")
        
        print("\n[OK] Preprocessing test passed!")
        
    except Exception as e:
        print(f"[ERROR] Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()


def test_utils():
    """Test utility functions."""
    print("\nTesting utility functions...")
    
    # Test filename parsing
    test_cases = [
        ("000_R0_0.bmp", ("000", 0)),
        ("123_R0_4.bmp", ("123", 4)),
        ("001_R0_2.bmp", ("001", 2)),
    ]
    
    for filename, expected in test_cases:
        person_id, image_index = parse_filename(filename)
        if person_id == expected[0] and image_index == expected[1]:
            print(f"[OK] Parsed {filename} correctly")
        else:
            print(f"[FAIL] Failed to parse {filename}: got ({person_id}, {image_index}), expected {expected}")
    
    # Test directory grouping
    train_dir = "project-data/Project-Data/train"
    if os.path.exists(train_dir):
        images_by_person = get_images_by_person(train_dir)
        print(f"\nFound {len(images_by_person)} persons in train directory")
        if images_by_person:
            first_person = list(images_by_person.keys())[0]
            print(f"Person {first_person} has {len(images_by_person[first_person])} images")
            print(f"  Images: {images_by_person[first_person]}")


if __name__ == "__main__":
    print("=" * 50)
    print("Testing Preprocessing Module")
    print("=" * 50)
    
    test_preprocessing()
    test_utils()
    
    print("\n" + "=" * 50)
    print("Tests complete!")
    print("=" * 50)



