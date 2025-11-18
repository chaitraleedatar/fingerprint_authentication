"""
Batch Feature Extraction Script

Extracts features from thinned images in thinned_output_from_preprocessed
and saves them for faster enrollment and matching.
"""

import os
import json
import numpy as np
from typing import Dict, List
from tqdm import tqdm

from preprocessing import load_image
from feature_extraction import extract_features, minutiae_to_features
from utils import get_images_by_person, get_all_person_ids


def extract_features_from_thinned(image_path: str, 
                                  block_size: int = 16,
                                  min_distance: int = 10) -> Dict:
    """
    Extract features from a thinned image.
    
    Args:
        image_path: Path to thinned image
        block_size: Block size for orientation/frequency estimation
        min_distance: Minimum distance between minutiae
        
    Returns:
        Dictionary containing features, orientation_field, frequency_field
    """
    # Load thinned image
    thinned_image = load_image(image_path)
    
    # Extract features (is_thinned=True since image is already thinned)
    features_dict = extract_features(
        thinned_image,
        block_size=block_size,
        min_distance=min_distance,
        is_thinned=True,
        remove_border_false_positives=True
    )
    
    # Convert minutiae to feature array
    features_array = minutiae_to_features(features_dict['minutiae'])
    
    return {
        'features': features_array,
        'minutiae_count': len(features_dict['minutiae']),
        'endings': len([m for m in features_dict['minutiae'] if m.type == 'ending']),
        'bifurcations': len([m for m in features_dict['minutiae'] if m.type == 'bifurcation']),
        'orientation_field': features_dict['orientation_field'],
        'frequency_field': features_dict['frequency_field'],
        'image_shape': thinned_image.shape
    }


def batch_extract_features(input_base_dir: str = "thinned_output_from_preprocessed",
                          output_base_dir: str = "extracted_features",
                          block_size: int = 16,
                          min_distance: int = 10) -> None:
    """
    Batch extract features from all thinned images.
    
    Args:
        input_base_dir: Base directory containing thinned images (with train/validate/test subdirs)
        output_base_dir: Base directory to save extracted features
        block_size: Block size for feature extraction
        min_distance: Minimum distance between minutiae
    """
    print("\n" + "="*70)
    print("BATCH FEATURE EXTRACTION")
    print("="*70)
    print(f"Input directory: {input_base_dir}")
    print(f"Output directory: {output_base_dir}")
    print(f"Using Crossing Number method for minutiae extraction")
    print("="*70)
    
    subdirs = ['train', 'validate', 'test']
    total_images = 0
    total_features = 0
    
    for subdir in subdirs:
        input_dir = os.path.join(input_base_dir, subdir)
        output_dir = os.path.join(output_base_dir, subdir)
        
        if not os.path.exists(input_dir):
            print(f"\n⚠ Skipping {subdir}: directory not found")
            continue
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all images
        images_by_person = get_images_by_person(input_dir)
        person_ids = sorted(images_by_person.keys(), key=lambda x: int(x))
        
        print(f"\nProcessing {subdir}: {len(person_ids)} persons")
        
        for person_id in tqdm(person_ids, desc=f"  {subdir}"):
            image_files = images_by_person[person_id]
            
            person_features = []
            
            for image_file in image_files:
                image_path = os.path.join(input_dir, image_file)
                
                try:
                    # Extract features
                    features_dict = extract_features_from_thinned(
                        image_path,
                        block_size=block_size,
                        min_distance=min_distance
                    )
                    
                    if features_dict['minutiae_count'] > 0:
                        person_features.append({
                            'image_file': image_file,
                            'features': features_dict['features'].tolist(),
                            'minutiae_count': features_dict['minutiae_count'],
                            'endings': features_dict['endings'],
                            'bifurcations': features_dict['bifurcations'],
                            'image_shape': features_dict['image_shape']
                        })
                        total_features += features_dict['minutiae_count']
                    
                except Exception as e:
                    print(f"\n  ⚠ Error processing {image_file}: {e}")
                    continue
            
            if len(person_features) > 0:
                # Save person's features
                output_file = os.path.join(output_dir, f"{person_id}.json")
                with open(output_file, 'w') as f:
                    json.dump({
                        'person_id': person_id,
                        'num_images': len(person_features),
                        'images': person_features
                    }, f, indent=2)
                
                total_images += len(person_features)
    
    print("\n" + "="*70)
    print("FEATURE EXTRACTION COMPLETE")
    print("="*70)
    print(f"Total images processed: {total_images}")
    print(f"Total minutiae extracted: {total_features}")
    print(f"Average minutiae per image: {total_features / total_images if total_images > 0 else 0:.1f}")
    print(f"Features saved to: {output_base_dir}")
    print("="*70)


if __name__ == "__main__":
    batch_extract_features(
        input_base_dir="thinned_output_from_preprocessed",
        output_base_dir="extracted_features",
        block_size=16,
        min_distance=10
    )

