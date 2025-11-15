"""
Utility functions for fingerprint authentication system.

This module provides helper functions for:
- Parsing fingerprint image filenames
- Managing data directories
- Extracting person IDs and image indices
"""

import os
import re
from typing import Tuple, Optional, List


def parse_filename(filename: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Parse fingerprint image filename to extract person ID and image index.
    
    Filename format: YYY_R0_KKK.bmp
    where YYY is the person ID and KKK is the image index.
    
    Args:
        filename: Image filename (e.g., "000_R0_0.bmp")
        
    Returns:
        Tuple of (person_id, image_index) or (None, None) if parsing fails
    """
    # Extract pattern: YYY_R0_KKK.bmp
    pattern = r'(\d+)_R0_(\d+)\.bmp'
    match = re.match(pattern, filename)
    
    if match:
        person_id = match.group(1)
        image_index = int(match.group(2))
        return person_id, image_index
    
    return None, None


def get_person_id_from_filename(filename: str) -> Optional[str]:
    """
    Extract person ID from filename.
    
    Args:
        filename: Image filename
        
    Returns:
        Person ID as string, or None if parsing fails
    """
    person_id, _ = parse_filename(filename)
    return person_id


def get_image_index_from_filename(filename: str) -> Optional[int]:
    """
    Extract image index from filename.
    
    Args:
        filename: Image filename
        
    Returns:
        Image index as integer, or None if parsing fails
    """
    _, image_index = parse_filename(filename)
    return image_index


def get_images_by_person(directory: str) -> dict:
    """
    Group images by person ID.
    
    Args:
        directory: Directory containing fingerprint images
        
    Returns:
        Dictionary mapping person_id to list of image filenames
    """
    images_by_person = {}
    
    if not os.path.exists(directory):
        return images_by_person
    
    for filename in os.listdir(directory):
        if filename.lower().endswith('.bmp'):
            person_id, _ = parse_filename(filename)
            if person_id:
                if person_id not in images_by_person:
                    images_by_person[person_id] = []
                images_by_person[person_id].append(filename)
    
    # Sort images by index for each person
    for person_id in images_by_person:
        images_by_person[person_id].sort(key=lambda x: get_image_index_from_filename(x) or 0)
    
    return images_by_person


def get_all_person_ids(directory: str) -> List[str]:
    """
    Get all unique person IDs from a directory.
    
    Args:
        directory: Directory containing fingerprint images
        
    Returns:
        List of unique person IDs (sorted)
    """
    images_by_person = get_images_by_person(directory)
    person_ids = sorted(images_by_person.keys(), key=lambda x: int(x))
    return person_ids


def get_image_path(directory: str, person_id: str, image_index: int) -> Optional[str]:
    """
    Construct image path from person ID and image index.
    
    Args:
        directory: Directory containing images
        person_id: Person ID (e.g., "000")
        image_index: Image index (e.g., 0, 1, 2, 3, 4)
        
    Returns:
        Full path to image file, or None if not found
    """
    filename = f"{person_id}_R0_{image_index}.bmp"
    filepath = os.path.join(directory, filename)
    
    if os.path.exists(filepath):
        return filepath
    
    return None


def get_train_images_for_person(person_id: str, train_dir: str = "project-data/Project-Data/train") -> List[str]:
    """
    Get all training images for a specific person.
    
    Args:
        person_id: Person ID
        train_dir: Training data directory
        
    Returns:
        List of image paths for the person
    """
    images = []
    images_by_person = get_images_by_person(train_dir)
    
    if person_id in images_by_person:
        for filename in images_by_person[person_id]:
            images.append(os.path.join(train_dir, filename))
    
    return images


def get_validate_image_for_person(person_id: str, validate_dir: str = "project-data/Project-Data/validate") -> Optional[str]:
    """
    Get validation image for a specific person.
    
    Args:
        person_id: Person ID
        validate_dir: Validation data directory
        
    Returns:
        Path to validation image, or None if not found
    """
    return get_image_path(validate_dir, person_id, 3)


def get_test_image_for_person(person_id: str, test_dir: str = "project-data/Project-Data/test") -> Optional[str]:
    """
    Get test image for a specific person.
    
    Args:
        person_id: Person ID
        test_dir: Test data directory
        
    Returns:
        Path to test image, or None if not found
    """
    return get_image_path(test_dir, person_id, 4)



