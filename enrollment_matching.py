"""
Fingerprint Enrollment and Matching Module

This module provides:
- Enrollment: Store fingerprint features in a database
- Matching: Compare query fingerprints against enrolled data
- Identification: Identify unseen fingerprints
"""

import os
import json
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional

from preprocessing import load_image, preprocess_pipeline
from feature_extraction import extract_features, minutiae_to_features, Minutiae
from utils import get_images_by_person, get_all_person_ids


class FingerprintDatabase:
    """Manages fingerprint template database."""
    
    def __init__(self, database_dir: str = "database"):
        """
        Initialize database.
        
        Args:
            database_dir: Directory to store templates
        """
        self.database_dir = database_dir
        self.templates_dir = os.path.join(database_dir, "templates")
        os.makedirs(self.templates_dir, exist_ok=True)
    
    def save_template(self, person_id: str, features: np.ndarray, 
                     orientation_field: np.ndarray = None,
                     frequency_field: np.ndarray = None,
                     metadata: Dict = None):
        """
        Save fingerprint template to database with all features.
        
        Args:
            person_id: Person identifier
            features: Feature array (N×4) where each row is [x, y, orientation, type]
            orientation_field: Ridge orientation field (optional, can be downsampled)
            frequency_field: Ridge frequency field (optional, can be downsampled)
            metadata: Optional metadata (e.g., number of images used, enrollment date)
        """
        template_path = os.path.join(self.templates_dir, f"{person_id}.json")
        
        template_data = {
            'person_id': person_id,
            'features': features.tolist(),
            'num_minutiae': len(features),
            'metadata': metadata or {}
        }
        
        # Store orientation field (downsampled to save space)
        if orientation_field is not None:
            # Downsample by factor of 4 to reduce storage
            h, w = orientation_field.shape
            downsampled_orientation = orientation_field[::4, ::4]
            template_data['orientation_field'] = downsampled_orientation.tolist()
            template_data['orientation_shape'] = [h, w]
        
        # Store frequency field (downsampled to save space)
        if frequency_field is not None:
            h, w = frequency_field.shape
            downsampled_frequency = frequency_field[::4, ::4]
            template_data['frequency_field'] = downsampled_frequency.tolist()
            template_data['frequency_shape'] = [h, w]
        
        with open(template_path, 'w') as f:
            json.dump(template_data, f, indent=2)
    
    def load_template(self, person_id: str) -> Optional[Dict]:
        """
        Load fingerprint template from database.
        
        Args:
            person_id: Person identifier
            
        Returns:
            Template dictionary or None if not found
        """
        template_path = os.path.join(self.templates_dir, f"{person_id}.json")
        
        if not os.path.exists(template_path):
            return None
        
        with open(template_path, 'r') as f:
            template_data = json.load(f)
        
        # Convert features back to numpy array
        template_data['features'] = np.array(template_data['features'])
        
        # Reconstruct orientation field if available
        if 'orientation_field' in template_data:
            downsampled = np.array(template_data['orientation_field'])
            h, w = template_data['orientation_shape']
            # Upsample back to original size using nearest neighbor
            from scipy.ndimage import zoom
            zoom_factors = (h / downsampled.shape[0], w / downsampled.shape[1])
            template_data['orientation_field'] = zoom(downsampled, zoom_factors, order=0)
        
        # Reconstruct frequency field if available
        if 'frequency_field' in template_data:
            downsampled = np.array(template_data['frequency_field'])
            h, w = template_data['frequency_shape']
            from scipy.ndimage import zoom
            zoom_factors = (h / downsampled.shape[0], w / downsampled.shape[1])
            template_data['frequency_field'] = zoom(downsampled, zoom_factors, order=0)
        
        return template_data
    
    def list_enrolled_persons(self) -> List[str]:
        """
        Get list of all enrolled person IDs.
        
        Returns:
            List of person IDs
        """
        if not os.path.exists(self.templates_dir):
            return []
        
        person_ids = []
        for filename in os.listdir(self.templates_dir):
            if filename.endswith('.json'):
                person_id = filename[:-5]  # Remove .json extension
                person_ids.append(person_id)
        
        return sorted(person_ids, key=lambda x: int(x))
    
    def template_exists(self, person_id: str) -> bool:
        """Check if template exists for a person."""
        template_path = os.path.join(self.templates_dir, f"{person_id}.json")
        return os.path.exists(template_path)


def enroll_person(person_id: str, image_paths: List[str], 
                 database: FingerprintDatabase,
                 block_size: int = 16, min_distance: int = 20,
                 use_thinned: bool = False, use_binarized: bool = False) -> bool:
    """
    Enroll a person by processing multiple images and creating a template.
    
    Args:
        person_id: Person identifier
        image_paths: List of image file paths for this person
        database: FingerprintDatabase instance
        block_size: Block size for feature extraction
        min_distance: Minimum distance between minutiae
        use_thinned: If True, image is already thinned/skeletonized
        use_binarized: If True, image is binarized and needs thinning
        
    Returns:
        True if enrollment successful, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"Enrolling Person ID: {person_id}")
    print(f"{'='*70}")
    
    all_features = []
    all_orientation_fields = []
    all_frequency_fields = []
    successful_images = 0
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"\nProcessing image {i}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        try:
            # Load image
            image = load_image(image_path)
            
            if use_thinned:
                # Image is already thinned, use directly
                processed = image
            elif use_binarized:
                # Image is binarized, need to thin it first
                from preprocessing import skeletonize
                # Ensure image is binary (0 or 255)
                if image.max() > 1:
                    binary = (image > 127).astype(np.uint8) * 255
                else:
                    binary = image.astype(np.uint8) * 255
                # Thin the binarized image
                processed = skeletonize(binary)
            else:
                # Preprocess normally
                processed = preprocess_pipeline(
                    image,
                    normalize=True,
                    denoise=True,
                    enhance_contrast=True,
                    enhance_ridges=False,
                    binarize=False
                )
            
            # Extract all features (minutiae, orientation field, frequency field)
            features_dict = extract_features(processed, block_size=block_size, 
                                           min_distance=min_distance,
                                           is_thinned=(use_thinned or use_binarized))
            minutiae = features_dict['minutiae']
            orientation_field = features_dict['orientation_field']
            frequency_field = features_dict['frequency_field']
            
            if len(minutiae) == 0:
                print(f"  Warning: No minutiae found in image {i}")
                continue
            
            # Convert to feature array
            features = minutiae_to_features(minutiae)
            all_features.append(features)
            all_orientation_fields.append(orientation_field)
            all_frequency_fields.append(frequency_field)
            successful_images += 1
            
            print(f"  ✓ Extracted {len(minutiae)} minutiae")
            print(f"  ✓ Orientation field: {orientation_field.shape}")
            print(f"  ✓ Frequency field: {frequency_field.shape}")
            
        except Exception as e:
            print(f"  ✗ Error processing image {i}: {str(e)}")
            continue
    
    if len(all_features) == 0:
        print(f"\n✗ Enrollment failed: No features extracted from any image")
        return False
    
    # Combine features from all images
    # Strategy: Use all minutiae from all images
    combined_features = np.vstack(all_features)
    
    # Average orientation and frequency fields from all images
    # This creates a more robust template
    avg_orientation = np.mean(all_orientation_fields, axis=0)
    avg_frequency = np.mean(all_frequency_fields, axis=0)
    
    print(f"\n✓ Enrollment successful!")
    print(f"  - Processed {successful_images}/{len(image_paths)} images")
    print(f"  - Total minutiae in template: {len(combined_features)}")
    print(f"  - Averaged orientation field: {avg_orientation.shape}")
    print(f"  - Averaged frequency field: {avg_frequency.shape}")
    
    # Save template with all features
    metadata = {
        'num_images': successful_images,
        'total_minutiae': len(combined_features)
    }
    database.save_template(person_id, combined_features, 
                          orientation_field=avg_orientation,
                          frequency_field=avg_frequency,
                          metadata=metadata)
    
    return True


def convert_to_polar_coordinates(features: np.ndarray, ref_idx: int) -> np.ndarray:
    """
    Convert minutiae to polar coordinates relative to a reference point.
    
    Based on Equation (4) and (5) from the algorithm:
    - r: radial distance = √((row - row_ref)² + (col - col_ref)²)
    - φ: radial angle = tan⁻¹((row - row_ref) / (col - col_ref))
    - θ: orientation angle = θ - θ_ref
    
    Args:
        features: Feature array (N×4) [x, y, orientation, type]
                 Note: x=col, y=row in image coordinates
        ref_idx: Index of reference minutia
        
    Returns:
        Polar coordinates array (N×3) [r, φ, θ]
    """
    if len(features) == 0:
        return np.array([]).reshape(0, 3)
    
    # Extract reference point (note: features use [x, y] = [col, row])
    ref_x, ref_y, ref_theta, _ = features[ref_idx]
    
    # Extract all points
    x = features[:, 0]  # col
    y = features[:, 1]  # row
    theta = features[:, 2]  # orientation
    
    # Calculate radial distance: r = √((row - row_ref)² + (col - col_ref)²)
    r = np.sqrt((y - ref_y)**2 + (x - ref_x)**2)
    
    # Calculate radial angle: φ = tan⁻¹((row - row_ref) / (col - col_ref))
    # Use atan2 for proper quadrant handling
    phi = np.arctan2(y - ref_y, x - ref_x)
    
    # Calculate relative orientation: θ = θ - θ_ref
    theta_rel = theta - ref_theta
    # Normalize to [-π, π]
    theta_rel = np.where(theta_rel > np.pi, theta_rel - 2*np.pi, theta_rel)
    theta_rel = np.where(theta_rel < -np.pi, theta_rel + 2*np.pi, theta_rel)
    
    # Stack into polar coordinates
    polar = np.column_stack([r, phi, theta_rel])
    
    return polar


def align_minutiae_polar(features1: np.ndarray, features2: np.ndarray,
                         max_ref_pairs: int = 50) -> Tuple[float, float, float, int]:
    """
    Align minutiae using polar coordinate transformation with reference points.
    
    Algorithm based on the provided method:
    1. Select reference points from both template and input
    2. Convert all minutiae to polar coordinates relative to reference points
    3. Match minutiae in polar space (rotation and translation invariant)
    4. Return best transformation
    
    Args:
        features1: Template features (M×4) [x, y, orientation, type]
        features2: Query features (N×4) [x, y, orientation, type]
        max_ref_pairs: Maximum number of reference point pairs to try
        
    Returns:
        Tuple of (best_rotation, best_tx, best_ty, best_match_count)
    """
    if len(features1) == 0 or len(features2) == 0:
        return (0.0, 0.0, 0.0, 0)
    
    if len(features1) < 2 or len(features2) < 2:
        # Fallback: use centroid alignment
        return align_centroid_fallback(features1, features2)
    
    best_score = 0
    best_rotation = 0.0
    best_tx = 0.0
    best_ty = 0.0
    
    # Try different reference point pairs
    # Limit to avoid too many combinations
    n1 = min(len(features1), 15)
    n2 = min(len(features2), 15)
    
    pairs_tried = 0
    
    for i in range(n1):
        for j in range(n2):
            if pairs_tried >= max_ref_pairs:
                break
            
            # Convert both sets to polar coordinates
            polar1 = convert_to_polar_coordinates(features1, i)
            polar2 = convert_to_polar_coordinates(features2, j)
            
            # Match in polar space
            matches = match_polar_coordinates(polar1, polar2, features1, features2, i, j)
            
            if matches > best_score:
                best_score = matches
                # Calculate transformation from reference points
                ref1 = features1[i]
                ref2 = features2[j]
                
                # Rotation: difference in reference orientations
                rotation = ref1[2] - ref2[2]
                # Normalize
                while rotation > np.pi:
                    rotation -= 2*np.pi
                while rotation < -np.pi:
                    rotation += 2*np.pi
                
                # Translation: difference in reference positions after rotation
                cos_r = np.cos(rotation)
                sin_r = np.sin(rotation)
                rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
                ref2_rotated = ref2[:2] @ rotation_matrix.T
                translation = ref1[:2] - ref2_rotated
                
                best_rotation = rotation
                best_tx = translation[0]
                best_ty = translation[1]
            
            pairs_tried += 1
    
    return (best_rotation, best_tx, best_ty, best_score)


def match_polar_coordinates(polar1: np.ndarray, polar2: np.ndarray,
                           features1: np.ndarray, features2: np.ndarray,
                           ref_idx1: int, ref_idx2: int,
                           r_threshold: float = 5.0,
                           phi_threshold: float = 0.2,
                           theta_threshold: float = 0.3) -> int:
    """
    Match minutiae in polar coordinate space.
    
    Args:
        polar1: Template polar coordinates (M×3) [r, φ, θ]
        polar2: Query polar coordinates (N×3) [r, φ, θ]
        features1: Template features (for type matching)
        features2: Query features (for type matching)
        ref_idx1: Template reference index
        ref_idx2: Query reference index
        r_threshold: Radial distance threshold
        phi_threshold: Radial angle threshold (radians)
        theta_threshold: Orientation threshold (radians)
        
    Returns:
        Number of matches
    """
    if len(polar1) == 0 or len(polar2) == 0:
        return 0
    
    matches = 0
    matched_indices = set()
    
    # Skip reference points themselves
    for i in range(len(polar1)):
        if i == ref_idx1:
            continue
        
        r1, phi1, theta1 = polar1[i]
        type1 = features1[i, 3]
        
        best_match_idx = -1
        best_match_score = float('inf')
        
        for j in range(len(polar2)):
            if j == ref_idx2 or j in matched_indices:
                continue
            
            r2, phi2, theta2 = polar2[j]
            type2 = features2[j, 3]
            
            # Check radial distance
            r_diff = abs(r1 - r2)
            if r_diff > r_threshold:
                continue
            
            # Check radial angle (account for rotation)
            phi_diff = abs(phi1 - phi2)
            phi_diff = min(phi_diff, 2*np.pi - phi_diff)
            if phi_diff > phi_threshold:
                continue
            
            # Check orientation
            theta_diff = abs(theta1 - theta2)
            theta_diff = min(theta_diff, 2*np.pi - theta_diff)
            if theta_diff > theta_threshold:
                continue
            
            # Calculate match quality (lower is better)
            quality = (r_diff / r_threshold) * 0.4 + \
                     (phi_diff / phi_threshold) * 0.3 + \
                     (theta_diff / theta_threshold) * 0.3
            
            # Type mismatch penalty
            if type1 != type2:
                quality *= 1.2
            
            if quality < best_match_score:
                best_match_score = quality
                best_match_idx = j
        
        if best_match_idx >= 0:
            matches += 1
            matched_indices.add(best_match_idx)
    
    return matches


def align_centroid_fallback(features1: np.ndarray, features2: np.ndarray) -> Tuple[float, float, float, int]:
    """Fallback alignment using centroid method."""
    pos1 = features1[:, :2]
    pos2 = features2[:, :2]
    
    centroid1 = np.mean(pos1, axis=0)
    centroid2 = np.mean(pos2, axis=0)
    
    rotations = np.linspace(-np.pi/6, np.pi/6, 13)
    
    best_score = 0
    best_rotation = 0.0
    best_tx = 0.0
    best_ty = 0.0
    
    for rotation in rotations:
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        
        pos2_rotated = (pos2 - centroid2) @ rotation_matrix.T + centroid2
        translation = centroid1 - np.mean(pos2_rotated, axis=0)
        
        # Quick match count
        pos2_transformed = pos2 @ rotation_matrix.T + np.array(translation)
        matches = 0
        matched = set()
        for p1 in pos1:
            distances = np.sqrt(np.sum((pos2_transformed - p1)**2, axis=1))
            min_idx = np.argmin(distances)
            if distances[min_idx] < 15.0 and min_idx not in matched:
                matches += 1
                matched.add(min_idx)
        
        if matches > best_score:
            best_score = matches
            best_rotation = rotation
            best_tx = translation[0]
            best_ty = translation[1]
    
    return (best_rotation, best_tx, best_ty, best_score)


def filter_minutiae(features: np.ndarray, image_width: int = 356, image_height: int = 328,
                    border_margin: int = 30) -> np.ndarray:
    """
    Filter minutiae to keep only high-quality ones (remove edge points).
    
    Args:
        features: Feature array (N×4) [x, y, orientation, type]
        image_width: Image width
        image_height: Image height
        border_margin: Margin from edges to exclude
        
    Returns:
        Filtered feature array
    """
    if len(features) == 0:
        return features
    
    x = features[:, 0]
    y = features[:, 1]
    
    # Keep only minutiae in the center region (away from edges)
    mask = (x >= border_margin) & (x < image_width - border_margin) & \
           (y >= border_margin) & (y < image_height - border_margin)
    
    return features[mask]


def calculate_match_score(features1: np.ndarray, features2: np.ndarray,
                         rotation: float, tx: float, ty: float,
                         orientation_field1: np.ndarray = None,
                         orientation_field2: np.ndarray = None,
                         frequency_field1: np.ndarray = None,
                         frequency_field2: np.ndarray = None,
                         distance_threshold: float = 20.0,  # Slightly more lenient
                         orientation_threshold: float = np.pi/4) -> float:  # More lenient
    """
    Calculate matching score between two aligned minutiae sets.
    
    Args:
        features1: Template features (N×4) [x, y, orientation, type]
        features2: Query features (M×4) [x, y, orientation, type]
        rotation: Rotation angle applied to features2
        tx: Translation in x applied to features2
        ty: Translation in y applied to features2
        distance_threshold: Maximum distance for matching (pixels)
        orientation_threshold: Maximum orientation difference (radians)
        
    Returns:
        Match score (0.0 to 1.0)
    """
    if len(features1) == 0 or len(features2) == 0:
        return 0.0
    
    # Transform query features
    pos2 = features2[:, :2].copy()
    ori2 = features2[:, 2].copy()
    type2 = features2[:, 3].copy()
    
    # Apply rotation and translation
    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)
    rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    pos2_transformed = pos2 @ rotation_matrix.T + np.array([tx, ty])
    
    # Adjust orientation
    ori2_transformed = (ori2 + rotation) % (2 * np.pi)
    # Normalize to [-pi, pi]
    ori2_transformed = np.where(ori2_transformed > np.pi, 
                               ori2_transformed - 2*np.pi, 
                               ori2_transformed)
    
    # Extract template features
    pos1 = features1[:, :2]
    ori1 = features1[:, 2]
    type1 = features1[:, 3]
    
    # Find matching minutiae using strict bidirectional matching
    # Build list of all potential matches with quality scores
    potential_matches = []
    
    for i in range(len(features1)):
        for j in range(len(features2)):
            # Calculate distance
            dist = np.sqrt(np.sum((pos2_transformed[j] - pos1[i])**2))
            
            if dist <= distance_threshold:
                # Check orientation difference
                ori_diff = abs(ori1[i] - ori2_transformed[j])
                ori_diff = min(ori_diff, 2*np.pi - ori_diff)
                
                if ori_diff <= orientation_threshold:
                    # Type must match for high-quality match
                    type_match = (type1[i] == type2[j])
                    
                    # Calculate match quality (lower is better)
                    # Weight: distance 50%, orientation 30%, type 20%
                    quality = (dist / distance_threshold) * 0.5 + (ori_diff / orientation_threshold) * 0.3
                    
                    # Penalty if types don't match (allow but penalize)
                    if not type_match:
                        if dist > 12.0:  # Only allow type mismatch if close
                            continue  # Skip this match
                        quality += 0.2  # Light penalty
                    else:
                        quality += 0.0  # No penalty for type match
                    
                    potential_matches.append((i, j, quality, dist, ori_diff, type_match))
    
    # Sort by quality (best matches first)
    potential_matches.sort(key=lambda x: x[2])
    
    # Greedy matching: assign best matches first
    matched_template = set()
    matched_query = set()
    matches = 0
    type_matched_count = 0  # Count matches where types match
    
    for i, j, quality, dist, ori_diff, type_match in potential_matches:
        if i not in matched_template and j not in matched_query:
            matches += 1
            if type_match:
                type_matched_count += 1
            matched_template.add(i)
            matched_query.add(j)
    
    # Calculate match score using the formula from slide:
    # score = #matches / ((M + N) / 2)
    # where M = number of minutiae in template, N = number of minutiae in query
    M = len(features1)
    N = len(features2)
    
    if M == 0 or N == 0:
        return 0.0
    
    # Base score formula from slide: score = #matches / ((M + N) / 2)
    base_score = matches / ((M + N) / 2.0)
    
    # Apply quality-based adjustments
    match_score = base_score
    
    # Penalty if too few absolute matches (likely false match)
    if matches < 6:
        match_score *= 0.5  # Moderate penalty for very few matches
    elif matches < 10:
        match_score *= 0.7
    elif matches < 15:
        match_score *= 0.9
    # Bonus if many matches (likely genuine match)
    elif matches >= 25:
        match_score *= 1.2
    elif matches >= 20:
        match_score *= 1.15
    elif matches >= 15:
        match_score *= 1.1
    
    # Type match ratio: penalize if too many type mismatches
    if matches > 0:
        type_match_ratio = type_matched_count / matches
        # Require at least 65% type matches for high score (more relaxed)
        if type_match_ratio < 0.65:
            match_score *= (0.7 + 0.3 * type_match_ratio)  # Scale down based on ratio
        elif type_match_ratio >= 0.9:
            match_score *= 1.15  # Bonus for very high type match ratio
        elif type_match_ratio >= 0.8:
            match_score *= 1.08
        elif type_match_ratio >= 0.75:
            match_score *= 1.05
    
    # Use orientation field similarity as additional validation
    if orientation_field1 is not None and orientation_field2 is not None:
        # Transform orientation field 2
        h, w = orientation_field2.shape
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Apply rotation and translation
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        
        coords = np.stack([x_coords.flatten(), y_coords.flatten()], axis=1)
        coords_transformed = (coords @ rotation_matrix.T + np.array([tx, ty])).T
        
        # Sample orientation field 1 at transformed locations
        x_trans = coords_transformed[0].reshape(h, w)
        y_trans = coords_transformed[1].reshape(h, w)
        
        # Clip to valid range
        x_trans = np.clip(x_trans, 0, w - 1)
        y_trans = np.clip(y_trans, 0, h - 1)
        
        # Interpolate orientation field 1
        from scipy.ndimage import map_coordinates
        ori1_sampled = map_coordinates(orientation_field1, [y_trans, x_trans], order=1, mode='nearest')
        
        # Adjust orientation field 2 by rotation
        ori2_adjusted = (orientation_field2 + rotation) % (2 * np.pi)
        ori2_adjusted = np.where(ori2_adjusted > np.pi, ori2_adjusted - 2*np.pi, ori2_adjusted)
        
        # Calculate orientation field similarity
        ori_diff = np.abs(ori1_sampled - ori2_adjusted)
        ori_diff = np.minimum(ori_diff, 2*np.pi - ori_diff)
        
        # Valid region (where both fields are meaningful)
        valid_mask = (x_trans >= 0) & (x_trans < w) & (y_trans >= 0) & (y_trans < h)
        if np.sum(valid_mask) > 0:
            avg_ori_similarity = 1.0 - np.mean(ori_diff[valid_mask]) / np.pi
            # Boost score if orientation fields match well
            if avg_ori_similarity > 0.7:
                match_score *= 1.1
            elif avg_ori_similarity < 0.5:
                match_score *= 0.9
    
    # Use frequency field similarity as additional validation
    if frequency_field1 is not None and frequency_field2 is not None:
        # Similar transformation for frequency field
        h, w = frequency_field2.shape
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        
        coords = np.stack([x_coords.flatten(), y_coords.flatten()], axis=1)
        coords_transformed = (coords @ rotation_matrix.T + np.array([tx, ty])).T
        
        x_trans = np.clip(coords_transformed[0].reshape(h, w), 0, w - 1)
        y_trans = np.clip(coords_transformed[1].reshape(h, w), 0, h - 1)
        
        from scipy.ndimage import map_coordinates
        freq1_sampled = map_coordinates(frequency_field1, [y_trans, x_trans], order=1, mode='nearest')
        
        # Frequency should be similar (not affected by rotation)
        freq_diff = np.abs(freq1_sampled - frequency_field2)
        valid_mask = (x_trans >= 0) & (x_trans < w) & (y_trans >= 0) & (y_trans < h)
        if np.sum(valid_mask) > 0:
            avg_freq_similarity = 1.0 - np.minimum(np.mean(freq_diff[valid_mask]) / 0.1, 1.0)
            # Boost score if frequency fields match well
            if avg_freq_similarity > 0.8:
                match_score *= 1.05
    
    return match_score


def extract_orb_descriptors(image: np.ndarray, max_features: int = 1000) -> Tuple[List, np.ndarray]:
    """
    Extract ORB descriptors from entire image.
    
    Based on main.py method: detectAndCompute on entire image, not just minutiae locations.
    
    Args:
        image: Fingerprint image (grayscale)
        max_features: Maximum number of ORB features to detect (default 500)
        
    Returns:
        Tuple of (keypoints, descriptors)
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Initialize ORB detector (same as main.py)
    orb = cv2.ORB_create(max_features)
    
    # Detect and compute ORB features on entire image
    keypoints, descriptors = orb.detectAndCompute(image, None)
    
    if descriptors is None:
        descriptors = np.array([])
    
    return keypoints, descriptors


def match_with_orb_bf(query_image: np.ndarray, template_image: np.ndarray,
                     distance_threshold: float = 50.0, max_features: int = 1000) -> Tuple[int, List]:
    """
    Match fingerprints using ORB descriptors and Brute-Force matcher.
    
    Based on main.py method: get_best_matches()
    - Extract ORB features from entire image (not just minutiae locations)
    - Match using BFMatcher
    - Matching function = number of matches with distance < threshold
    
    Args:
        query_image: Query fingerprint image
        template_image: Template fingerprint image
        distance_threshold: Maximum Hamming distance for matching (default 50.0)
        max_features: Maximum number of ORB features (default 500)
        
    Returns:
        Tuple of (number of matches, list of match objects)
    """
    # Extract ORB descriptors from entire images
    query_kp, query_des = extract_orb_descriptors(query_image, max_features)
    template_kp, template_des = extract_orb_descriptors(template_image, max_features)
    
    if query_des is None or len(query_des) == 0 or template_des is None or len(template_des) == 0:
        return 0, []
    
    # Initialize BFMatcher with Hamming distance (same as main.py)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(query_des, template_des)
    
    # Ensure matches is a list
    if not isinstance(matches, list):
        matches = list(matches)
    
    # Sort matches by distance (lower is better) - same as main.py
    matches.sort(key=lambda x: x.distance, reverse=False)
    
    # Filter matches by distance threshold
    # Matching function = length of best matches to given threshold (same as main.py)
    best_matches = [m for m in matches if m.distance < distance_threshold]
    
    return len(best_matches), best_matches


def match_fingerprint(query_features: np.ndarray, template_features: np.ndarray,
                     threshold: float = 0.22,
                     query_image: np.ndarray = None,
                     template_image: np.ndarray = None,
                     query_orientation_field: np.ndarray = None,
                     template_orientation_field: np.ndarray = None,
                     query_frequency_field: np.ndarray = None,
                     template_frequency_field: np.ndarray = None,
                     use_orb: bool = False) -> Tuple[bool, float, Dict]:
    """
    Match query fingerprint against a template.
    
    Args:
        query_features: Query feature array (M×4)
        template_features: Template feature array (N×4)
        threshold: Minimum match score for positive match
        
    Returns:
        Tuple of (is_match, match_score, match_info)
    """
    if len(query_features) == 0 or len(template_features) == 0:
        return (False, 0.0, {'error': 'Empty feature set'})
    
    # Use polar coordinate matching method (original method)
    # Filter features to remove edge minutiae (more reliable)
    filtered_template = filter_minutiae(template_features)
    filtered_query = filter_minutiae(query_features)
    
    # Use filtered features for alignment if we have enough
    align_template = filtered_template if len(filtered_template) >= 10 else template_features
    align_query = filtered_query if len(filtered_query) >= 10 else query_features
    
    # Find best alignment using polar coordinate method
    rotation, tx, ty, alignment_score = align_minutiae_polar(align_template, align_query)
    
    # Check alignment quality - if alignment score is too low, likely false match
    if alignment_score < 2:  # Very few matches in alignment (more relaxed)
        return (False, 0.0, {
            'error': 'Poor alignment',
            'alignment_score': alignment_score,
            'num_query_minutiae': len(query_features),
            'num_template_minutiae': len(template_features),
            'method': 'Polar_Coordinate',
            'rotation': 0.0,
            'translation': (0.0, 0.0)
        })
    
    # Calculate detailed match score using the aligned transformation
    # Use filtered features for matching if available
    match_template = filtered_template if len(filtered_template) >= 10 else template_features
    match_query = filtered_query if len(filtered_query) >= 10 else query_features
    
    match_score = calculate_match_score(
        match_template, match_query, 
        rotation, tx, ty,
        orientation_field1=template_orientation_field,
        orientation_field2=query_orientation_field,
        frequency_field1=template_frequency_field,
        frequency_field2=query_frequency_field
    )
    
    # Additional check: if match score is below threshold, reject
    # Also check if alignment was reasonable (not too extreme transformation)
    max_reasonable_rotation = np.pi / 2  # 90 degrees (more relaxed)
    max_reasonable_translation = 250  # pixels (more relaxed)
    
    if abs(rotation) > max_reasonable_rotation or abs(tx) > max_reasonable_translation or abs(ty) > max_reasonable_translation:
        match_score *= 0.8  # Lighter penalty for extreme transformations
    
    is_match = match_score >= threshold
    
    match_info = {
        'match_score': match_score,
        'rotation': rotation,
        'translation': (tx, ty),
        'alignment_score': alignment_score,
        'num_query_minutiae': len(query_features),
        'num_template_minutiae': len(template_features),
        'method': 'Polar_Coordinate_Fallback'
    }
    
    return (is_match, match_score, match_info)


def identify_fingerprint(query_image_path: str, database: FingerprintDatabase,
                        threshold: float = 0.22,
                        block_size: int = 16, min_distance: int = 20,
                        use_thinned: bool = False, use_binarized: bool = False) -> Optional[Dict]:
    """
    Identify a fingerprint by comparing against all enrolled templates.
    
    Args:
        query_image_path: Path to query fingerprint image
        database: FingerprintDatabase instance
        threshold: Minimum match score for identification
        block_size: Block size for feature extraction
        min_distance: Minimum distance between minutiae
        use_thinned: If True, image is already thinned/skeletonized
        use_binarized: If True, image is binarized and needs thinning
        
    Returns:
        Dictionary with identification result, or None if no match found
    """
    print(f"\n{'='*70}")
    print(f"Identifying fingerprint: {os.path.basename(query_image_path)}")
    print(f"{'='*70}")
    
    try:
        # Process query image
        print("\n1. Processing query image...")
        image = load_image(query_image_path)
        
        if use_thinned:
            # Image is already thinned, use directly
            processed = image
        elif use_binarized:
            # Image is binarized, need to thin it first
            from preprocessing import skeletonize
            # Ensure image is binary (0 or 255)
            if image.max() > 1:
                binary = (image > 127).astype(np.uint8) * 255
            else:
                binary = image.astype(np.uint8) * 255
            # Thin the binarized image
            processed = skeletonize(binary)
        else:
            # Preprocess normally
            processed = preprocess_pipeline(
                image,
                normalize=True,
                denoise=True,
                enhance_contrast=True,
                enhance_ridges=False,
                binarize=False
            )
        
        # Extract all features (minutiae, orientation field, frequency field)
        print("2. Extracting features...")
        features_dict = extract_features(processed, block_size=block_size, 
                                       min_distance=min_distance,
                                       is_thinned=(use_thinned or use_binarized))
        query_minutiae = features_dict['minutiae']
        query_orientation_field = features_dict['orientation_field']
        query_frequency_field = features_dict['frequency_field']
        
        if len(query_minutiae) == 0:
            print("  ✗ No minutiae found in query image")
            return None
        
        query_features = minutiae_to_features(query_minutiae)
        print(f"  ✓ Found {len(query_minutiae)} minutiae")
        print(f"  ✓ Orientation field: {query_orientation_field.shape}")
        print(f"  ✓ Frequency field: {query_frequency_field.shape}")
        
        # Get all enrolled persons
        all_enrolled_persons = database.list_enrolled_persons()
        
        if len(all_enrolled_persons) == 0:
            print("\n✗ No enrolled persons in database")
            return None
        
        # Compare against all enrolled persons
        enrolled_persons = all_enrolled_persons
        print(f"\n3. Comparing against {len(enrolled_persons)} enrolled templates...")
        
        best_match = None
        best_score = 0.0
        
        for person_id in enrolled_persons:
            template_data = database.load_template(person_id)
            if template_data is None:
                continue
            
            template_features = template_data['features']
            template_orientation = template_data.get('orientation_field')
            template_frequency = template_data.get('frequency_field')
            
            # Match against this template with all features
            is_match, score, match_info = match_fingerprint(
                query_features, template_features, 
                threshold=threshold,
                query_orientation_field=query_orientation_field,
                template_orientation_field=template_orientation,
                query_frequency_field=query_frequency_field,
                template_frequency_field=template_frequency
            )
            
            print(f"  Person {person_id}: score = {score:.3f}", end="")
            if is_match:
                print(" ✓ MATCH")
            else:
                print()
            
            if score > best_score:
                best_score = score
                best_match = {
                    'person_id': person_id,
                    'match_score': score,
                    'match_info': match_info,
                    'is_match': is_match
                }
        
        print(f"\n{'='*70}")
        if best_match and best_match['is_match']:
            print(f"✓ IDENTIFIED: Person {best_match['person_id']}")
            print(f"  Match score: {best_match['match_score']:.3f}")
            return best_match
        else:
            print("✗ NO MATCH FOUND")
            if best_match:
                print(f"  Best score: {best_match['match_score']:.3f} (threshold: {threshold})")
            return None
            
    except Exception as e:
        print(f"\n✗ Error during identification: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def enroll_from_directory(train_dir: str, database: FingerprintDatabase,
                         person_ids: List[str] = None,
                         block_size: int = 16, min_distance: int = 20,
                         use_thinned: bool = False, use_binarized: bool = False):
    """
    Enroll multiple persons from training directory.
    
    Args:
        train_dir: Directory containing training images
        database: FingerprintDatabase instance
        person_ids: List of person IDs to enroll (None = enroll all)
        block_size: Block size for feature extraction
        min_distance: Minimum distance between minutiae
        use_thinned: If True, images are already thinned/skeletonized
        use_binarized: If True, images are binarized and need thinning
    """
    print("\n" + "="*70)
    print("ENROLLMENT FROM TRAINING DIRECTORY")
    print("="*70)
    
    if not os.path.exists(train_dir):
        print(f"Error: Training directory not found: {train_dir}")
        return
    
    # Get images by person
    images_by_person = get_images_by_person(train_dir)
    
    if person_ids is None:
        # Default: enroll all persons
        person_ids = sorted(images_by_person.keys(), key=lambda x: int(x))
        print(f"\nEnrolling all {len(person_ids)} persons from training directory")
    else:
        print(f"\nEnrolling specified {len(person_ids)} persons")
    
    print(f"Person IDs to enroll: {person_ids[:10]}..." if len(person_ids) > 10 else f"Person IDs to enroll: {person_ids}")
    
    enrolled_count = 0
    for person_id in person_ids:
        if person_id not in images_by_person:
            print(f"\n⚠ Person {person_id} not found in training directory")
            continue
        
        image_files = images_by_person[person_id]
        image_paths = [os.path.join(train_dir, f) for f in image_files]
        
        success = enroll_person(person_id, image_paths, database, 
                               block_size=block_size, min_distance=min_distance,
                               use_thinned=use_thinned, use_binarized=use_binarized)
        
        if success:
            enrolled_count += 1
    
    print("\n" + "="*70)
    print(f"ENROLLMENT COMPLETE: {enrolled_count}/{len(person_ids)} persons enrolled")
    print("="*70)


if __name__ == "__main__":
    # Example usage
    print("Fingerprint Enrollment and Matching System")
    print("="*70)
    
    # Initialize database
    db = FingerprintDatabase()
    
    # Example: Enroll from training directory
    train_dir = "project-data/Project-Data/train"
    if os.path.exists(train_dir):
        enroll_from_directory(train_dir, db, person_ids=None)
    else:
        print(f"Training directory not found: {train_dir}")
        print("Please ensure the dataset is in the correct location.")

