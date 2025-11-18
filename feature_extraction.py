"""
Fingerprint Feature Extraction Module

This module extracts features from preprocessed fingerprint images:
- Ridge orientation field
- Ridge frequency
- Minutiae points (ridge endings and bifurcations)
- Feature representation for matching
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from typing import List, Tuple, Dict, Optional
import math

# Add visualization function to the module
def visualize_features(image: np.ndarray, features: dict, output_path: str = None):
    """
    Visualize extracted features.
    
    Args:
        image: Original preprocessed image
        features: Dictionary from extract_features()
        output_path: Path to save visualization
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Preprocessed Image')
    axes[0, 0].axis('off')
    
    # Orientation field
    orientation = features['orientation_field']
    h, w = orientation.shape
    y, x = np.meshgrid(np.arange(0, h, 8), np.arange(0, w, 8), indexing='ij')
    u = np.cos(orientation[::8, ::8])
    v = np.sin(orientation[::8, ::8])
    
    axes[0, 1].imshow(image, cmap='gray', alpha=0.5)
    axes[0, 1].quiver(x, y, u, v, scale=20, width=0.002, color='red')
    axes[0, 1].set_title('Ridge Orientation Field')
    axes[0, 1].axis('off')
    
    # Enhanced image
    axes[0, 2].imshow(features['enhanced_image'], cmap='gray')
    axes[0, 2].set_title('Gabor-Enhanced Image')
    axes[0, 2].axis('off')
    
    # Binary image
    axes[1, 0].imshow(features['binary_image'], cmap='gray')
    axes[1, 0].set_title('Binarized Image')
    axes[1, 0].axis('off')
    
    # Skeleton
    axes[1, 1].imshow(features['skeleton'], cmap='gray')
    axes[1, 1].set_title('Skeletonized Image')
    axes[1, 1].axis('off')
    
    # Minutiae points
    axes[1, 2].imshow(image, cmap='gray')
    minutiae = features['minutiae']
    
    endings = [m for m in minutiae if m.type == 'ending']
    bifurcations = [m for m in minutiae if m.type == 'bifurcation']
    
    if endings:
        x_end = [m.x for m in endings]
        y_end = [m.y for m in endings]
        axes[1, 2].scatter(x_end, y_end, c='green', marker='o', s=50, 
                          label=f'Endings ({len(endings)})', alpha=0.7)
    
    if bifurcations:
        x_bif = [m.x for m in bifurcations]
        y_bif = [m.y for m in bifurcations]
        axes[1, 2].scatter(x_bif, y_bif, c='red', marker='s', s=50,
                          label=f'Bifurcations ({len(bifurcations)})', alpha=0.7)
    
    axes[1, 2].set_title(f'Minutiae Points (Total: {len(minutiae)})')
    axes[1, 2].legend()
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


class Minutiae:
    """Represents a minutiae point."""
    def __init__(self, x: int, y: int, orientation: float, type: str):
        """
        Args:
            x: X coordinate
            y: Y coordinate
            orientation: Ridge orientation at this point (in radians)
            type: 'ending' or 'bifurcation'
        """
        self.x = x
        self.y = y
        self.orientation = orientation
        self.type = type  # 'ending' or 'bifurcation'
    
    def __repr__(self):
        return f"Minutiae({self.type}, x={self.x}, y={self.y}, orient={self.orientation:.2f})"


def estimate_orientation_field(image: np.ndarray, block_size: int = 16, 
                               smooth: bool = True) -> np.ndarray:
    """
    Estimate ridge orientation field using gradient-based method.
    
    Args:
        image: Preprocessed grayscale fingerprint image
        block_size: Size of blocks for orientation estimation
        smooth: Whether to smooth the orientation field
        
    Returns:
        Orientation field in radians (same size as input image)
    """
    h, w = image.shape
    
    # Calculate gradients
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Initialize orientation field
    orientation_field = np.zeros((h, w))
    
    # Estimate orientation for each block
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block_gx = gx[i:i+block_size, j:j+block_size]
            block_gy = gy[i:i+block_size, j:j+block_size]
            
            # Calculate orientation using least squares method
            vx = np.sum(2 * block_gx * block_gy)
            vy = np.sum(block_gx**2 - block_gy**2)
            
            # Orientation is perpendicular to gradient direction
            orientation = 0.5 * np.arctan2(vx, vy)
            
            # Assign orientation to all pixels in block
            orientation_field[i:i+block_size, j:j+block_size] = orientation
    
    # Handle remaining pixels
    if h % block_size != 0:
        for i in range((h // block_size) * block_size, h):
            for j in range(0, w - block_size, block_size):
                block_gx = gx[i:i+block_size, j:j+block_size]
                block_gy = gy[i:i+block_size, j:j+block_size]
                vx = np.sum(2 * block_gx * block_gy)
                vy = np.sum(block_gx**2 - block_gy**2)
                orientation = 0.5 * np.arctan2(vx, vy)
                orientation_field[i:i+block_size, j:j+block_size] = orientation
    
    if w % block_size != 0:
        for i in range(0, h - block_size, block_size):
            for j in range((w // block_size) * block_size, w):
                block_gx = gx[i:i+block_size, j:j+block_size]
                block_gy = gy[i:i+block_size, j:j+block_size]
                vx = np.sum(2 * block_gx * block_gy)
                vy = np.sum(block_gx**2 - block_gy**2)
                orientation = 0.5 * np.arctan2(vx, vy)
                orientation_field[i:i+block_size, j:j+block_size] = orientation
    
    # Smooth orientation field
    if smooth:
        # Convert to complex representation for smoothing
        orientation_complex = np.exp(2j * orientation_field)
        orientation_complex = gaussian_filter(orientation_complex, sigma=2.0)
        orientation_field = np.angle(orientation_complex) / 2.0
    
    return orientation_field


def estimate_ridge_frequency(image: np.ndarray, orientation_field: np.ndarray,
                             block_size: int = 32, window_size: int = 5) -> np.ndarray:
    """
    Estimate ridge frequency (spacing between ridges) using FFT.
    
    Args:
        image: Preprocessed grayscale fingerprint image
        orientation_field: Ridge orientation field
        block_size: Size of blocks for frequency estimation
        window_size: Window size for frequency calculation
        
    Returns:
        Frequency field (inverse of ridge spacing)
    """
    h, w = image.shape
    frequency_field = np.zeros((h, w))
    
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            # Get block
            block = image[i:i+block_size, j:j+block_size]
            block_orientation = orientation_field[i+block_size//2, j+block_size//2]
            
            # Rotate block to align ridges horizontally
            rotation_angle = -block_orientation * 180 / np.pi
            center = (block_size // 2, block_size // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            rotated_block = cv2.warpAffine(block, rotation_matrix, (block_size, block_size))
            
            # Project to get ridge profile
            profile = np.mean(rotated_block, axis=1)
            
            # Estimate frequency using FFT
            fft = np.fft.fft(profile)
            fft_magnitude = np.abs(fft[1:block_size//2])
            
            # Find dominant frequency
            if len(fft_magnitude) > 0:
                max_idx = np.argmax(fft_magnitude)
                frequency = (max_idx + 1) / block_size
                # Convert to pixels per ridge (typical range: 3-25 pixels)
                frequency = np.clip(frequency, 1/25, 1/3)
            else:
                frequency = 1/10  # Default frequency
            
            # Assign frequency to all pixels in block
            frequency_field[i:i+block_size, j:j+block_size] = frequency
    
    return frequency_field


def create_gabor_filter_bank(num_orientations: int = 8, 
                             frequencies: List[float] = None) -> List[np.ndarray]:
    """
    Create a bank of Gabor filters for ridge enhancement.
    
    Args:
        num_orientations: Number of orientation angles
        frequencies: List of frequencies to use (default: [0.1])
        
    Returns:
        List of Gabor filter kernels
    """
    if frequencies is None:
        frequencies = [0.1]
    
    filters = []
    orientations = np.linspace(0, np.pi, num_orientations, endpoint=False)
    
    for orientation in orientations:
        for frequency in frequencies:
            kernel = cv2.getGaborKernel(
                (21, 21),
                4.0,  # sigma_x
                orientation,
                2 * np.pi * frequency,
                0.5,  # gamma
                0,    # psi
                ktype=cv2.CV_32F
            )
            filters.append((kernel, orientation, frequency))
    
    return filters


def enhance_ridges_gabor(image: np.ndarray, orientation_field: np.ndarray,
                         frequency_field: np.ndarray, block_size: int = 16) -> np.ndarray:
    """
    Enhance ridges using orientation-adaptive Gabor filtering.
    
    Args:
        image: Preprocessed grayscale fingerprint image
        orientation_field: Ridge orientation field
        frequency_field: Ridge frequency field
        block_size: Block size for filtering
        
    Returns:
        Enhanced image with emphasized ridges
    """
    h, w = image.shape
    enhanced = np.zeros_like(image, dtype=np.float32)
    
    # Convert image to float
    img_float = image.astype(np.float32) / 255.0
    
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = img_float[i:i+block_size, j:j+block_size]
            orientation = orientation_field[i+block_size//2, j+block_size//2]
            frequency = frequency_field[i+block_size//2, j+block_size//2]
            
            # Create Gabor filter for this block
            kernel = cv2.getGaborKernel(
                (21, 21),
                4.0,
                orientation,
                2 * np.pi * frequency,
                0.5,
                0,
                ktype=cv2.CV_32F
            )
            
            # Apply filter
            filtered = cv2.filter2D(block, cv2.CV_32F, kernel)
            
            # Store result
            enhanced[i:i+block_size, j:j+block_size] = filtered
    
    # Normalize to [0, 255]
    enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
    
    return enhanced


def binarize_adaptive(image: np.ndarray, orientation_field: np.ndarray,
                      block_size: int = 16) -> np.ndarray:
    """
    Binarize image using orientation-adaptive thresholding.
    
    Args:
        image: Preprocessed grayscale fingerprint image
        orientation_field: Ridge orientation field
        block_size: Block size for adaptive thresholding
        
    Returns:
        Binary image (0 = valley, 255 = ridge)
    """
    h, w = image.shape
    binary = np.zeros((h, w), dtype=np.uint8)
    
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = image[i:i+block_size, j:j+block_size]
            threshold = np.mean(block)
            binary[i:i+block_size, j:j+block_size] = (block > threshold).astype(np.uint8) * 255
    
    return binary


def skeletonize(binary_image: np.ndarray) -> np.ndarray:
    """
    Skeletonize binary image to get 1-pixel wide ridges.
    Uses Zhang-Suen thinning algorithm for clean lines.
    
    Args:
        binary_image: Binary fingerprint image (255 = ridge, 0 = valley)
        
    Returns:
        Skeletonized image (clean 1-pixel wide lines)
    """
    # Import from preprocessing to use the same algorithm
    from preprocessing import skeletonize as preprocess_skeletonize
    return preprocess_skeletonize(binary_image)


def has_adjacent_ridges_on_both_sides(binary: np.ndarray, i: int, j: int, 
                                     orientation: float, check_distance: int = 5) -> bool:
    """
    Check if a minutiae point has adjacent ridges on both sides.
    
    Based on the heuristic: "minutiae that do not have an adjacent ridge on either side
    (mainly the endpoints of ridges along the finger border)" should be removed.
    
    Args:
        binary: Binary skeleton image
        i, j: Pixel coordinates
        orientation: Ridge orientation at this point
        check_distance: Distance to check for adjacent ridges
        
    Returns:
        True if ridges exist on both sides, False otherwise
    """
    h, w = binary.shape
    
    # Calculate perpendicular direction to the ridge
    perp_angle = orientation + np.pi / 2
    
    # Check on both sides perpendicular to the ridge
    cos_perp = np.cos(perp_angle)
    sin_perp = np.sin(perp_angle)
    
    # Check left side
    left_has_ridge = False
    for d in range(1, check_distance + 1):
        check_i = int(i + d * sin_perp)
        check_j = int(j + d * cos_perp)
        
        if 0 <= check_i < h and 0 <= check_j < w:
            if binary[check_i, check_j] > 0:
                left_has_ridge = True
                break
    
    # Check right side
    right_has_ridge = False
    for d in range(1, check_distance + 1):
        check_i = int(i - d * sin_perp)
        check_j = int(j - d * cos_perp)
        
        if 0 <= check_i < h and 0 <= check_j < w:
            if binary[check_i, check_j] > 0:
                right_has_ridge = True
                break
    
    # Return True only if ridges exist on BOTH sides
    return left_has_ridge and right_has_ridge


def extract_minutiae(skeleton: np.ndarray, orientation_field: np.ndarray,
                    min_distance: int = 20,
                    remove_border_false_positives: bool = True) -> List[Minutiae]:
    """
    Extract minutiae points from skeletonized image using Crossing Number method.
    
    Crossing Number is defined as half of the sum of differences between intensity 
    values of two adjacent pixels (in the 8-neighborhood).
    
    Classification:
    - Crossing Number = 1: Termination (ridge ending)
    - Crossing Number = 2: Normal ridge pixel (not a minutiae)
    - Crossing Number = 3 or >3: Bifurcation
    
    Note: Terminations at outer boundaries are not considered minutiae points.
    
    Args:
        skeleton: Skeletonized binary image (thinned, 1-pixel wide ridges)
        orientation_field: Ridge orientation field
        min_distance: Minimum distance between minutiae points
        remove_border_false_positives: If True, remove minutiae without adjacent ridges on both sides
        
    Returns:
        List of Minutiae objects (with false positives removed)
    """
    # Convert to binary (0 or 1)
    binary = (skeleton > 0).astype(np.uint8)
    
    h, w = binary.shape
    minutiae = []
    
    # 8-connected neighbors in clockwise order
    # Starting from top-left, going clockwise
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                (0, 1),   (1, 1),  (1, 0),
                (1, -1),  (0, -1)]
    
    # Also include the first neighbor at the end to form a closed loop
    neighbors_loop = neighbors + [neighbors[0]]
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if binary[i, j] == 0:  # Skip background pixels
                continue
            
            # Calculate Crossing Number
            # CN = 0.5 * sum of |P_i - P_{i+1}| for i=0 to 7
            # where P_i are the 8 neighbors in order
            crossing_number = 0
            neighbor_values = []
            
            # Get neighbor values in order (as integers, not uint8 to avoid overflow)
            for di, dj in neighbors:
                neighbor_values.append(int(binary[i + di, j + dj]))
            
            # Calculate sum of differences between adjacent neighbors
            # Add the last-to-first difference to close the loop
            for idx in range(len(neighbors)):
                next_idx = (idx + 1) % len(neighbors)
                diff = abs(neighbor_values[idx] - neighbor_values[next_idx])
                crossing_number += diff
            
            # Crossing Number = half of the sum
            crossing_number = crossing_number // 2
            
            # Classify based on Crossing Number
            orientation = orientation_field[i, j]
            
            if crossing_number == 1:
                # Termination (ridge ending)
                m = Minutiae(j, i, orientation, 'ending')
            elif crossing_number >= 3:
                # Bifurcation (or higher order junction)
                m = Minutiae(j, i, orientation, 'bifurcation')
            else:
                # Crossing Number = 2: Normal ridge pixel, not a minutiae
                continue
            
            # Apply heuristic: remove if no adjacent ridges on both sides
            # This removes false positives at finger borders
            if remove_border_false_positives:
                if not has_adjacent_ridges_on_both_sides(binary, i, j, orientation):
                    continue  # Skip this false positive
            
            minutiae.append(m)
    
    # Remove minutiae that are too close to each other
    # Use a more robust filtering method that handles dense regions better
    filtered_minutiae = _filter_minutiae_by_distance(minutiae, min_distance)
    
    return filtered_minutiae


def _filter_minutiae_by_distance(minutiae: List[Minutiae], min_distance: float) -> List[Minutiae]:
    """
    Filter minutiae to ensure minimum distance between points.
    
    Uses a greedy approach with improved handling of dense regions:
    1. Sort by quality (endings preferred, then by position)
    2. For each minutia, check distance to all already kept minutiae
    3. In dense regions, only the first (highest quality) minutia is kept
    
    This helps reduce dense clusters while preserving important minutiae.
    
    Args:
        minutiae: List of Minutiae objects
        min_distance: Minimum distance between minutiae points (in pixels)
        
    Returns:
        Filtered list of Minutiae objects
    """
    if len(minutiae) == 0:
        return []
    
    # Sort minutiae by quality:
    # 1. Endings are generally more reliable than bifurcations
    # 2. Then by position (for deterministic ordering)
    def sort_key(m: Minutiae) -> Tuple[int, int, int]:
        # Prefer endings (type 'ending' = 0) over bifurcations (type 'bifurcation' = 1)
        type_priority = 0 if m.type == 'ending' else 1
        return (type_priority, m.y, m.x)
    
    sorted_minutiae = sorted(minutiae, key=sort_key)
    
    filtered = []
    min_distance_sq = min_distance * min_distance  # Use squared distance for efficiency
    
    for m in sorted_minutiae:
        too_close = False
        for existing in filtered:
            # Calculate squared distance (avoid sqrt for efficiency)
            dx = m.x - existing.x
            dy = m.y - existing.y
            dist_sq = dx * dx + dy * dy
            
            if dist_sq < min_distance_sq:
                too_close = True
                break
        
        if not too_close:
            filtered.append(m)
    
    return filtered


def extract_features(image: np.ndarray, 
                    block_size: int = 16,
                    min_distance: int = 20,
                    is_thinned: bool = False,
                    remove_border_false_positives: bool = True) -> Dict:
    """
    Complete feature extraction pipeline.
    
    Args:
        image: Preprocessed grayscale fingerprint image (or thinned image if is_thinned=True)
        block_size: Block size for orientation/frequency estimation
        min_distance: Minimum distance between minutiae points
        is_thinned: If True, image is already thinned/skeletonized (skip steps 3-5)
        remove_border_false_positives: If True, remove minutiae without adjacent ridges on both sides
        
    Returns:
        Dictionary containing:
            - orientation_field: Ridge orientation field
            - frequency_field: Ridge frequency field
            - enhanced_image: Gabor-enhanced image (or None if is_thinned)
            - binary_image: Binarized image (or thinned image if is_thinned)
            - skeleton: Skeletonized image (same as input if is_thinned)
            - minutiae: List of Minutiae objects (with false positives removed)
    """
    # Step 1: Estimate orientation field (always needed)
    orientation_field = estimate_orientation_field(image, block_size)
    
    # Step 2: Estimate ridge frequency (always needed)
    frequency_field = estimate_ridge_frequency(image, orientation_field, block_size)
    
    if is_thinned:
        # Image is already thinned, skip enhancement, binarization, and skeletonization
        skeleton = image.copy()
        binary_image = image.copy()
        enhanced_image = None
    else:
        # Step 3: Enhance ridges using Gabor filters
        enhanced_image = enhance_ridges_gabor(image, orientation_field, frequency_field, block_size)
        
        # Step 4: Binarize
        binary_image = binarize_adaptive(enhanced_image, orientation_field, block_size)
        
        # Step 5: Skeletonize
        skeleton = skeletonize(binary_image)
    
    # Step 6: Extract minutiae (always needed, with false positive removal)
    minutiae = extract_minutiae(skeleton, orientation_field, min_distance, remove_border_false_positives)
    
    return {
        'orientation_field': orientation_field,
        'frequency_field': frequency_field,
        'enhanced_image': enhanced_image,
        'binary_image': binary_image,
        'skeleton': skeleton,
        'minutiae': minutiae
    }


def minutiae_to_features(minutiae_list: List[Minutiae]) -> np.ndarray:
    """
    Convert minutiae list to feature vector for matching.
    
    Args:
        minutiae_list: List of Minutiae objects
        
    Returns:
        Feature array: [x, y, orientation, type] for each minutiae
    """
    if len(minutiae_list) == 0:
        return np.array([]).reshape(0, 4)
    
    features = []
    for m in minutiae_list:
        type_code = 1 if m.type == 'ending' else 2  # 1=ending, 2=bifurcation
        features.append([m.x, m.y, m.orientation, type_code])
    
    return np.array(features)

