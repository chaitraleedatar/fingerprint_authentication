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
    Uses morphological operations for thinning.
    
    Args:
        binary_image: Binary fingerprint image (255 = ridge, 0 = valley)
        
    Returns:
        Skeletonized image
    """
    # Convert to binary (0 or 1)
    skeleton = (binary_image > 127).astype(np.uint8)
    
    # Use morphological thinning with hit-or-miss transform
    # This is a simplified version - for better results, use Zhang-Suen algorithm
    kernel = np.ones((3, 3), np.uint8)
    
    # Iterative thinning
    prev = np.zeros_like(skeleton)
    while not np.array_equal(skeleton, prev):
        prev = skeleton.copy()
        
        # Erode with small kernel
        skeleton = cv2.erode(skeleton, kernel, iterations=1)
        
        # Reconstruct using dilation
        marker = skeleton.copy()
        mask = (binary_image > 127).astype(np.uint8)
        for _ in range(10):  # Limited iterations
            marker = cv2.dilate(marker, kernel, iterations=1)
            marker = cv2.bitwise_and(marker, mask)
            if np.array_equal(marker, skeleton):
                break
            skeleton = marker
    
    return skeleton * 255


def extract_minutiae(skeleton: np.ndarray, orientation_field: np.ndarray,
                    min_distance: int = 10) -> List[Minutiae]:
    """
    Extract minutiae points from skeletonized image.
    
    Args:
        skeleton: Skeletonized binary image
        orientation_field: Ridge orientation field
        min_distance: Minimum distance between minutiae points
        
    Returns:
        List of Minutiae objects
    """
    # Convert to binary (0 or 1)
    binary = (skeleton > 0).astype(np.uint8)
    
    # Use 3x3 cross pattern to detect minutiae
    # Ridge ending: pixel with exactly 1 neighbor
    # Bifurcation: pixel with exactly 3 neighbors
    
    h, w = binary.shape
    minutiae = []
    
    # 8-connected neighbors
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1)]
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if binary[i, j] == 0:  # Skip background
                continue
            
            # Count neighbors
            neighbor_count = 0
            for di, dj in neighbors:
                if binary[i + di, j + dj] > 0:
                    neighbor_count += 1
            
            # Detect minutiae
            if neighbor_count == 1:
                # Ridge ending
                orientation = orientation_field[i, j]
                minutiae.append(Minutiae(j, i, orientation, 'ending'))
            elif neighbor_count == 3:
                # Bifurcation
                orientation = orientation_field[i, j]
                minutiae.append(Minutiae(j, i, orientation, 'bifurcation'))
    
    # Remove minutiae that are too close to each other
    filtered_minutiae = []
    for m in minutiae:
        too_close = False
        for existing in filtered_minutiae:
            distance = np.sqrt((m.x - existing.x)**2 + (m.y - existing.y)**2)
            if distance < min_distance:
                too_close = True
                break
        if not too_close:
            filtered_minutiae.append(m)
    
    return filtered_minutiae


def extract_features(image: np.ndarray, 
                    block_size: int = 16,
                    min_distance: int = 10) -> Dict:
    """
    Complete feature extraction pipeline.
    
    Args:
        image: Preprocessed grayscale fingerprint image
        block_size: Block size for orientation/frequency estimation
        min_distance: Minimum distance between minutiae points
        
    Returns:
        Dictionary containing:
            - orientation_field: Ridge orientation field
            - frequency_field: Ridge frequency field
            - enhanced_image: Gabor-enhanced image
            - binary_image: Binarized image
            - skeleton: Skeletonized image
            - minutiae: List of Minutiae objects
    """
    # Step 1: Estimate orientation field
    orientation_field = estimate_orientation_field(image, block_size)
    
    # Step 2: Estimate ridge frequency
    frequency_field = estimate_ridge_frequency(image, orientation_field, block_size)
    
    # Step 3: Enhance ridges using Gabor filters
    enhanced_image = enhance_ridges_gabor(image, orientation_field, frequency_field, block_size)
    
    # Step 4: Binarize
    binary_image = binarize_adaptive(enhanced_image, orientation_field, block_size)
    
    # Step 5: Skeletonize
    skeleton = skeletonize(binary_image)
    
    # Step 6: Extract minutiae
    minutiae = extract_minutiae(skeleton, orientation_field, min_distance)
    
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

