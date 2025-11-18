"""
Fingerprint Image Preprocessing Module

This module implements various preprocessing techniques for fingerprint images:
- Noise reduction (Gaussian blur, median filtering)
- Image enhancement (contrast enhancement, histogram equalization)
- Normalization (intensity normalization)
- Ridge enhancement (Gabor filtering)
- Binarization (optional)
- Thinning/Skeletonization (extract 1-pixel wide ridges)
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional
import os


def load_image(image_path: str) -> np.ndarray:
    """
    Load a fingerprint image from file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Grayscale image as numpy array
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    return image


def normalize_image(image: np.ndarray, target_mean: float = 127.5, 
                   target_std: float = 127.5) -> np.ndarray:
    """
    Normalize image intensity to have specified mean and standard deviation.
    
    Args:
        image: Input grayscale image
        target_mean: Target mean value (default: 127.5)
        target_std: Target standard deviation (default: 127.5)
        
    Returns:
        Normalized image
    """
    # Calculate current mean and std
    current_mean = np.mean(image)
    current_std = np.std(image)
    
    # Avoid division by zero
    if current_std == 0:
        return image.astype(np.float32)
    
    # Normalize
    normalized = (image - current_mean) / current_std * target_std + target_mean
    
    # Clip values to valid range [0, 255]
    normalized = np.clip(normalized, 0, 255)
    
    return normalized.astype(np.uint8)


def reduce_noise_gaussian(image: np.ndarray, kernel_size: int = 5, 
                         sigma: float = 1.0) -> np.ndarray:
    """
    Reduce noise using Gaussian blur.
    
    Args:
        image: Input grayscale image
        kernel_size: Size of the Gaussian kernel (must be odd)
        sigma: Standard deviation of the Gaussian kernel
        
    Returns:
        Denoised image
    """
    # Ensure kernel_size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return blurred


def reduce_noise_median(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Reduce noise using median filtering.
    
    Args:
        image: Input grayscale image
        kernel_size: Size of the median filter kernel (must be odd)
        
    Returns:
        Denoised image
    """
    # Ensure kernel_size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    filtered = cv2.medianBlur(image, kernel_size)
    return filtered


def enhance_contrast_clahe(image: np.ndarray, clip_limit: float = 2.0, 
                          tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Enhance contrast using Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Args:
        image: Input grayscale image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        Contrast-enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image)
    return enhanced


def enhance_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Enhance image using histogram equalization.
    
    Args:
        image: Input grayscale image
        
    Returns:
        Histogram-equalized image
    """
    equalized = cv2.equalizeHist(image)
    return equalized


def enhance_ridge_orientation(image: np.ndarray, block_size: int = 16) -> np.ndarray:
    """
    Estimate ridge orientation field (for visualization/debugging).
    
    Args:
        image: Input grayscale image
        block_size: Size of blocks for orientation estimation
        
    Returns:
        Orientation field (in radians)
    """
    # Calculate gradients
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate orientation
    orientation = np.arctan2(gy, gx) / 2.0
    
    # Smooth orientation field
    h, w = image.shape
    orientation_field = np.zeros((h // block_size, w // block_size))
    
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block_gx = gx[i:i+block_size, j:j+block_size]
            block_gy = gy[i:i+block_size, j:j+block_size]
            
            # Calculate dominant orientation
            vx = np.sum(2 * block_gx * block_gy)
            vy = np.sum(block_gx**2 - block_gy**2)
            orientation_field[i//block_size, j//block_size] = np.arctan2(vx, vy) / 2.0
    
    return orientation_field


def enhance_gabor_filter(image: np.ndarray, frequency: float = 0.1, 
                        orientation: float = 0, sigma_x: float = 4.0, 
                        sigma_y: float = 4.0) -> np.ndarray:
    """
    Enhance ridges using Gabor filter.
    
    Args:
        image: Input grayscale image
        frequency: Spatial frequency of the filter
        orientation: Orientation of the filter (in radians)
        sigma_x: Standard deviation in x-direction
        sigma_y: Standard deviation in y-direction
        
    Returns:
        Gabor-filtered image
    """
    # Convert to float
    img_float = image.astype(np.float32) / 255.0
    
    # Create Gabor kernel
    kernel = cv2.getGaborKernel(
        (21, 21), 
        sigma_x, 
        orientation, 
        2 * np.pi * frequency, 
        0.5, 
        0, 
        ktype=cv2.CV_32F
    )
    
    # Apply filter
    filtered = cv2.filter2D(img_float, cv2.CV_32F, kernel)
    
    # Normalize back to [0, 255]
    filtered = np.clip(filtered * 255, 0, 255).astype(np.uint8)
    
    return filtered


def binarize_image(image: np.ndarray, method: str = 'adaptive', 
                  block_size: int = 11, c: int = 2) -> np.ndarray:
    """
    Binarize the fingerprint image.
    
    Args:
        image: Input grayscale image
        method: Binarization method ('adaptive', 'otsu', or 'threshold')
        block_size: Block size for adaptive thresholding (must be odd)
        c: Constant subtracted from mean for adaptive thresholding
        
    Returns:
        Binary image (0 or 255)
    """
    if method == 'adaptive':
        # Ensure block_size is odd
        if block_size % 2 == 0:
            block_size += 1
        
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, block_size, c
        )
    elif method == 'otsu':
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:  # simple threshold
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    return binary


def skeletonize(binary_image: np.ndarray) -> np.ndarray:
    """
    Skeletonize binary image to get clean 1-pixel wide ridges.
    
    Uses scikit-image's skeletonize function which implements Zhang-Suen algorithm.
    This produces clean, reliable thinning without circles or artifacts.
    
    Args:
        binary_image: Binary fingerprint image (255 = ridge, 0 = valley)
        
    Returns:
        Skeletonized/thinned image (255 = ridge, 0 = background)
    """
    try:
        from skimage.morphology import skeletonize as sk_skeletonize
    except ImportError:
        raise ImportError("scikit-image is required. Install with: pip install scikit-image")
    
    # Ensure ridges are white (255) and background is black (0)
    if np.mean(binary_image) < 127:
        binary_image = 255 - binary_image
    
    # Convert to boolean (True = ridge, False = valley)
    # scikit-image expects True for foreground (ridges)
    binary = (binary_image > 127).astype(bool)
    
    # Use scikit-image's skeletonize (Zhang-Suen algorithm)
    skeleton = sk_skeletonize(binary)
    
    # Very light cleanup: Only remove isolated single pixels (no neighbors)
    skeleton = _remove_isolated_pixels(skeleton)
    
    # Convert back to uint8 (255 = ridge, 0 = background)
    return (skeleton.astype(np.uint8) * 255)


def _disconnect_junctions(skeleton: np.ndarray) -> np.ndarray:
    """
    Disconnect unwanted connections at junction points.
    Removes pixels that have 3 or more neighbors (junctions) to break connections.
    
    Args:
        skeleton: Boolean skeleton image
        
    Returns:
        Skeleton with some junctions disconnected
    """
    cleaned = skeleton.copy()
    h, w = skeleton.shape
    
    # Find junction points (pixels with 3+ neighbors) and remove some to disconnect
    for i in range(2, h - 2):
        for j in range(2, w - 2):
            if not cleaned[i, j]:
                continue
            
            # Count 8-connected neighbors
            neighbors = [
                cleaned[i-1, j-1], cleaned[i-1, j], cleaned[i-1, j+1],
                cleaned[i, j-1],                     cleaned[i, j+1],
                cleaned[i+1, j-1], cleaned[i+1, j], cleaned[i+1, j+1]
            ]
            neighbor_count = sum(neighbors)
            
            # If it's a junction (3+ neighbors), check if it should be disconnected
            # Remove if it has 4 or more neighbors (likely unwanted connection)
            if neighbor_count >= 4:
                cleaned[i, j] = False
    
    return cleaned


def _remove_isolated_pixels(skeleton: np.ndarray) -> np.ndarray:
    """
    Remove only completely isolated pixels (no neighbors at all).
    This is a very gentle cleanup that only removes scattered noise points.
    
    Args:
        skeleton: Boolean skeleton image
        
    Returns:
        Cleaned skeleton with isolated pixels removed
    """
    cleaned = skeleton.copy()
    h, w = skeleton.shape
    
    # Remove pixels with zero neighbors (completely isolated)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if not cleaned[i, j]:
                continue
            
            # Count 8-connected neighbors
            neighbors = [
                cleaned[i-1, j-1], cleaned[i-1, j], cleaned[i-1, j+1],
                cleaned[i, j-1],                     cleaned[i, j+1],
                cleaned[i+1, j-1], cleaned[i+1, j], cleaned[i+1, j+1]
            ]
            neighbor_count = sum(neighbors)
            
            # Remove only if completely isolated (0 neighbors)
            if neighbor_count == 0:
                cleaned[i, j] = False
    
    return cleaned


def preprocess_pipeline(image: np.ndarray,
                       normalize: bool = True,
                       denoise: bool = True,
                       enhance_contrast: bool = True,
                       enhance_ridges: bool = False,
                       binarize: bool = False,
                       thin: bool = False,
                       **kwargs) -> np.ndarray:
    """
    Complete preprocessing pipeline for fingerprint images.
    
    Args:
        image: Input grayscale image
        normalize: Whether to normalize the image
        denoise: Whether to apply denoising (uses median filter)
        enhance_contrast: Whether to enhance contrast (uses CLAHE)
        enhance_ridges: Whether to enhance ridges using Gabor filter
        binarize: Whether to binarize the image
        thin: Whether to apply thinning/skeletonization (requires binarize=True)
             This creates 1-pixel wide ridges (line-only image)
        **kwargs: Additional parameters for specific preprocessing steps
        
    Returns:
        Preprocessed image (if thin=True, returns skeletonized/thinned image)
    """
    processed = image.copy()
    
    # Step 1: Normalization
    if normalize:
        processed = normalize_image(processed)
    
    # Step 2: Noise reduction
    if denoise:
        denoise_method = kwargs.get('denoise_method', 'median')
        if denoise_method == 'median':
            kernel_size = kwargs.get('denoise_kernel_size', 5)
            processed = reduce_noise_median(processed, kernel_size)
        elif denoise_method == 'gaussian':
            kernel_size = kwargs.get('denoise_kernel_size', 5)
            sigma = kwargs.get('denoise_sigma', 1.0)
            processed = reduce_noise_gaussian(processed, kernel_size, sigma)
    
    # Step 3: Contrast enhancement
    if enhance_contrast:
        clip_limit = kwargs.get('clahe_clip_limit', 2.0)
        tile_size = kwargs.get('clahe_tile_size', (8, 8))
        processed = enhance_contrast_clahe(processed, clip_limit, tile_size)
    
    # Step 4: Ridge enhancement (optional)
    if enhance_ridges:
        frequency = kwargs.get('gabor_frequency', 0.1)
        orientation = kwargs.get('gabor_orientation', 0)
        sigma_x = kwargs.get('gabor_sigma_x', 4.0)
        sigma_y = kwargs.get('gabor_sigma_y', 4.0)
        processed = enhance_gabor_filter(processed, frequency, orientation, sigma_x, sigma_y)
    
    # Step 5: Binarization (required for thinning)
    if thin or binarize:
        method = kwargs.get('binarize_method', 'adaptive')
        block_size = kwargs.get('binarize_block_size', 11)
        c = kwargs.get('binarize_c', 2)
        processed = binarize_image(processed, method, block_size, c)
    
    # Step 6: Thinning/Skeletonization (optional)
    if thin:
        processed = skeletonize(processed)
    
    return processed


def save_image(image: np.ndarray, output_path: str) -> None:
    """
    Save processed image to file.
    
    Args:
        image: Image to save
        output_path: Path where to save the image
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if path contains a directory
        os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(output_path, image)


def preprocess_directory(input_dir: str, output_dir: str, 
                        **preprocessing_kwargs) -> None:
    """
    Preprocess all images in a directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save preprocessed images
        **preprocessing_kwargs: Arguments to pass to preprocess_pipeline
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.bmp', '.jpg', '.jpeg', '.png']
    image_files = [f for f in os.listdir(input_dir) 
                   if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    print(f"Processing {len(image_files)} images from {input_dir}...")
    
    for filename in image_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            # Load image
            image = load_image(input_path)
            
            # Preprocess
            processed = preprocess_pipeline(image, **preprocessing_kwargs)
            
            # Save
            save_image(processed, output_path)
            
            print(f"Processed: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    print(f"Preprocessing complete! Output saved to {output_dir}")

