# Fingerprint Authentication System
CSC 591: Cyber-Physical Systems - Biometrics

## Project Overview

This project implements a fingerprint authentication system with four main components:
1. **Preprocessing**: Image enhancement, noise reduction, and normalization
2. **Feature Extraction**: Extract relevant features from fingerprint images
3. **Enrollment and Matching**: Compare fingerprints against enrolled data
4. **Evaluation**: Evaluate system performance and visualize results

## Project Structure

```
├── preprocessing.py          # Preprocessing module with image enhancement functions
├── preprocessing_example.py  # Example script demonstrating preprocessing
├── test_preprocessing.py     # Test script for preprocessing functionality
├── utils.py                  # Utility functions for filename parsing and data management
├── requirements.txt          # Python dependencies
└── project-data/             # Fingerprint image data
    └── Project-Data/
        ├── train/            # Training images (indices 0, 1, 2)
        ├── validate/         # Validation images (index 3)
        └── test/             # Test images (index 4)
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. The data should be in `project-data/Project-Data/` directory with subdirectories:
   - `train/`: Images for enrollment (format: YYY_R0_KKK.bmp)
   - `validate/`: Images for validation and debugging
   - `test/`: Images for final evaluation (do not use for tuning)

## Preprocessing Module

The preprocessing module (`preprocessing.py`) provides:

- **Normalization**: Intensity normalization to standardize image properties
- **Noise Reduction**: Gaussian blur and median filtering
- **Contrast Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Ridge Enhancement**: Gabor filtering (optional)
- **Binarization**: Adaptive thresholding (optional)

### Usage Example

```python
from preprocessing import load_image, preprocess_pipeline, save_image

# Load image
image = load_image("project-data/Project-Data/train/000_R0_0.bmp")

# Preprocess with default settings
processed = preprocess_pipeline(
    image,
    normalize=True,
    denoise=True,
    enhance_contrast=True,
    enhance_ridges=False,
    binarize=False
)

# Save processed image
save_image(processed, "output/preprocessed_000_R0_0.bmp")
```

### Running Examples

1. Test preprocessing on a sample image:
```bash
python test_preprocessing.py
```

2. Visualize preprocessing steps:
```bash
python preprocessing_example.py
```

## Current Status

✅ **Preprocessing**: Complete implementation with multiple enhancement techniques
⏳ **Feature Extraction**: To be implemented
⏳ **Enrollment and Matching**: To be implemented
⏳ **Evaluation**: To be implemented
