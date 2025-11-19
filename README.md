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
├── preprocessing.py                # Preprocessing pipeline
├── preprocessing_example.py        # Preprocessing visualization script
├── test_preprocessing.py           # Preprocessing smoke test
├── feature_extraction.py           # Minutiae-based feature extraction
├── test_feature_extraction.py      # Feature extraction smoke test
├── enrollment.py                   # Enrollment helpers (template generation)
├── matching.py                     # Matching helpers (identification)
├── evaluation.py                   # Evaluation utilities (FAR/FRR, plots)
├── comprehensive_test.py           # End-to-end integration test
├── utils.py                        # Shared helpers (filename parsing, etc.)
├── requirements.txt                # Python dependencies
├── project-data/Project-Data/      # Dataset folders (train/validate/test)
├── database/templates/             # Generated enrollment templates (gitignored)
├── preprocessed_output/            # Generated preprocessing outputs (gitignored)
├── feature_output/                 # Generated feature visualizations (gitignored)
└── evaluation_output/              # Generated evaluation reports (gitignored)
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

## Feature Extraction Module

The feature extraction module (`feature_extraction.py`) detects minutiae points
(ridge endings and bifurcations) that uniquely characterize each fingerprint.

### Usage Example

```python
from preprocessing import load_image, preprocess_pipeline
from feature_extraction import extract_features, minutiae_to_features

image = load_image("project-data/Project-Data/train/000_R0_0.bmp")
processed = preprocess_pipeline(image)
features = extract_features(processed, block_size=16, min_distance=10)
feature_vector = minutiae_to_features(features["minutiae"])
print("Minutiae count:", len(feature_vector))
```

### Running Examples

1. Test feature extraction on a sample image:
```bash
python test_feature_extraction.py
```

2. Run the integration test (preprocessing + feature extraction + stats):
```bash
python comprehensive_test.py
```

## Enrollment & Matching

Use `enrollment.py` to generate templates from the training set, then use
`matching.py` to identify query fingerprints.

### Enroll all training fingerprints
```bash
python enrollment.py --train-dir project-data/Project-Data/train --template-dir database/templates
```

### Match a single fingerprint image
```bash
python matching.py project-data/Project-Data/validate/000_R0_3.bmp \
  --template-dir database/templates --threshold 0.4
```

## Evaluation

`evaluation.py` benchmarks the system on the validation or test sets and
produces metrics (Accuracy, FAR, FRR) plus simple visualizations.

```bash
# Evaluate on validation set
python evaluation.py --template-dir database/templates \
  --query-dir project-data/Project-Data/validate --threshold 0.4

# Evaluate on test set (after you are satisfied with validation performance)
python evaluation.py --template-dir database/templates \
  --query-dir project-data/Project-Data/test --threshold 0.4
```

Results (JSON, TXT, PNG) are saved to `evaluation_output/`.

## Current Status

✅ **Preprocessing**: Complete (normalization, denoising, CLAHE, demos)  
✅ **Feature Extraction**: Complete (minutiae detection, visualizations)  
✅ **Enrollment and Matching**: Complete (template generation + identification)  
✅ **Evaluation**: Complete (metrics, reports, visualizations)
