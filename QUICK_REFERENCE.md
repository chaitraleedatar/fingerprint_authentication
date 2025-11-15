# Quick Reference Guide

## Running Tests

### 1. Test Preprocessing
```bash
python test_preprocessing.py
```
- Tests image loading and preprocessing
- Verifies preprocessing pipeline
- Creates test output

### 2. Visualize Preprocessing Steps
```bash
python demo_preprocessing.py
```
- Shows each preprocessing step visually
- Creates comparison images
- Output: `preprocessed_output/demo/`

### 3. Test Feature Extraction
```bash
python test_feature_extraction.py
```
- Extracts features from sample image
- Shows minutiae points
- Creates visualization
- Output: `feature_output/`

### 4. Comprehensive Test
```bash
python comprehensive_test.py
```
- Tests multiple images
- Checks consistency
- Provides statistics
- Best for full review

## Module Usage

### Preprocessing
```python
from preprocessing import load_image, preprocess_pipeline

image = load_image("path/to/image.bmp")
processed = preprocess_pipeline(
    image,
    normalize=True,
    denoise=True,
    enhance_contrast=True
)
```

### Feature Extraction
```python
from feature_extraction import extract_features, visualize_features

features = extract_features(processed, block_size=16, min_distance=10)
minutiae = features['minutiae']
print(f"Found {len(minutiae)} minutiae points")
```

### Utilities
```python
from utils import parse_filename, get_images_by_person

person_id, index = parse_filename("000_R0_0.bmp")
images = get_images_by_person("project-data/Project-Data/train")
```

## Key Statistics

### Typical Results
- **Minutiae per image:** 100-200 points
- **Ridge endings:** 30-40% of minutiae
- **Bifurcations:** 60-70% of minutiae
- **Processing time:** ~1-2 seconds per image

### File Locations
- Preprocessed images: `preprocessed_output/`
- Feature visualizations: `feature_output/`
- Test outputs: `preprocessed_output/test/`

## Common Issues

### Issue: No minutiae detected
- **Cause:** Poor image quality or preprocessing
- **Solution:** Check preprocessing output, adjust parameters

### Issue: Too many minutiae (>300)
- **Cause:** Noise not properly filtered
- **Solution:** Increase `min_distance` or improve denoising

### Issue: Too few minutiae (<20)
- **Cause:** Over-processing or poor image
- **Solution:** Check preprocessing, verify image quality

## Next Steps

1. ✅ Preprocessing - Complete
2. ✅ Feature Extraction - Complete
3. ⏳ Enrollment - Store features
4. ⏳ Matching - Compare fingerprints
5. ⏳ Evaluation - Measure performance

