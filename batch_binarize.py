"""
Batch process all fingerprint images to binarized images (no thinning).
Outputs to preprocessed_output directory.
"""

from preprocessing import load_image, preprocess_pipeline, save_image
import os
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: tqdm not installed. Progress bar will not be shown.")


def process_all_to_binarized(base_dir: str = "project-data/Project-Data",
                            output_base_dir: str = "preprocessed_output") -> None:
    """
    Process all images from train, validate, and test directories to binarized images.
    
    Args:
        base_dir: Base directory containing train/validate/test subdirectories
        output_base_dir: Base output directory for binarized images
    """
    subdirs = ['train', 'validate', 'test']
    
    for subdir in subdirs:
        input_dir = os.path.join(base_dir, subdir)
        output_dir = os.path.join(output_base_dir, subdir)
        
        if not os.path.exists(input_dir):
            print(f"Warning: {input_dir} does not exist, skipping...")
            continue
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_extensions = ['.bmp', '.jpg', '.jpeg', '.png']
        image_files = [f for f in os.listdir(input_dir) 
                      if any(f.lower().endswith(ext) for ext in image_extensions)]
        
        if not image_files:
            print(f"No image files found in {input_dir}")
            continue
        
        print(f"\nProcessing {len(image_files)} images from {input_dir}...")
        print(f"Output directory: {output_dir}")
        
        # Process with progress bar if available
        iterator = tqdm(image_files) if HAS_TQDM else image_files
        
        success_count = 0
        error_count = 0
        
        for filename in iterator:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                # Load image
                image = load_image(input_path)
                
                # Preprocess with binarization only (no thinning)
                binarized = preprocess_pipeline(
                    image,
                    normalize=True,
                    denoise=True,
                    enhance_contrast=True,
                    enhance_ridges=False,
                    binarize=True,      # Enable binarization
                    thin=False          # No thinning
                )
                
                # Save binarized image
                save_image(binarized, output_path)
                success_count += 1
                
                if not HAS_TQDM:
                    print(f"  ✓ {filename}")
                    
            except Exception as e:
                error_count += 1
                error_msg = f"  ✗ Error processing {filename}: {str(e)}"
                if HAS_TQDM:
                    tqdm.write(error_msg)
                else:
                    print(error_msg)
        
        print(f"\n  Completed: {success_count} successful, {error_count} errors")
    
    print("\n✓ All binarized images saved successfully!")


if __name__ == "__main__":
    import sys
    
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "project-data/Project-Data"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "preprocessed_output"
    
    print("=" * 60)
    print("Batch Binarization Processing")
    print("=" * 60)
    print(f"Input base directory: {base_dir}")
    print(f"Output base directory: {output_dir}")
    print("=" * 60)
    
    process_all_to_binarized(base_dir, output_dir)

