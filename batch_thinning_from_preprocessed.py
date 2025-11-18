"""
Batch process binarized images from preprocessed_output to thinned images using Block Filter.

The binarized image is thinned using Block Filter to reduce the thickness of all ridge lines 
to a single pixel width to extract minutiae points effectively.
"""

from preprocessing import load_image, skeletonize, save_image
import os

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: tqdm not installed. Progress bar will not be shown.")


def process_all_to_thinned_from_preprocessed(
    input_base_dir: str = "preprocessed_output",
    output_base_dir: str = "thinned_output_from_preprocessed") -> None:
    """
    Process all binarized images from preprocessed_output to thinned images using Block Filter.
    
    Args:
        input_base_dir: Base directory containing binarized images (preprocessed_output)
        output_base_dir: Base output directory for thinned images
    """
    subdirs = ['train', 'validate', 'test']
    
    print("="*70)
    print("BATCH THINNING: Converting Binarized Images to Thinned Images")
    print("Using Block Filter method (dilation and erosion)")
    print("="*70)
    print(f"Input base directory: {input_base_dir}")
    print(f"Output base directory: {output_base_dir}")
    print("="*70)
    
    total_processed = 0
    total_errors = 0
    
    for subdir in subdirs:
        input_dir = os.path.join(input_base_dir, subdir)
        output_dir = os.path.join(output_base_dir, subdir)
        
        if not os.path.exists(input_dir):
            print(f"\n⚠ Directory not found: {input_dir}")
            continue
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_extensions = ['.bmp', '.jpg', '.jpeg', '.png']
        image_files = [f for f in os.listdir(input_dir) 
                      if any(f.lower().endswith(ext) for ext in image_extensions)]
        
        if not image_files:
            print(f"\n⚠ No image files found in {input_dir}")
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing {len(image_files)} images from {subdir}/")
        print(f"{'='*70}")
        
        # Process with progress bar if available
        iterator = tqdm(image_files, desc=f"Processing {subdir}") if HAS_TQDM else image_files
        
        subdir_processed = 0
        subdir_errors = 0
        
        for filename in iterator:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                # Load binarized image
                binarized_image = load_image(input_path)
                
                # Ensure image is binary (0 or 255)
                import numpy as np
                if binarized_image.max() > 1:
                    binary = (binarized_image > 127).astype(np.uint8) * 255
                else:
                    binary = binarized_image.astype(np.uint8) * 255
                
                # Thin using Block Filter method (skeletonize function)
                thinned_image = skeletonize(binary)
                
                # Save thinned image
                save_image(thinned_image, output_path)
                subdir_processed += 1
                total_processed += 1
                
                if not HAS_TQDM:
                    print(f"  ✓ {filename}")
                    
            except Exception as e:
                subdir_errors += 1
                total_errors += 1
                error_msg = f"  ✗ Error processing {filename}: {str(e)}"
                if HAS_TQDM:
                    tqdm.write(error_msg)
                else:
                    print(error_msg)
        
        print(f"\n  {subdir}/: {subdir_processed} successful, {subdir_errors} errors")
    
    print("\n" + "="*70)
    print("BATCH THINNING COMPLETE")
    print("="*70)
    print(f"Total processed: {total_processed}")
    print(f"Total errors: {total_errors}")
    print(f"Output directory: {output_base_dir}")
    print("="*70)
    print("\n✓ All thinned images saved successfully!")
    print("\nNote: These thinned images can now be used for feature extraction")
    print("      with is_thinned=True parameter.")


if __name__ == "__main__":
    import sys
    
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "preprocessed_output"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "thinned_output_from_preprocessed"
    
    process_all_to_thinned_from_preprocessed(input_dir, output_dir)

