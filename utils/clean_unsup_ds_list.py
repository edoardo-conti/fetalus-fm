import argparse
import os
import sys
from collections import defaultdict
from PIL import Image
import numpy as np

def extract_sub_dataset(path):
    """Extract sub-dataset name from the first part of the path."""
    return path.strip().split('/')[0] if '/' in path.strip() else None

def count_lines(txt_file):
    """Count total lines and lines per sub-dataset."""
    total_count = 0
    sub_counts = defaultdict(int)
    
    with open(txt_file, 'r') as f:
        for line in f:
            path = line.strip()
            if path:  # Skip empty lines
                total_count += 1
                sub = extract_sub_dataset(path)
                if sub:
                    sub_counts[sub] += 1
    
    return total_count, dict(sub_counts)

def scan_black_images(txt_file, cleaning_prefix, threshold=50.0):
    """Scan images in the specified sub-dataset prefix and identify mostly black ones."""
    black_list = []
    
    with open(txt_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            path = line.strip()
            if path and path.startswith(cleaning_prefix):
                base_prefix = "/leonardo_scratch/fast/IscrC_FoSAM-X/datasets/UNSUPERVISED"
                full_path = os.path.normpath(os.path.join(base_prefix, path.lstrip('/'))) # Ensure we don't duplicate slashes and normalize the path
                try:
                    img = Image.open(full_path)
                    arr = np.array(img)
                    mean_intensity = np.mean(arr)
                    if mean_intensity < threshold:
                        black_list.append((full_path, mean_intensity, line_num))
                except Exception as e:
                    print(f"Error loading {full_path}: {e}", file=sys.stderr)
    
    return black_list

def main():
    parser = argparse.ArgumentParser(description="Process unsupervised dataset list and scan for black images.")
    parser.add_argument('--txt_file', required=True, help="Path to the .txt file containing image paths.")
    parser.add_argument('--cleaning', help="Sub-dataset prefix for cleaning scan (optional).")
    parser.add_argument('--threshold', type=float, default=20.0, help="Threshold for mean intensity to consider image black (default: 20.0).")
    
    args = parser.parse_args()
    
    # Always perform counting
    total, sub_counts = count_lines(args.txt_file)
    print(f"Total images: {total}")
    print("Counts per sub-dataset:")
    for sub, count in sorted(sub_counts.items()):
        print(f"  {sub}: {count}")
    
    # Scan if cleaning specified
    if args.cleaning:
        print(f"\nScanning paths starting with '{args.cleaning}' for black images (threshold: {args.threshold})...")
        black_images = scan_black_images(args.txt_file, args.cleaning, args.threshold)
        count_black = len(black_images)
        print(f"Number of mostly black images: {count_black}")
        
        if black_images:
            txt_name = os.path.basename(args.txt_file)
            output_file = f"black_images_{args.cleaning}_{txt_name}"
            with open(output_file, 'w') as f:
                f.write("full_path\tmean_intensity\tline_number\n")
                for full_path, mean, line_num in black_images:
                    f.write(f"{full_path}\t{mean:.2f}\t{line_num}\n")
            print(f"List of black images saved to: {output_file}")
        else:
            print("No black images found.")

if __name__ == "__main__":
    main()
