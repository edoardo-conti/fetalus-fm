import os
import ast
import argparse
import csv
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Extract frames from ultrasound video sweeps')
    parser.add_argument('--base_path', 
                        default='/Volumes/Seagate/FETAL_US/ACSLC',
                        help='Path to ACOUSLIC dataset base directory')
    parser.add_argument('--output_dir', 
                        default='acslc_mha2frames_output',
                        help='Output directory for frames')
    parser.add_argument('--csv_path', 
                        default='data_csv/ACSLC/acslc.csv',
                        help='Path to CSV file')
    parser.add_argument('--mode', choices=['annotated', 'unannotated'], default='annotated',
                       help="Mode: 'annotated' for frames with non-empty masks, 'unannotated' for unsupervised frames")
    parser.add_argument('--start_row', type=int, default=0,
                       help='Starting row index in CSV (0-based)')
    parser.add_argument('--batch_size', type=int, default=-1,
                       help='Number of rows to process (-1 for all rows)')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(args.csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        # Process rows in batch
        rows = list(reader)
        total_rows = len(rows)
        
        # Apply batch parameters
        start = args.start_row
        end = start + args.batch_size if args.batch_size > 0 else total_rows
        end = min(end, total_rows)
        
        if start >= total_rows:
            print(f"Error: start_row {start} is beyond CSV length {total_rows}")
            return
            
        print(f"Processing rows {start} to {end-1} of {total_rows}")
        for row in tqdm(rows[start:end], desc="Processing sweep videos", total=end-start):
            try:
                # Get values by column name
                image_rel_path = row['images_path']
                mask_rel_path = row['masks_path']
                patient_id = row['patient_id']
                
                # Parse indices (handle empty lists)
                opt_idxs = ast.literal_eval(row['optplane_masks_idxs'] or '[]')
                subopt_idxs = ast.literal_eval(row['suboptplane_masks_idxs'] or '[]')
                all_idxs = opt_idxs + subopt_idxs
                
                if not all_idxs:
                    continue
                    
                # Build full paths
                image_path = os.path.join(args.base_path, image_rel_path)
                mask_path = os.path.join(args.base_path, mask_rel_path)
                
                # Verify files exist before processing
                if not all(os.path.exists(p) for p in [image_path, mask_path]):
                    print(f"Warning: Missing files for patient {patient_id}")
                    continue
                
                # Load video sweeps
                try:
                    # Read as numpy arrays for efficient frame access
                    img_array = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
                    mask_array = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
                    
                    # Verify dimensions match expected (frames, height, width)
                    if img_array.ndim != 3 or mask_array.ndim != 3:
                        print(f"Warning: Expected video sweep but got different dimensionality for patient {patient_id}")
                        continue
                        
                    # Verify frame counts match CSV
                    num_frames = int(row['num_frames'])
                    if img_array.shape[0] != num_frames or mask_array.shape[0] != num_frames:
                        print(f"Warning: Frame count mismatch for patient {patient_id}")
                        continue
                        
                except Exception as e:
                    print(f"Error loading video sweeps for patient {patient_id}: {str(e)}")
                    continue
                
                # Process frames based on mode
                if args.mode == 'annotated':
                    indices_to_process = all_idxs
                else:  # 'unannotated' mode
                    all_frames = range(int(row['num_frames']))
                    indices_to_process = [i for i in all_frames if i not in all_idxs]
                
                if not indices_to_process:
                    continue
                    
                # Process each frame index  
                for idx in indices_to_process:
                    try:
                        # Verify frame index is valid
                        if idx < 0 or idx >= img_array.shape[0]:
                            print(f"Warning: Frame index {idx} out of bounds (0-{img_array.shape[0]-1}) for patient {patient_id}")
                            continue
                            
                        # Extract frames
                        img_frame = img_array[idx]
                        mask_frame = mask_array[idx]
                        
                        # Convert back to SimpleITK images for PNG saving
                        img_slice = sitk.GetImageFromArray(img_frame)
                        
                        # Process mask - ensure binary values are properly scaled
                        mask_frame = (mask_frame > 0).astype(np.uint8) * 255  # Convert to binary 0-255
                        mask_slice = sitk.GetImageFromArray(mask_frame)
                        
                        # Extract UUID from mha filename
                        uuid = os.path.basename(image_rel_path).replace('.mha', '')
                        
                        # Save images with UUID and patient ID
                        # Ensure output subdirectories exist
                        img_dir = os.path.join(args.output_dir, "images")
                        os.makedirs(img_dir, exist_ok=True)
                        
                        # Compose output file path for image
                        output_img = os.path.join(img_dir, f"{uuid}_p{patient_id}_{idx}.jpg")
                        sitk.WriteImage(img_slice, output_img, useCompression=True, compressionLevel=90)
                        
                        # Only save masks in annotated mode
                        if args.mode == 'annotated':
                            mask_dir = os.path.join(args.output_dir, "masks")
                            os.makedirs(mask_dir, exist_ok=True)
                            
                            output_mask = os.path.join(mask_dir, f"{uuid}_p{patient_id}_{idx}.jpg")
                            sitk.WriteImage(mask_slice, output_mask, useCompression=True, compressionLevel=100)
                        
                    except Exception as e:
                        print(f"Error processing frame {idx} for patient {patient_id}: {str(e)}")
                        continue
                        
            except Exception as e:
                print(f"Error processing row for patient {patient_id if 'patient_id' in locals() else 'unknown'}: {str(e)}")
                continue

if __name__ == "__main__":
    main()
