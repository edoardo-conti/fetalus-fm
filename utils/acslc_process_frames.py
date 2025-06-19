import csv
import ast

def process_csv(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        
        # Define output fields (original fields minus mask columns and has_mask)
        output_fields = [
            'images_path', 'class', 'patient_id', 'frame_id', 'tot_frames', 
            'resolution', 'masks_path', 'masks_structures',  'mask_plane'
        ]
        
        writer = csv.DictWriter(outfile, fieldnames=output_fields)
        writer.writeheader()
        
        for row in reader:
            # Parse mask indexes
            opt_idxs = ast.literal_eval(row['optplane_masks_idxs']) if row['optplane_masks_idxs'] else []
            subopt_idxs = ast.literal_eval(row['suboptplane_masks_idxs']) if row['suboptplane_masks_idxs'] else []
            num_frames = int(row['num_frames'])
            
            # Create new row for each frame
            for frame_id in range(num_frames):
                # Only process frames with masks
                if frame_id in opt_idxs or frame_id in subopt_idxs:
                    # Extract UUID from original path
                    uuid = row['images_path'].split('/')[-1].split('.')[0]
                    
                    new_row = {
                        'images_path': f"annotated/images/{uuid}_p{row['patient_id']}_{frame_id}.jpg",
                        'class': row['class'],
                        'patient_id': row['patient_id'],
                        'frame_id': frame_id,
                        'tot_frames': row['num_frames'],
                        'resolution': row['resolution'],
                        'masks_path': f"annotated/masks/{uuid}_p{row['patient_id']}_{frame_id}.jpg",
                        'masks_structures': row['masks_structures'],
                    }
                    
                    # Determine mask_plane value
                    in_opt = frame_id in opt_idxs
                    in_subopt = frame_id in subopt_idxs
                    if in_opt and in_subopt:
                        new_row['mask_plane'] = 'both'
                    elif in_opt:
                        new_row['mask_plane'] = 'opt'
                    else:
                        new_row['mask_plane'] = 'subopt'
                    
                    writer.writerow(new_row)

if __name__ == '__main__':
    input_csv = 'data_csv/ACSLC/acslc_test.csv'
    output_csv = 'data_csv/ACSLC/acslc_ann_test.csv'
    process_csv(input_csv, output_csv)
