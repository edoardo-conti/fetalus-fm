import pandas as pd
import numpy as np

def split_dataset(input_path, output_dir, ratios=[0.7, 0.2, 0.1]):
    """
    Split dataset into train/val/test sets while:
    1. Keeping all rows from same patient_id together
    2. Maintaining proportional distribution of non_empty_masks counts
    
    Args:
        input_path: Path to input CSV file
        output_dir: Directory to save output CSVs
        ratios: List of ratios for train/val/test splits (must sum to 1)
    """
    # 1. Load data and group by patient
    df = pd.read_csv(input_path)
    patient_groups = df.groupby('patient_id')
    
    # 2. Create patient records with total mask counts
    patients = []
    for pid, group in patient_groups:
        patients.append({
            'patient_id': pid,
            'mask_count': group['non_empty_masks'].sum(),
            'rows': group
        })
    
    # 3. Calculate total masks and target allocations
    total_patients = len(patients)
    total_masks = sum(p['mask_count'] for p in patients)
    targets = [total_masks * ratio for ratio in ratios]
    
    # 4. Sort patients by mask count (descending)
    patients.sort(key=lambda x: x['mask_count'], reverse=True)
    
    # 5. Greedy assignment to sets with minimum allocation
    sets = {
        'train': {'current': 0, 'target': targets[0], 'patients': []},
        'val': {'current': 0, 'target': targets[1], 'patients': []},
        'test': {'current': 0, 'target': targets[2], 'patients': []}
    }
    
    # First pass - assign at least one patient to each set if possible
    if len(patients) >= len(sets):
        for set_name in sets:
            if patients:
                patient = patients.pop(0)
                sets[set_name]['patients'].append(patient)
                sets[set_name]['current'] += patient['mask_count']
    
    # Second pass - assign remaining patients greedily
    for patient in patients:
        # Find set with largest remaining capacity
        set_name = max(sets.keys(),
                      key=lambda k: sets[k]['target'] - sets[k]['current'])
        sets[set_name]['patients'].append(patient)
        sets[set_name]['current'] += patient['mask_count']
    
    # 6. Write output CSVs and print statistics
    print("\n=== Dataset Split Summary ===")
    print(f"Total patients: {total_patients}")
    print(f"Total non-empty masks: {total_masks}\n")
    
    for set_name in sets:
        print(f"Processing {set_name} set with {len(sets[set_name]['patients'])} patients")
        if sets[set_name]['patients']:
            set_df = pd.concat([p['rows'] for p in sets[set_name]['patients']])
            set_mask_count = sum(p['mask_count'] for p in sets[set_name]['patients'])
            print(f"Writing {len(set_df)} rows with {set_mask_count} masks to {set_name}.csv")
            set_df.to_csv(f"{output_dir}/{set_name}.csv", index=False)
        else:
            print(f"Warning: No patients assigned to {set_name} set")
    
    # Print final distribution
    print("\n=== Final Distribution ===")
    for set_name in sets:
        if sets[set_name]['patients']:
            set_mask_count = sum(p['mask_count'] for p in sets[set_name]['patients'])
            print(f"{set_name}: {len(sets[set_name]['patients'])} patients, {set_mask_count} masks ({set_mask_count/total_masks:.1%})")

if __name__ == "__main__":
    # Example usage with the provided file
    split_dataset(
        input_path='data_csv/ACSLC/acslc.csv',
        output_dir='data_csv/ACSLC'
    )
