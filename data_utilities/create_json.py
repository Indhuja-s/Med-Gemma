import pandas as pd
import json
import os
import nibabel as nib
import numpy as np

def generate_medical_json(csv_file, output_json):
    df = pd.read_csv(csv_file)
    training_data = []

    print(f"Generating JSON for {len(df)} patients based on real NIfTI masks...")

    for idx, row in df.iterrows():
        p_id = row['Patient_ID']
        seg_path = row['Seg_Path']
        
        # Check if the mask file exists
        if not os.path.exists(seg_path):
            continue
            
        try:
            # 1. Load Real Mask
            mask_img = nib.load(seg_path)
            mask_data = mask_img.get_fdata()
            
            # 2. Get Voxel Dimensions (usually 1x1x1 mm for BraTS, but safe to check)
            voxel_volume_mm3 = np.prod(mask_img.header.get_zooms())
            
            # 3. Calculate True Volume (BraTS labels: 1, 2, 4 are tumor core/edema/enhancing)
            tumor_voxels = np.sum(mask_data > 0)
            target_volume_mm3 = tumor_voxels * voxel_volume_mm3
            volume_cm3 = round(target_volume_mm3 / 1000, 2)
            
            # 4. Generate Clinical Impression based on True Data
            if volume_cm3 > 25.0:
                severity = "large, pronounced"
                impression = "High risk mass effect. Urgent surgical consultation and contrast-enhanced imaging required."
            elif volume_cm3 > 5.0:
                severity = "moderate"
                impression = "Recommend surgical consultation and potential targeted biopsy."
            elif volume_cm3 > 0.0:
                severity = "small, localized"
                impression = "Recommend close follow-up monitoring in 3 months. Correlate with clinical history."
            else:
                severity = "indiscernible"
                impression = "No discrete space-occupying lesion observed on current sequences. Unremarkable."
                
            entry = {
                "instruction": "Analyze the MRI volumetric data and provide a clinical impression.",
                "input": f"Patient_ID: {p_id}, Tumor Volume: {volume_cm3} cm3, Modality: FLAIR.",
                "output": f"The volumetric analysis for {p_id} reveals a {volume_cm3} cm3 {severity} abnormality. {impression}"
            }
            training_data.append(entry)
            
            if idx % 50 == 0:
                print(f"Processed {idx} patients...")
                
        except Exception as e:
            print(f"Could not process {seg_path} for {p_id}: {e}")
            continue

    with open(output_json, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"Success! {output_json} created with {len(training_data)} true volumetric patient records.")

if __name__ == '__main__':
    generate_medical_json('dataset_index.csv', 'training_data.json')