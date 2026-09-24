import os
import pandas as pd

# The path to your unzipped data
base_path = 'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
data_list = []

print("Crawling through folders... this might take a second.")

# Check if path exists first
if not os.path.exists(base_path):
    print(f"Error: Path not found: {base_path}")
else:
    for patient_folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, patient_folder)
        
        # We only want folders that start with 'BraTS20_Training_'
        if os.path.isdir(folder_path) and "BraTS20_Training_" in patient_folder:
            data_list.append({
                'Patient_ID': patient_folder,
                'Flair_Path': os.path.join(folder_path, f"{patient_folder}_flair.nii"),
                'Seg_Path': os.path.join(folder_path, f"{patient_folder}_seg.nii")
            })

    df = pd.DataFrame(data_list)
    df.to_csv('dataset_index.csv', index=False)
    print(f"Done! Found {len(df)} patients.")
    print(df.head())