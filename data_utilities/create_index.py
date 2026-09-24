import os
import pandas as pd

base_path = 'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
data_list = []

print("Crawling through folders... this might take a second.")
for patient_folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, patient_folder)
    if os.path.isdir(folder_path):
        # Store the relative paths for flair and segmentation
        data_list.append({
            'Patient_ID': patient_folder,
            'Flair_Path': os.path.join(folder_path, f"{patient_folder}_flair.nii"),
            'Seg_Path': os.path.join(folder_path, f"{patient_folder}_seg.nii")
        })

df = pd.DataFrame(data_list)
df.to_csv('dataset_index.csv', index=False)
print(f"Done! Found {len(df^)} patients.")
print(df.head())
