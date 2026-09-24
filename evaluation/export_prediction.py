import torch
import nibabel as nib
import numpy as np
import pandas as pd
import os
from model import SimpleUNet

# 1. Load Model
device = torch.device("cpu")
model = SimpleUNet().to(device)
model.load_state_dict(torch.load("medgemma_model.pth", map_location=device))
model.eval()

# 2. Select Patient
df = pd.read_csv('dataset_index.csv')
patient = df.iloc[10] # Using patient 10 as our example
img = nib.load(patient['Flair_Path'])
data = img.get_fdata()

# 3. Predict slice-by-slice to build a 3D volume
# We create an empty array with the same shape as the original brain
full_prediction = np.zeros_like(data)

print(f"Generating 3D Mask for {patient['Patient_ID']}...")

for s in range(data.shape[2]):
    slice_data = data[:, :, s]
    if slice_data.max() > 0: # Only process slices that aren't empty
        # Normalize
        norm_slice = (slice_data - slice_data.mean()) / (slice_data.std() + 1e-8)
        input_tensor = torch.from_numpy(norm_slice).float().unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            pred = model(input_tensor)
            mask = (pred > 0.5).float().numpy()[0, 0]
            full_prediction[:, :, s] = mask

# 4. Save as a NIfTI file
# IMPORTANT: We use the 'affine' from the original image so the 
# prediction aligns perfectly with the brain in 3D space.
output_filename = f"{patient['Patient_ID']}_AI_PREDICTION.nii.gz"
new_img = nib.Nifti1Image(full_prediction.astype(np.uint8), img.affine)
nib.save(new_img, output_filename)

print(f"Success! Prediction saved as: {output_filename}")