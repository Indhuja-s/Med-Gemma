import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Load a sample from your index
path = 'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii'
img = nib.load(path)
data = img.get_fdata()

# Find where the brain actually is (where intensity > 0)
coords = np.array(np.nonzero(data > 0))
start_coords = coords.min(axis=1)
end_coords = coords.max(axis=1)

# Crop the 3D volume
cropped_data = data[start_coords[0]:end_coords[0], 
                    start_coords[1]:end_coords[1], 
                    start_coords[2]:end_coords[2]]

print(f"Original shape: {data.shape}")
print(f"Cropped shape:  {cropped_data.shape}")

# Visualize the "Before vs After"
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(data[:, :, 75], cmap='gray')
plt.title("Original (Lots of black space)")

plt.subplot(1, 2, 2)
plt.imshow(cropped_data[:, :, cropped_data.shape[2]//2], cmap='gray')
plt.title("Cropped (Focused on Brain)")
plt.show()