import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Load a sample
path = 'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii'
mask_path = 'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_seg.nii'

data = nib.load(path).get_fdata()
mask = nib.load(mask_path).get_fdata()

# Flip horizontally (Left-to-Right)
# In MRI data, the 1st axis (0) is usually the left-right orientation
flipped_data = np.flip(data, axis=0)
flipped_mask = np.flip(mask, axis=0)

# Visualize the Mirror Effect
slice_idx = 75
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(data[:, :, slice_idx], cmap='gray')
plt.title("Original Brain")

plt.subplot(2, 2, 2)
plt.imshow(flipped_data[:, :, slice_idx], cmap='gray')
plt.title("Flipped Brain (Augmented)")

plt.subplot(2, 2, 3)
plt.imshow(mask[:, :, slice_idx], cmap='jet')
plt.title("Original Mask")

plt.subplot(2, 2, 4)
plt.imshow(flipped_mask[:, :, slice_idx], cmap='jet')
plt.title("Flipped Mask (Matches!)")

plt.tight_layout()
plt.show()