import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# Exact paths from your dir command
brain_path = 'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii'
mask_path  = 'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_seg.nii'

# Load data
brain_data = nib.load(brain_path).get_fdata()
mask_data  = nib.load(mask_path).get_fdata()

# Choose a slice where the tumor is visible
slice_idx = 70

plt.figure(figsize=(12, 6))

# Left side: The MRI Scan
plt.subplot(1, 2, 1)
plt.imshow(brain_data[:, :, slice_idx], cmap='gray')
plt.title('MRI Scan (FLAIR)')
plt.axis('off')

# Right side: The Tumor Map (Segmentation)
plt.subplot(1, 2, 2)
plt.imshow(mask_data[:, :, slice_idx], cmap='jet')
plt.title('Tumor Regions (Mask)')
plt.axis('off')

plt.show()
