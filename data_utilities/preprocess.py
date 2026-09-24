import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load our index
df = pd.read_csv('dataset_index.csv')

# Let's process the first patient
sample_path = df.iloc[0]['Flair_Path']
img = nib.load(sample_path)
data = img.get_fdata()

# Z-score Normalization: (x - mean) / std
# We only normalize the brain, not the black background (zeros)
brain_mask = data > 0
mean = data[brain_mask].mean()
std = data[brain_mask].std()
normalized_data = (data - mean) / std

# Visualize the difference
slice_idx = 75
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(data[:, :, slice_idx], cmap='gray')
plt.title(f"Original (Max: {data.max():.2f})")

plt.subplot(1, 2, 2)
plt.imshow(normalized_data[:, :, slice_idx], cmap='gray')
plt.title(f"Normalized (Mean: ~0, Std: 1)")

plt.show()