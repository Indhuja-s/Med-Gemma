import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model import SimpleUNet # We import our architecture

# 1. Load the model and the weights
device = torch.device("cpu") # Using CPU for testing is fine
model = SimpleUNet().to(device)
model.load_state_dict(torch.load("medgemma_model.pth", map_location=device))
model.eval() # Put model in evaluation mode

# 2. Pick a patient to test
df = pd.read_csv('dataset_index.csv')
sample_idx = 10 # Let's look at patient #10
brain_path = df.iloc[sample_idx]['Flair_Path']
mask_path = df.iloc[sample_idx]['Seg_Path']

# 3. Process the image exactly like training
brain_data = nib.load(brain_path).get_fdata()
mask_data = nib.load(mask_path).get_fdata()

slice_idx = 75
x = brain_data[:, :, slice_idx]
y_true = (mask_data[:, :, slice_idx] > 0).astype(np.float32)

# Normalize
if x.std() > 0:
    x_norm = (x - x.mean()) / x.std()
else:
    x_norm = x

# Predict
input_tensor = torch.from_numpy(x_norm).float().unsqueeze(0).unsqueeze(0)
with torch.no_grad():
    prediction = model(input_tensor)
    prediction = (prediction > 0.5).float().numpy()[0, 0] # Threshold at 0.5

# 4. Visualize Results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(x, cmap='gray')
plt.title("Original MRI (Input)")

plt.subplot(1, 3, 2)
plt.imshow(y_true, cmap='Reds')
plt.title("Doctor's Label (Ground Truth)")

plt.subplot(1, 3, 3)
plt.imshow(prediction, cmap='Greens')
plt.title("AI Prediction (MedGemma Output)")

plt.show()