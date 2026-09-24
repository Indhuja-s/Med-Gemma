import torch
import numpy as np
import pandas as pd
import nibabel as nib
from model import SimpleUNet

def calculate_dice(pred, target):
    smooth = 1e-6
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# 1. Setup
device = torch.device("cpu")
model = SimpleUNet().to(device)
model.load_state_dict(torch.load("medgemma_model.pth", map_location=device))
model.eval()

# 2. Load Patient 10 (as a test case)
df = pd.read_csv('dataset_index.csv')
sample = df.iloc[10]
brain = nib.load(sample['Flair_Path']).get_fdata()
mask = nib.load(sample['Seg_Path']).get_fdata()

# 3. Get middle slice
slice_idx = 75
x = brain[:, :, slice_idx]
y_true = (mask[:, :, slice_idx] > 0).astype(np.float32)

# Normalize
if x.std() > 0:
    x_norm = (x - x.mean()) / x.std()
else:
    x_norm = x

# 4. Predict
input_tensor = torch.from_numpy(x_norm).float().unsqueeze(0).unsqueeze(0)
with torch.no_grad():
    prediction = model(input_tensor)
    prediction = (prediction > 0.5).float().numpy()[0, 0]

# 5. Calculate Score
score = calculate_dice(prediction, y_true)
print("-" * 30)
print(f"Patient ID: {sample['Patient_ID']}")
print(f"Dice Similarity Score: {score:.4f}")
print("-" * 30)

if score > 0.7:
    print("Result: Excellent overlap!")
elif score > 0.3:
    print("Result: Decent overlap, but needs more training epochs.")
else:
    print("Result: Poor overlap. Check data or train longer.")