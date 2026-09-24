import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import nibabel as nib
import numpy as np
import os

# --- 1. MODEL ARCHITECTURE (SimpleUNet) ---
def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.down1 = double_conv(1, 64)
        self.pool = nn.MaxPool2d(2)
        self.down2 = double_conv(64, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = double_conv(128, 64)
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        c1 = self.down1(x)
        p1 = self.pool(c1)
        c2 = self.down2(p1)
        t1 = self.up1(c2)
        combined = torch.cat([t1, c1], dim=1)
        c3 = self.conv_up1(combined)
        return torch.sigmoid(self.out(c3))

# --- 2. SAFE DATA LOADER ---
class BrainDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        flair_path = self.df.iloc[idx]['Flair_Path']
        seg_path = self.df.iloc[idx]['Seg_Path']
        
        # Safety Check: If file is missing, skip to index 0
        if not os.path.exists(flair_path) or not os.path.exists(seg_path):
            return self.__getitem__(0) 

        # Load MRI and Mask
        brain = nib.load(flair_path).get_fdata()
        mask = nib.load(seg_path).get_fdata()
        
        # Use middle slice (75)
        slice_idx = 75
        x = brain[:, :, slice_idx]
        y = (mask[:, :, slice_idx] > 0).astype(np.float32)
        
        # Z-Score Normalization
        if x.std() > 0:
            x = (x - x.mean()) / x.std()
        
        # Format for PyTorch (Channel, H, W)
        x = torch.from_numpy(x).float().unsqueeze(0)
        y = torch.from_numpy(y).float().unsqueeze(0)
        return x, y

# --- 3. TRAINING LOOP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleUNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss() # Binary Cross Entropy

dataset = BrainDataset('dataset_index.csv')
loader = DataLoader(dataset, batch_size=4, shuffle=True)

print(f"Training started on {device}...")

for epoch in range(5):
    epoch_loss = 0
    for batch_idx, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / len(loader)
    print(f"Epoch [{epoch+1}/5] Completed. Average Loss: {avg_loss:.4f}")

# Save the learned weights
torch.save(model.state_dict(), "medgemma_model.pth")
print("Training Complete! File saved as medgemma_model.pth")