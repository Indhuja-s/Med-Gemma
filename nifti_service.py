import os
import io
import base64
import random
import numpy as np
import nibabel as nib
from PIL import Image

def get_base64_img(arr, brightness=1.0, contrast=1.0):
    """Normalize a 2D numpy array and encode as base64 PNG."""
    if arr is None or arr.size == 0:
        arr_norm = np.zeros((240, 240), dtype=np.uint8)
    else:
        min_v, max_v = arr.min(), arr.max()
        if max_v - min_v == 0:
            arr_norm = np.zeros_like(arr, dtype=np.uint8)
        else:
            arr_norm = ((arr - min_v) / (max_v - min_v) * 255.0)
            arr_norm = np.clip(arr_norm * brightness, 0, 255).astype(np.uint8)
    
    arr_norm = np.rot90(arr_norm)
    img = Image.fromarray(arr_norm)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_slice_2d(data, slice_idx, orientation="axial"):
    """Extract a 2D slice from 3D data (X, Y, Z) for the given orientation."""
    shape = data.shape
    if orientation == "axial":
        max_idx = shape[2] - 1
        idx = max(0, min(int(slice_idx), max_idx))
        return data[:, :, idx], idx
    elif orientation == "coronal":
        max_idx = shape[1] - 1
        idx = max(0, min(int(slice_idx), max_idx))
        return data[:, idx, :], idx
    elif orientation == "sagittal":
        max_idx = shape[0] - 1
        idx = max(0, min(int(slice_idx), max_idx))
        return data[idx, :, :], idx
    else:
        max_idx = shape[2] - 1
        idx = max(0, min(int(slice_idx), max_idx))
        return data[:, :, idx], idx

def generate_synthetic_3d_volume(patient_id):
    """Generate a realistic synthetic 3D MRI volume (240, 240, 155) and 3D mask."""
    seed = sum(ord(c) for c in str(patient_id))
    rng = random.Random(seed)
    
    nx, ny, nz = 240, 240, 155
    data_3d = np.zeros((nx, ny, nz), dtype=np.float32)
    mask_3d = np.zeros((nx, ny, nz), dtype=np.float32)
    
    cx, cy, cz = 120, 120, 78
    t_radius = rng.randint(18, 30)
    tx = cx + rng.randint(-20, 20)
    ty = cy + rng.randint(-20, 20)
    tz = cz + rng.randint(-15, 15)
    
    y, x, z = np.ogrid[:nx, :ny, :nz]
    
    skull_dist = ((x - cx)/95.0)**2 + ((y - cy)/105.0)**2 + ((z - cz)/70.0)**2
    data_3d[skull_dist <= 1.0] = 60.0
    
    brain_dist = ((x - cx)/85.0)**2 + ((y - cy)/95.0)**2 + ((z - cz)/62.0)**2
    data_3d[brain_dist <= 1.0] = 130.0
    
    v_dist1 = ((x - (cx - 15))/10.0)**2 + ((y - cy)/30.0)**2 + ((z - cz)/25.0)**2
    v_dist2 = ((x - (cx + 15))/10.0)**2 + ((y - cy)/30.0)**2 + ((z - cz)/25.0)**2
    data_3d[v_dist1 <= 1.0] = 20.0
    data_3d[v_dist2 <= 1.0] = 20.0
    
    tumor_dist = ((x - tx)/float(t_radius))**2 + ((y - ty)/float(t_radius))**2 + ((z - tz)/float(t_radius*0.8))**2
    tumor_mask = tumor_dist <= 1.0
    
    data_3d[tumor_mask] = 240.0
    mask_3d[tumor_mask] = 1.0
    
    return data_3d, mask_3d
