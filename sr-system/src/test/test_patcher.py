import numpy as np
from PIL import Image
from src.data.downsampler import DownSampler
from src.data.patcher import PatchExtractor

# ---- Load image
img_path = "data/raw/div2k/train_hr/0001.png"
hr = np.array(Image.open(img_path)).astype(np.float32)

# ---- Downsample
downsampler = DownSampler(scale_factor=2)
lr = downsampler.degrade(hr)

# ---- Patch extractor
patcher = PatchExtractor(hr_patch_size=96, scale=2, stride=48)

hr_patches, lr_patches = patcher.extract_grid(hr, lr)

print("Total patches:", len(hr_patches))
print("HR patch shape:", hr_patches[0].shape)
print("LR patch shape:", lr_patches[0].shape)