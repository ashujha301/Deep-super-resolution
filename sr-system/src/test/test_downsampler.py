import numpy as np
from PIL import Image
from src.data.downsampler import DownSampler

# ---- Load sample image
img_path = "data/raw/div2k/train_hr/0001.png"
hr = np.array(Image.open(img_path)).astype(np.float32)

# ---- Create downsampler
downsampler = DownSampler(scale_factor=2, mode="bicubic")

# ---- Generate LR
lr = downsampler.degrade(hr)

print("HR shape:", hr.shape)
print("LR shape:", lr.shape)