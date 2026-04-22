import numpy as np
from PIL import Image
from src.data.downsampler import DownSampler
from src.data.patcher import PatchExtractor
from src.data.augmentation import Augmentor

# ---- Load image
img_path = "data/raw/div2k/train_hr/0001.png"
hr = np.array(Image.open(img_path)).astype(np.float32)

# ---- Downsample
downsampler = DownSampler(scale_factor=2)
lr = downsampler.degrade(hr)

# ---- Extract patch
patcher = PatchExtractor()
hr_patches, lr_patches = patcher.extract_random(hr, lr, n_patches=1)

hr_patch = hr_patches[0]
lr_patch = lr_patches[0]

# ---- Augment
augmentor = Augmentor()
hr_aug, lr_aug = augmentor.augment(hr_patch, lr_patch)

print("HR shape:", hr_aug.shape)
print("LR shape:", lr_aug.shape)

# ---- Save for visual check
Image.fromarray(hr_aug.astype(np.uint8)).save("data/samples/hr_aug.png")
Image.fromarray(lr_aug.astype(np.uint8)).save("data/samples/lr_aug.png")