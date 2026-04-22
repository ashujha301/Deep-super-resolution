# ---- Patch Extraction Engine

import numpy as np


class PatchExtractor:
    def __init__(
        self,
        hr_patch_size=96,
        scale=2,
        stride=48,
        variance_threshold=100.0
    ):
        self.hr_size = hr_patch_size
        self.lr_size = hr_patch_size // scale
        self.scale = scale
        self.stride = stride
        self.variance_threshold = variance_threshold

    # ---- Variance filter
    def _is_valid_patch(self, patch):
        return np.var(patch) > self.variance_threshold

    # ---- Grid extraction (offline)
    def extract_grid(self, hr_img, lr_img):
        hr_patches = []
        lr_patches = []

        H, W = hr_img.shape[:2]

        for y in range(0, H - self.hr_size + 1, self.stride):
            for x in range(0, W - self.hr_size + 1, self.stride):

                hr_patch = hr_img[y:y+self.hr_size, x:x+self.hr_size]

                # ---- Corresponding LR patch
                lr_y = y // self.scale
                lr_x = x // self.scale

                lr_patch = lr_img[
                    lr_y:lr_y+self.lr_size,
                    lr_x:lr_x+self.lr_size
                ]

                # ---- Skip low-variance patches
                if not self._is_valid_patch(hr_patch):
                    continue

                hr_patches.append(hr_patch)
                lr_patches.append(lr_patch)

        return hr_patches, lr_patches

    # ---- Random extraction (online)
    def extract_random(self, hr_img, lr_img, n_patches=10):
        hr_patches = []
        lr_patches = []

        H, W = hr_img.shape[:2]

        for _ in range(n_patches):
            y = np.random.randint(0, H - self.hr_size)
            x = np.random.randint(0, W - self.hr_size)

            hr_patch = hr_img[y:y+self.hr_size, x:x+self.hr_size]

            lr_y = y // self.scale
            lr_x = x // self.scale

            lr_patch = lr_img[
                lr_y:lr_y+self.lr_size,
                lr_x:lr_x+self.lr_size
            ]

            hr_patches.append(hr_patch)
            lr_patches.append(lr_patch)

        return hr_patches, lr_patches