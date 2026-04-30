# ==================================================
# src/data/patch.py
# Responsibility: Patch extraction and image tiling.
# Used by both training (random crop) and inference (tiled overlap).
# ==================================================

import random
from typing import List, Tuple

import cv2
import numpy as np


def extract_random_patch(
    hr_img: np.ndarray,
    hr_patch_size: int,
    scale: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract one random aligned HR patch + its bicubic LR counterpart.

    Alignment rule: top/left snapped to scale-grid boundary so that
    LR pixel (i,j) maps exactly to HR pixel (i*scale, j*scale).
    Without this, sub-pixel misalignment corrupts the L1 loss target.

    Args:
        hr_img:        (H, W, 3) float32 HR image
        hr_patch_size: square patch size in HR pixels (e.g. 128)
        scale:         SR scale factor (2, 4, or 8)

    Returns:
        hr_patch:  (hr_patch_size, hr_patch_size, 3) float32
        lr_region: (hr_patch_size // scale, ..., 3) float32 bicubic LR

    Raises:
        ValueError: if image is smaller than hr_patch_size
    """
    H, W, _ = hr_img.shape

    if H < hr_patch_size or W < hr_patch_size:
        raise ValueError(
            f"Image ({H}×{W}) smaller than patch size ({hr_patch_size}). "
            "Use a larger image or reduce hr_patch_size."
        )

    max_y = H - hr_patch_size
    max_x = W - hr_patch_size

    # Snap to scale boundary for pixel-perfect alignment
    top  = random.randint(0, max_y // scale) * scale
    left = random.randint(0, max_x // scale) * scale

    hr_patch = hr_img[top:top + hr_patch_size,
                      left:left + hr_patch_size, :]

    lr_size = hr_patch_size // scale
    lr_region = cv2.resize(
        hr_patch, (lr_size, lr_size), interpolation=cv2.INTER_CUBIC
    )

    return hr_patch, lr_region


def extract_centre_crop(
    hr_img: np.ndarray,
    hr_patch_size: int,
    scale: int,
) -> np.ndarray:
    """
    Extract a centre crop from hr_img, aligned to scale grid.
    Used in validation mode for deterministic evaluation.

    Returns:
        hr_crop: (H', W', 3) where H', W' are the largest multiples
                 of (hr_patch_size * scale) that fit in the image.
    """
    H, W, _ = hr_img.shape
    step = hr_patch_size  # crop in multiples of patch size

    crop_h = (H // step) * step
    crop_w = (W // step) * step

    top  = (H - crop_h) // 2
    left = (W - crop_w) // 2

    return hr_img[top:top + crop_h, left:left + crop_w, :]


def extract_tiled_patches(
    img: np.ndarray,
    patch_size: int,
    overlap: int = 16,
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """
    Tile image into overlapping patches for full-image inference.
    Overlap avoids visible boundary artefacts when stitching.

    Args:
        img:        (H, W, 3) float32 image (LR space)
        patch_size: tile size in pixels
        overlap:    overlap between adjacent tiles

    Returns:
        patches: list of (patch_size, patch_size, 3) arrays
        coords:  list of (top, left, bottom, right) in original img coords
    """
    H, W, _ = img.shape
    stride = patch_size - overlap

    patches, coords = [], []

    for top in range(0, H, stride):
        for left in range(0, W, stride):
            bottom = min(top  + patch_size, H)
            right  = min(left + patch_size, W)
            top_adj  = max(0, bottom - patch_size)
            left_adj = max(0, right  - patch_size)

            patch = img[top_adj:bottom, left_adj:right, :]
            patches.append(patch)
            coords.append((top_adj, left_adj, bottom, right))

    return patches, coords
