# ==================================================
# src/data/transforms.py
# Responsibility: Spatial augmentation for HR patches.
#
# Rules:
#   1. Only SPATIAL transforms (flip, rotate) — never colour.
#      Colour transforms break pixel-level L1/L2 loss targets.
#   2. Augmentation is applied to HR BEFORE LR generation,
#      so both share the same spatial transformation.
#   3. Validation pipeline is always identity (no-op).
# ==================================================

import numpy as np
import albumentations as A


def build_training_transforms() -> A.Compose:
    """
    Spatial augmentation pipeline for training.
    Applied to HR patch before LR degradation is generated.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.3),
    ])


def build_validation_transforms() -> A.Compose:
    """
    No-op pipeline for validation.
    Must be deterministic for reproducible PSNR/SSIM.
    """
    return A.Compose([])


def apply_augmentation(img: np.ndarray, transform: A.Compose) -> np.ndarray:
    """
    Apply albumentations transform to a float32 [0,1] image.
    Converts to uint8 internally (albumentations requirement)
    then converts back to float32 [0,1].

    Args:
        img:       (H, W, 3) float32 [0, 1]
        transform: albumentations Compose pipeline

    Returns:
        augmented: (H, W, 3) float32 [0, 1]
    """
    img_uint8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    result = transform(image=img_uint8)["image"]
    return result.astype(np.float32) / 255.0


# --------------------------------------------------
# Tensor conversion utilities
# Kept here so there is exactly ONE place to change
# if CHW ↔ HWC convention ever needs updating.
# --------------------------------------------------

def to_tensor(img: np.ndarray) -> "torch.Tensor":
    """(H, W, 3) float32 numpy → (3, H, W) float32 torch.Tensor."""
    import torch
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


def to_numpy(tensor: "torch.Tensor") -> np.ndarray:
    """(3, H, W) torch.Tensor → (H, W, 3) float32 numpy."""
    return tensor.detach().cpu().numpy().transpose(1, 2, 0)
