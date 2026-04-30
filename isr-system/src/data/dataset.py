# ==================================================
# src/data/dataset.py
# Responsibility: PyTorch Dataset classes.
#
#   DIV2KDataset       — base class (shared logic)
#   RealESRGANDataset  — uses stochastic degradation
#   EDSRDataset        — uses bicubic degradation
#
# Each class has ONE job: map an index to a (lr, hr) tensor pair.
# All helpers (I/O, patching, degradation) live in their own modules.
# ==================================================

import logging
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.image_io import load_image_rgb, scan_image_directory
from src.data.patch import extract_random_patch, extract_centre_crop
from src.data.degradation import generate_lr_bicubic, generate_lr_realesrgan
from src.data.transforms import (
    build_training_transforms,
    build_validation_transforms,
    apply_augmentation,
    to_tensor,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------
# Data paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # isr-system/
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "div2k"
TRAIN_HR_DIR = DATA_DIR / "train_HR"
VALID_HR_DIR = DATA_DIR / "valid_HR"


# --------------------------------------------------
# Worker seed initialisation
# Each DataLoader worker must get a unique seed.
# Without this, all workers share the same random state
# → identical augmentation + degradation sequences.
# --------------------------------------------------

def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


# --------------------------------------------------
# Base Dataset
# --------------------------------------------------

class DIV2KDataset(Dataset):
    """
    Base Dataset for DIV2K super-resolution.

    Args:
        hr_dir:            directory of HR images
        scale:             SR scale factor (2, 4, or 8)
        hr_patch_size:     HR patch size in pixels
        patches_per_image: random patches sampled per image per epoch
        is_train:          True = random crop + augment; False = centre crop
        cache_in_memory:   load all HR images into RAM at init
                           (DIV2K ~3.3 GB — fits in 16 GB RAM)
    """

    VALID_SCALES = (2, 3, 4, 8)

    def __init__(
        self,
        hr_dir:            Path,
        scale:             int,
        hr_patch_size:     int  = 128,
        patches_per_image: int  = 10,
        is_train:          bool = True,
        cache_in_memory:   bool = False,
    ):
        super().__init__()

        self.scale             = scale
        self.hr_patch_size     = hr_patch_size
        self.patches_per_image = patches_per_image
        self.is_train          = is_train

        self._validate_config()

        self.image_paths = scan_image_directory(Path(hr_dir))
        if not self.image_paths:
            raise RuntimeError(f"No valid images found in {hr_dir}")

        self.transform = (
            build_training_transforms() if is_train
            else build_validation_transforms()
        )

        # Optional in-memory cache — eliminates disk I/O per batch
        self._cache: Optional[Dict[int, np.ndarray]] = None
        if cache_in_memory:
            logger.info("Loading all HR images into RAM...")
            self._cache = {
                i: load_image_rgb(p) for i, p in enumerate(self.image_paths)
            }
            logger.info(f"Cached {len(self._cache)} images.")

        logger.info(
            f"{self.__class__.__name__} | "
            f"{len(self.image_paths)} images | "
            f"scale=×{scale} | patch={hr_patch_size}px | "
            f"total={len(self)} samples"
        )

    def _validate_config(self) -> None:
        if self.scale not in self.VALID_SCALES:
            raise ValueError(
                f"scale must be one of {self.VALID_SCALES}, got {self.scale}"
            )
        if self.hr_patch_size % self.scale != 0:
            raise ValueError(
                f"hr_patch_size ({self.hr_patch_size}) must be divisible "
                f"by scale ({self.scale}) for pixel-perfect LR alignment."
            )

    def __len__(self) -> int:
        return len(self.image_paths) * self.patches_per_image

    def _load(self, idx: int) -> np.ndarray:
        if self._cache:
            return self._cache[idx].copy()
        return load_image_rgb(self.image_paths[idx])

    def _generate_lr(self, hr_patch: np.ndarray) -> np.ndarray:
        """Override in subclasses. Default: bicubic (EDSR)."""
        return generate_lr_bicubic(hr_patch, self.scale)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Pipeline:
            load HR → crop → augment → generate LR → tensors
        """
        img_idx = idx % len(self.image_paths)
        hr_img  = self._load(img_idx)

        if self.is_train:
            hr_patch, _ = extract_random_patch(
                hr_img, self.hr_patch_size, self.scale
            )
            hr_patch = apply_augmentation(hr_patch, self.transform)
        else:
            hr_patch = extract_centre_crop(
                hr_img, self.hr_patch_size, self.scale
            )

        lr_patch = self._generate_lr(hr_patch)

        return {"lr": to_tensor(lr_patch), "hr": to_tensor(hr_patch)}


# --------------------------------------------------
# Real-ESRGAN Dataset
# --------------------------------------------------

class RealESRGANDataset(DIV2KDataset):
    """
    Dataset for Real-ESRGAN training.
    Applies stochastic second-order degradation per sample.
    """

    def __init__(
        self,
        hr_dir:            Path,
        scale:             int              = 4,
        hr_patch_size:     int              = 128,
        patches_per_image: int              = 10,
        is_train:          bool             = True,
        cache_in_memory:   bool             = False,
        blur_sigma:        Tuple[float, float] = (0.2, 3.0),
        noise_sigma:       Tuple[float, float] = (0.0, 0.1),
        jpeg_quality:      Tuple[int, int]     = (30, 95),
    ):
        super().__init__(
            hr_dir=hr_dir, scale=scale, hr_patch_size=hr_patch_size,
            patches_per_image=patches_per_image, is_train=is_train,
            cache_in_memory=cache_in_memory,
        )
        self.blur_sigma   = blur_sigma
        self.noise_sigma  = noise_sigma
        self.jpeg_quality = jpeg_quality

    def _generate_lr(self, hr_patch: np.ndarray) -> np.ndarray:
        return generate_lr_realesrgan(
            hr_patch, self.scale,
            blur_sigma=self.blur_sigma,
            noise_sigma=self.noise_sigma,
            jpeg_quality=self.jpeg_quality,
        )


# --------------------------------------------------
# EDSR Dataset
# --------------------------------------------------

class EDSRDataset(DIV2KDataset):
    """
    Dataset for EDSR training.
    Uses bicubic degradation (inherited from base class).
    cache_in_memory=True recommended — HR fits in 16 GB RAM.
    """

    def __init__(
        self,
        hr_dir:            Path,
        scale:             int  = 4,
        hr_patch_size:     int  = 96,
        patches_per_image: int  = 10,
        is_train:          bool = True,
        cache_in_memory:   bool = True,
    ):
        super().__init__(
            hr_dir=hr_dir, scale=scale, hr_patch_size=hr_patch_size,
            patches_per_image=patches_per_image, is_train=is_train,
            cache_in_memory=cache_in_memory,
        )
        # _generate_lr not overridden → bicubic from base class


# --------------------------------------------------
# DataLoader factory
# --------------------------------------------------

def build_dataloader(
    dataset:             Dataset,
    batch_size:          int,
    num_workers:         int  = 4,
    shuffle:             bool = True,
    pin_memory:          bool = True,
    prefetch_factor:     int  = 2,
    persistent_workers:  bool = True,
) -> torch.utils.data.DataLoader:
    """
    Build a DataLoader with production-grade settings.
    Always passes worker_init_fn to ensure unique seeds per worker.

    Set num_workers=0 during debugging for clean tracebacks.
    """
    if num_workers == 0:
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size,
            shuffle=shuffle, num_workers=0,
        )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )


# --------------------------------------------------
# Dataset Factory Functions
# --------------------------------------------------

def get_train_dataset(
    dataset_type: str = "edsr",
    scale: int = 4,
    batch_size: int = 16,
    num_workers: int = 4,
) -> Tuple[Dataset, torch.utils.data.DataLoader]:
    """
    Create training dataset and dataloader from DIV2K train_HR directory.
    
    Args:
        dataset_type: "edsr" (bicubic) or "realesrgan" (stochastic degradation)
        scale: SR scale factor (2, 3, 4, or 8)
        batch_size: training batch size
        num_workers: dataloader workers
    
    Returns:
        (dataset, dataloader) tuple
    """
    if not TRAIN_HR_DIR.exists():
        raise RuntimeError(
            f"Training dataset not found at {TRAIN_HR_DIR}\n"
            f"Expected: isr-system/data/raw/div2k/train_HR"
        )
    
    if dataset_type.lower() == "edsr":
        dataset = EDSRDataset(
            hr_dir=TRAIN_HR_DIR,
            scale=scale,
            hr_patch_size=96,
            patches_per_image=10,
            is_train=True,
            cache_in_memory=True,
        )
    elif dataset_type.lower() == "realesrgan":
        dataset = RealESRGANDataset(
            hr_dir=TRAIN_HR_DIR,
            scale=scale,
            hr_patch_size=128,
            patches_per_image=10,
            is_train=True,
            cache_in_memory=True,
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    
    return dataset, dataloader


def get_valid_dataset(
    dataset_type: str = "edsr",
    scale: int = 4,
    batch_size: int = 4,
    num_workers: int = 2,
) -> Tuple[Dataset, torch.utils.data.DataLoader]:
    """
    Create validation dataset and dataloader from DIV2K valid_HR directory.
    
    Args:
        dataset_type: "edsr" (bicubic) or "realesrgan" (stochastic degradation)
        scale: SR scale factor (2, 3, 4, or 8)
        batch_size: validation batch size
        num_workers: dataloader workers
    
    Returns:
        (dataset, dataloader) tuple
    """
    if not VALID_HR_DIR.exists():
        raise RuntimeError(
            f"Validation dataset not found at {VALID_HR_DIR}\n"
            f"Expected: isr-system/data/raw/div2k/valid_HR"
        )
    
    if dataset_type.lower() == "edsr":
        dataset = EDSRDataset(
            hr_dir=VALID_HR_DIR,
            scale=scale,
            hr_patch_size=96,
            patches_per_image=1,
            is_train=False,
            cache_in_memory=False,
        )
    elif dataset_type.lower() == "realesrgan":
        dataset = RealESRGANDataset(
            hr_dir=VALID_HR_DIR,
            scale=scale,
            hr_patch_size=128,
            patches_per_image=1,
            is_train=False,
            cache_in_memory=False,
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    
    return dataset, dataloader
