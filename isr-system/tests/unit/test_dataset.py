# ==================================================
# tests/unit/test_dataset.py
# Unit tests for src/data/dataset.py
# Run: pytest tests/unit/test_dataset.py -v
# ==================================================

from pathlib import Path
import numpy as np
import pytest
import torch
import cv2
import sys

# Ensure isr-system is in path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import (
    RealESRGANDataset, 
    EDSRDataset,
    get_train_dataset,
    get_valid_dataset,
    TRAIN_HR_DIR,
    VALID_HR_DIR,
)


@pytest.fixture
def img_dir(tmp_path) -> Path:
    """5 synthetic 256×256 PNG images."""
    d = tmp_path / "hr"
    d.mkdir()
    for i in range(5):
        img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
        cv2.imwrite(str(d / f"{i:04d}.png"), img)
    return d


class TestRealESRGANDataset:
    def test_length(self, img_dir):
        ds = RealESRGANDataset(img_dir, scale=4, hr_patch_size=64,
                               patches_per_image=3)
        assert len(ds) == 15  # 5 images × 3 patches

    def test_output_shapes(self, img_dir):
        ds = RealESRGANDataset(img_dir, scale=4, hr_patch_size=64)
        s = ds[0]
        assert s["lr"].shape == (3, 16, 16)
        assert s["hr"].shape == (3, 64, 64)

    def test_output_dtype(self, img_dir):
        ds = RealESRGANDataset(img_dir, scale=4, hr_patch_size=64)
        s = ds[0]
        assert s["lr"].dtype == torch.float32
        assert s["hr"].dtype == torch.float32

    def test_value_range(self, img_dir):
        ds = RealESRGANDataset(img_dir, scale=4, hr_patch_size=64)
        for i in range(5):
            s = ds[i]
            assert s["lr"].min() >= 0.0 and s["lr"].max() <= 1.0
            assert s["hr"].min() >= 0.0 and s["hr"].max() <= 1.0

    def test_invalid_scale_raises(self, img_dir):
        with pytest.raises(ValueError, match="scale must be"):
            RealESRGANDataset(img_dir, scale=7, hr_patch_size=64)

    def test_non_divisible_patch_raises(self, img_dir):
        with pytest.raises(ValueError, match="divisible"):
            RealESRGANDataset(img_dir, scale=4, hr_patch_size=100)

    def test_empty_dir_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(RuntimeError, match="No valid images"):
            RealESRGANDataset(empty, scale=4)


class TestEDSRDataset:
    def test_default_patch_size(self, img_dir):
        ds = EDSRDataset(img_dir, scale=4, cache_in_memory=False)
        s = ds[0]
        assert s["hr"].shape == (3, 96, 96)
        assert s["lr"].shape == (3, 24, 24)

    def test_val_mode_is_deterministic(self, img_dir):
        """In val mode (is_train=False), same index must give same result."""
        ds = EDSRDataset(img_dir, scale=4, hr_patch_size=64,
                         is_train=False, cache_in_memory=False)
        s1 = ds[0]
        s2 = ds[0]
        assert torch.allclose(s1["lr"], s2["lr"]), (
            "Val mode is not deterministic!"
        )


# ==================================================
# Integration Tests with Real Paths
# ==================================================

class TestDatasetPaths:
    """Verify dataset paths are correctly configured."""
    
    def test_train_hr_dir_exists(self):
        """Check if training HR directory exists."""
        if TRAIN_HR_DIR.exists():
            assert TRAIN_HR_DIR.is_dir(), f"{TRAIN_HR_DIR} is not a directory"
            files = list(TRAIN_HR_DIR.glob("*.png")) + list(TRAIN_HR_DIR.glob("*.jpg"))
            assert len(files) > 0, f"No images found in {TRAIN_HR_DIR}"
            print(f"✓ Train dataset: {len(files)} images at {TRAIN_HR_DIR}")
        else:
            pytest.skip(f"Training dataset not found at {TRAIN_HR_DIR}")
    
    def test_valid_hr_dir_exists(self):
        """Check if validation HR directory exists."""
        if VALID_HR_DIR.exists():
            assert VALID_HR_DIR.is_dir(), f"{VALID_HR_DIR} is not a directory"
            files = list(VALID_HR_DIR.glob("*.png")) + list(VALID_HR_DIR.glob("*.jpg"))
            assert len(files) > 0, f"No images found in {VALID_HR_DIR}"
            print(f"✓ Valid dataset: {len(files)} images at {VALID_HR_DIR}")
        else:
            pytest.skip(f"Validation dataset not found at {VALID_HR_DIR}")


class TestFactoryFunctions:
    """Test get_train_dataset and get_valid_dataset factories."""
    
    def test_get_train_dataset_edsr(self):
        """Test EDSR training dataset factory."""
        if not TRAIN_HR_DIR.exists():
            pytest.skip(f"Training dataset not found at {TRAIN_HR_DIR}")
        
        dataset, dataloader = get_train_dataset(
            dataset_type="edsr", scale=4, batch_size=2, num_workers=0
        )
        assert dataset is not None
        assert dataloader is not None
        assert len(dataset) > 0
        print(f"✓ EDSR train dataset created: {len(dataset)} samples")
    
    def test_get_train_dataset_realesrgan(self):
        """Test Real-ESRGAN training dataset factory."""
        if not TRAIN_HR_DIR.exists():
            pytest.skip(f"Training dataset not found at {TRAIN_HR_DIR}")
        
        dataset, dataloader = get_train_dataset(
            dataset_type="realesrgan", scale=4, batch_size=2, num_workers=0
        )
        assert dataset is not None
        assert dataloader is not None
        assert len(dataset) > 0
        print(f"✓ RealESRGAN train dataset created: {len(dataset)} samples")
    
    def test_get_valid_dataset_edsr(self):
        """Test EDSR validation dataset factory."""
        if not VALID_HR_DIR.exists():
            pytest.skip(f"Validation dataset not found at {VALID_HR_DIR}")
        
        dataset, dataloader = get_valid_dataset(
            dataset_type="edsr", scale=4, batch_size=1, num_workers=0
        )
        assert dataset is not None
        assert dataloader is not None
        assert len(dataset) > 0
        print(f"✓ EDSR valid dataset created: {len(dataset)} samples")
    
    def test_get_valid_dataset_realesrgan(self):
        """Test Real-ESRGAN validation dataset factory."""
        if not VALID_HR_DIR.exists():
            pytest.skip(f"Validation dataset not found at {VALID_HR_DIR}")
        
        dataset, dataloader = get_valid_dataset(
            dataset_type="realesrgan", scale=4, batch_size=1, num_workers=0
        )
        assert dataset is not None
        assert dataloader is not None
        assert len(dataset) > 0
        print(f"✓ RealESRGAN valid dataset created: {len(dataset)} samples")
