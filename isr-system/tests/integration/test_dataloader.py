# ==================================================
# tests/integration/test_dataloader.py
# Integration tests: Dataset + DataLoader working together.
#
# Unlike unit tests (which test one function in isolation),
# these tests verify the FULL pipeline:
#   scan → load → patch → augment → degrade → batch → tensor
#
# Run: pytest tests/integration/test_dataloader.py -v
# ==================================================

from pathlib import Path
import numpy as np
import pytest
import torch
import cv2
from src.data.dataset import (
    RealESRGANDataset, EDSRDataset, build_dataloader
)


@pytest.fixture(scope="module")
def img_dir(tmp_path_factory) -> Path:
    """
    20 synthetic 512×512 PNG images.
    scope="module" → created ONCE, reused by all tests in this file.
    Larger than unit test fixtures to stress batching + multiprocessing.
    """
    d = tmp_path_factory.mktemp("integration_hr")
    for i in range(20):
        img = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
        cv2.imwrite(str(d / f"{i:04d}.png"), img)
    return d


class TestDataLoaderBatching:
    def test_realesrgan_batch_shape(self, img_dir):
        ds = RealESRGANDataset(img_dir, scale=4, hr_patch_size=64,
                               patches_per_image=2)
        loader = build_dataloader(ds, batch_size=4, num_workers=0,
                                  shuffle=False)
        batch = next(iter(loader))
        assert batch["lr"].shape == (4, 3, 16, 16)
        assert batch["hr"].shape == (4, 3, 64, 64)

    def test_edsr_batch_shape(self, img_dir):
        ds = EDSRDataset(img_dir, scale=4, hr_patch_size=96,
                         patches_per_image=2, cache_in_memory=False)
        loader = build_dataloader(ds, batch_size=4, num_workers=0,
                                  shuffle=False)
        batch = next(iter(loader))
        assert batch["lr"].shape == (4, 3, 24, 24)
        assert batch["hr"].shape == (4, 3, 96, 96)

    def test_no_nan_in_batches(self, img_dir):
        ds = RealESRGANDataset(img_dir, scale=4, hr_patch_size=64,
                               patches_per_image=2)
        loader = build_dataloader(ds, batch_size=4, num_workers=0,
                                  shuffle=False)
        for i, batch in enumerate(loader):
            assert not torch.isnan(batch["lr"]).any(), f"NaN in LR batch {i}"
            assert not torch.isnan(batch["hr"]).any(), f"NaN in HR batch {i}"
            if i >= 3:
                break

    def test_value_range_across_batches(self, img_dir):
        ds = RealESRGANDataset(img_dir, scale=4, hr_patch_size=64,
                               patches_per_image=2)
        loader = build_dataloader(ds, batch_size=4, num_workers=0,
                                  shuffle=False)
        for i, batch in enumerate(loader):
            assert batch["lr"].min() >= 0.0 and batch["lr"].max() <= 1.0
            assert batch["hr"].min() >= 0.0 and batch["hr"].max() <= 1.0
            if i >= 3:
                break

    def test_drop_last_no_partial_batches(self, img_dir):
        """
        build_dataloader uses drop_last=True.
        Every batch must be exactly batch_size.
        Partial batches introduce noise in gradient accumulation.
        """
        batch_size = 6
        ds = RealESRGANDataset(img_dir, scale=4, hr_patch_size=64,
                               patches_per_image=2)
        loader = build_dataloader(ds, batch_size=batch_size,
                                  num_workers=0, shuffle=False)
        for batch in loader:
            assert batch["lr"].shape[0] == batch_size


class TestMultiprocessDataLoader:
    def test_two_workers_produces_valid_batches(self, img_dir):
        ds = RealESRGANDataset(img_dir, scale=4, hr_patch_size=64,
                               patches_per_image=3, cache_in_memory=False)
        loader = build_dataloader(ds, batch_size=4, num_workers=2,
                                  shuffle=True, pin_memory=False,
                                  persistent_workers=False)
        batch = next(iter(loader))
        assert batch["lr"].shape == (4, 3, 16, 16)
        assert not torch.isnan(batch["lr"]).any()

    def test_worker_diversity(self, img_dir):
        """
        With worker_init_fn correctly seeding each worker,
        batches must NOT all be identical.
        If they are, worker_init_fn is broken.
        """
        ds = RealESRGANDataset(img_dir, scale=4, hr_patch_size=64,
                               patches_per_image=5, cache_in_memory=False)
        loader = build_dataloader(ds, batch_size=2, num_workers=2,
                                  shuffle=False, persistent_workers=False)
        it = iter(loader)
        b1 = next(it)["lr"]
        b2 = next(it)["lr"]
        b3 = next(it)["lr"]
        # At least one batch pair must differ
        assert not (torch.allclose(b1, b2) and torch.allclose(b2, b3)), (
            "All batches identical — check worker_init_fn!"
        )


class TestFullEpochIteration:
    def test_full_epoch_no_error(self, img_dir):
        """Iterate one complete epoch without any exception."""
        ds = RealESRGANDataset(img_dir, scale=4, hr_patch_size=64,
                               patches_per_image=2, cache_in_memory=False)
        loader = build_dataloader(ds, batch_size=4, num_workers=0,
                                  shuffle=True)
        count = 0
        for batch in loader:
            count += 1
        assert count > 0, "Zero batches in full epoch!"

    def test_edsr_cache_full_epoch(self, img_dir):
        """EDSR with cache enabled must complete full epoch."""
        ds = EDSRDataset(img_dir, scale=4, hr_patch_size=96,
                         patches_per_image=2, cache_in_memory=True)
        loader = build_dataloader(ds, batch_size=4, num_workers=0,
                                  shuffle=True)
        count = 0
        for batch in loader:
            assert batch["lr"].shape[1:] == (3, 24, 24)
            count += 1
        assert count > 0
