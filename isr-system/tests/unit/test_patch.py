# ==================================================
# tests/unit/test_patch.py
# Unit tests for src/data/patch.py
# Run: pytest tests/unit/test_patch.py -v
# ==================================================

import numpy as np
import pytest
from src.data.patch import (
    extract_random_patch,
    extract_centre_crop,
    extract_tiled_patches,
)


@pytest.fixture
def hr_image():
    np.random.seed(0)
    return np.random.rand(256, 256, 3).astype(np.float32)


class TestExtractRandomPatch:
    @pytest.mark.parametrize("scale", [2, 4, 8])
    def test_output_shapes(self, hr_image, scale):
        ps = 64
        hr, lr = extract_random_patch(hr_image, ps, scale)
        assert hr.shape == (ps, ps, 3)
        assert lr.shape == (ps // scale, ps // scale, 3)

    def test_values_in_range(self, hr_image):
        hr, lr = extract_random_patch(hr_image, 64, 4)
        assert 0.0 <= hr.min() and hr.max() <= 1.0
        assert 0.0 <= lr.min() and lr.max() <= 1.0

    def test_too_small_raises(self):
        tiny = np.random.rand(32, 32, 3).astype(np.float32)
        with pytest.raises(ValueError, match="smaller than patch size"):
            extract_random_patch(tiny, hr_patch_size=64, scale=4)

    def test_lr_hr_alignment(self, hr_image):
        """
        Bicubic LR from extract_random_patch must closely match
        an independently-generated bicubic LR of the same HR patch.
        PSNR > 35 dB confirms pixel-level alignment is correct.
        """
        import cv2
        hr, lr = extract_random_patch(hr_image, 64, 4)
        lr_ref = cv2.resize(hr, (16, 16), interpolation=cv2.INTER_CUBIC)
        mse = np.mean((lr - lr_ref) ** 2)
        psnr = -10 * np.log10(mse + 1e-8)
        assert psnr > 35.0, f"Alignment broken: PSNR={psnr:.1f} dB"


class TestExtractCentreCrop:
    def test_output_divisible_by_patch(self):
        img = np.random.rand(300, 400, 3).astype(np.float32)
        crop = extract_centre_crop(img, hr_patch_size=128, scale=4)
        assert crop.shape[0] % 128 == 0
        assert crop.shape[1] % 128 == 0

    def test_crop_smaller_than_original(self, hr_image):
        crop = extract_centre_crop(hr_image, hr_patch_size=128, scale=4)
        assert crop.shape[0] <= hr_image.shape[0]
        assert crop.shape[1] <= hr_image.shape[1]


class TestExtractTiledPatches:
    def test_patch_count_positive(self, hr_image):
        patches, coords = extract_tiled_patches(hr_image, patch_size=64)
        assert len(patches) > 0
        assert len(patches) == len(coords)

    def test_all_patches_correct_size(self, hr_image):
        patches, _ = extract_tiled_patches(hr_image, patch_size=64)
        for p in patches:
            assert p.shape == (64, 64, 3)
