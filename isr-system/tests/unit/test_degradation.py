# ==================================================
# tests/unit/test_degradation.py
# Unit tests for src/data/degradation.py
# Run: pytest tests/unit/test_degradation.py -v
# ==================================================

import numpy as np
import pytest
from src.data.degradation import generate_lr_bicubic, generate_lr_realesrgan


@pytest.fixture
def hr_patch():
    np.random.seed(42)
    return np.random.rand(128, 128, 3).astype(np.float32)


class TestBicubicLR:
    @pytest.mark.parametrize("scale", [2, 4, 8])
    def test_output_shape(self, hr_patch, scale):
        lr = generate_lr_bicubic(hr_patch, scale)
        expected = (128 // scale, 128 // scale, 3)
        assert lr.shape == expected

    def test_output_range(self, hr_patch):
        lr = generate_lr_bicubic(hr_patch, 4)
        assert lr.min() >= 0.0 and lr.max() <= 1.0

    def test_is_deterministic(self, hr_patch):
        """Same input must always produce same output."""
        lr1 = generate_lr_bicubic(hr_patch, 4)
        lr2 = generate_lr_bicubic(hr_patch, 4)
        np.testing.assert_array_equal(lr1, lr2)

    def test_output_dtype(self, hr_patch):
        lr = generate_lr_bicubic(hr_patch, 4)
        assert lr.dtype == np.float32


class TestRealESRGANDegradation:
    def test_output_shape(self, hr_patch):
        lr = generate_lr_realesrgan(hr_patch, scale=4)
        assert lr.shape == (32, 32, 3)

    def test_output_range(self, hr_patch):
        lr = generate_lr_realesrgan(hr_patch, scale=4)
        assert lr.min() >= 0.0 and lr.max() <= 1.0

    def test_is_stochastic(self, hr_patch):
        """
        Two calls on same input must differ — stochasticity is
        the core feature of Real-ESRGAN degradation.
        If this fails, randomness in the pipeline is broken.
        """
        lr1 = generate_lr_realesrgan(hr_patch, 4)
        lr2 = generate_lr_realesrgan(hr_patch, 4)
        assert not np.allclose(lr1, lr2), (
            "Degradation produced identical outputs — randomness broken!"
        )

    def test_output_dtype(self, hr_patch):
        lr = generate_lr_realesrgan(hr_patch, 4)
        assert lr.dtype == np.float32

    @pytest.mark.parametrize("scale", [2, 4])
    def test_various_scales(self, hr_patch, scale):
        lr = generate_lr_realesrgan(hr_patch, scale)
        assert lr.shape == (128 // scale, 128 // scale, 3)
