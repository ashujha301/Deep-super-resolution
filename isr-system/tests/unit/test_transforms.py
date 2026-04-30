# ==================================================
# tests/unit/test_transforms.py
# Unit tests for src/data/transforms.py
# Run: pytest tests/unit/test_transforms.py -v
# ==================================================

import numpy as np
import pytest
import torch
from src.data.transforms import (
    build_training_transforms,
    build_validation_transforms,
    apply_augmentation,
    to_tensor,
    to_numpy,
)


@pytest.fixture
def patch():
    np.random.seed(7)
    return np.random.rand(128, 128, 3).astype(np.float32)


class TestAugmentation:
    def test_preserves_shape(self, patch):
        t = build_training_transforms()
        out = apply_augmentation(patch, t)
        assert out.shape == patch.shape

    def test_preserves_range(self, patch):
        t = build_training_transforms()
        out = apply_augmentation(patch, t)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_preserves_dtype(self, patch):
        t = build_training_transforms()
        out = apply_augmentation(patch, t)
        assert out.dtype == np.float32

    def test_validation_is_identity(self, patch):
        """Validation transform must not alter any pixel."""
        t = build_validation_transforms()
        out = apply_augmentation(patch, t)
        np.testing.assert_array_almost_equal(
            patch, out, decimal=5,
            err_msg="Validation transform must be a no-op!"
        )


class TestTensorConversion:
    def test_to_tensor_shape(self, patch):
        t = to_tensor(patch)
        assert t.shape == (3, 128, 128)

    def test_to_tensor_dtype(self, patch):
        t = to_tensor(patch)
        assert t.dtype == torch.float32

    def test_roundtrip_values(self, patch):
        """to_tensor → to_numpy must be lossless."""
        recovered = to_numpy(to_tensor(patch))
        np.testing.assert_array_almost_equal(patch, recovered, decimal=6)

    def test_to_tensor_range(self, patch):
        t = to_tensor(patch)
        assert t.min() >= 0.0 and t.max() <= 1.0
