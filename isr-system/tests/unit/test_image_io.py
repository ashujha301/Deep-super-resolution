# ==================================================
# tests/unit/test_image_io.py
# Unit tests for src/data/image_io.py
# Run: pytest tests/unit/test_image_io.py -v
# ==================================================

from pathlib import Path
import numpy as np
import pytest
import cv2
from src.data.image_io import (
    load_image_rgb, save_image_rgb,
    validate_image_file, scan_image_directory,
)


@pytest.fixture
def sample_image(tmp_path) -> Path:
    """Save a synthetic PNG and return its path."""
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    path = tmp_path / "sample.png"
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path


@pytest.fixture
def image_dir(tmp_path) -> Path:
    """Create a directory with 5 synthetic PNGs."""
    d = tmp_path / "images"
    d.mkdir()
    for i in range(5):
        img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        cv2.imwrite(str(d / f"{i:04d}.png"), img)
    return d


class TestLoadImageRgb:
    def test_returns_float32(self, sample_image):
        img = load_image_rgb(sample_image)
        assert img.dtype == np.float32

    def test_range_zero_to_one(self, sample_image):
        img = load_image_rgb(sample_image)
        assert img.min() >= 0.0 and img.max() <= 1.0

    def test_shape_is_hwc(self, sample_image):
        img = load_image_rgb(sample_image)
        assert img.ndim == 3 and img.shape[2] == 3

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_image_rgb(Path("/no/such/file.png"))


class TestSaveImageRgb:
    def test_roundtrip_quality(self, tmp_path):
        """PNG roundtrip error must be < 1/255 (uint8 quantisation only)."""
        img = np.random.rand(64, 64, 3).astype(np.float32)
        path = tmp_path / "out.png"
        save_image_rgb(img, path)
        loaded = load_image_rgb(path)
        assert np.abs(img - loaded).max() < (1 / 255 + 1e-6)

    def test_creates_parent_dirs(self, tmp_path):
        img = np.random.rand(32, 32, 3).astype(np.float32)
        path = tmp_path / "nested" / "dir" / "out.png"
        save_image_rgb(img, path)
        assert path.exists()


class TestScanImageDirectory:
    def test_finds_all_images(self, image_dir):
        paths = scan_image_directory(image_dir)
        assert len(paths) == 5

    def test_returns_sorted(self, image_dir):
        paths = scan_image_directory(image_dir)
        assert paths == sorted(paths)

    def test_empty_dir_returns_empty(self, tmp_path):
        paths = scan_image_directory(tmp_path)
        assert paths == []
