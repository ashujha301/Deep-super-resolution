# ==================================================
# src/data/image_io.py
# Responsibility: Image loading, saving, validation.
# Nothing else. No augmentation, no LR generation.
# ==================================================

import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def load_image_rgb(path: Path) -> np.ndarray:
    """
    Load image → RGB float32 [0, 1], shape (H, W, 3).
    Uses cv2 (faster than PIL for large images).

    Raises:
        FileNotFoundError: path does not exist
        ValueError: image is corrupt or not 3-channel
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError(f"cv2 failed to decode: {path}")

    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(f"Expected 3-channel, got shape {img_bgr.shape}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb.astype(np.float32) / 255.0


def save_image_rgb(img: np.ndarray, path: Path) -> None:
    """
    Save float32 RGB [0, 1] image to disk as uint8.
    Creates parent directories if they don't exist.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img_uint8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img_bgr)


def validate_image_file(path: Path) -> bool:
    """
    Check file is readable and not corrupt.
    Uses PIL.verify() — reads headers only (fast).
    Returns True if valid, False if corrupt.
    """
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception as e:
        logger.warning(f"Corrupt image skipped: {path} — {e}")
        return False


def scan_image_directory(directory: Path) -> List[Path]:
    """
    Recursively find all valid images in a directory.
    Skips corrupt files and logs a warning for each.

    Returns:
        Sorted list of valid image Paths.
    """
    extensions = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
    all_paths = sorted(
        p for p in Path(directory).rglob("*")
        if p.suffix in extensions
    )
    logger.info(f"Found {len(all_paths)} image files in {directory}")

    valid = [p for p in all_paths if validate_image_file(p)]
    skipped = len(all_paths) - len(valid)

    if skipped:
        logger.warning(f"Skipped {skipped} corrupt images in {directory}")

    logger.info(f"Using {len(valid)} valid images")
    return valid
