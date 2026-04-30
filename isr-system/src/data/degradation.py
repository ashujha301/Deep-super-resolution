# ==================================================
# src/data/degradation.py
# Responsibility: LR image generation pipelines.
#
#   generate_lr_bicubic()     → EDSR (deterministic)
#   generate_lr_realesrgan()  → Real-ESRGAN (stochastic)
#
# Why separated from patch.py:
#   Degradation is a MODEL-SPECIFIC concern.
#   patch.py handles geometry; degradation.py handles pixel quality.
#   Easier to swap degradation strategy without touching patch logic.
# ==================================================

import random
from typing import Optional, Tuple

import cv2
import numpy as np


# --------------------------------------------------
# EDSR: Simple bicubic (deterministic, cacheable)
# --------------------------------------------------

def generate_lr_bicubic(hr_img: np.ndarray, scale: int) -> np.ndarray:
    """
    Bicubic downsampling. Used for EDSR training.
    Deterministic — safe to pre-generate and cache to disk.

    Args:
        hr_img: (H, W, 3) float32 [0, 1]
        scale:  downsampling factor (2, 3, 4, or 8)

    Returns:
        lr_img: (H//scale, W//scale, 3) float32 [0, 1]
    """
    H, W, _ = hr_img.shape
    lr_h, lr_w = H // scale, W // scale
    return cv2.resize(
        hr_img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC
    ).astype(np.float32)


# --------------------------------------------------
# Real-ESRGAN: Second-order degradation (stochastic)
# Reference: Wang et al., ICCV 2021, Section 3.1
# --------------------------------------------------

def _apply_gaussian_blur(
    img: np.ndarray, kernel_size: int, sigma: float
) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)


def _apply_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0)


def _apply_jpeg_compression(img: np.ndarray, quality: int) -> np.ndarray:
    img_uint8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    img_bgr   = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    _, enc    = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    dec_bgr   = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    dec_rgb   = cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB)
    return dec_rgb.astype(np.float32) / 255.0


def generate_lr_realesrgan(
    hr_img:       np.ndarray,
    scale:        int,
    blur_sigma:   Tuple[float, float] = (0.2, 3.0),
    noise_sigma:  Tuple[float, float] = (0.0, 0.1),
    jpeg_quality: Tuple[int, int]     = (30, 95),
) -> np.ndarray:
    """
    Second-order degradation pipeline (Real-ESRGAN paper):
        HR → Blur₁ → Noise₁ → JPEG₁ → Bicubic↓ → Blur₂ → Noise₂ → JPEG₂ → LR

    All parameters are randomised per call.
    This stochasticity is why Real-ESRGAN generalises to real photos.

    Args:
        hr_img:       (H, W, 3) float32 [0, 1]
        scale:        downscale factor
        blur_sigma:   (min, max) Gaussian blur sigma
        noise_sigma:  (min, max) Gaussian noise sigma
        jpeg_quality: (min, max) JPEG compression quality

    Returns:
        lr_img: degraded LR image float32 [0, 1]
    """
    img = hr_img.copy()

    def rand_blur(x):
        sigma = random.uniform(*blur_sigma)
        ksize = int(2 * round(3 * sigma) + 1)
        return _apply_gaussian_blur(x, ksize, sigma)

    def rand_noise(x):
        sigma = random.uniform(*noise_sigma)
        return _apply_gaussian_noise(x, sigma) if sigma > 0 else x

    def rand_jpeg(x):
        quality = random.randint(*jpeg_quality)
        return _apply_jpeg_compression(x, quality)

    # First-order degradation
    img = rand_blur(img)
    img = rand_noise(img)
    img = rand_jpeg(img)

    # Bicubic downscale
    img = generate_lr_bicubic(img, scale)

    # Second-order degradation
    img = rand_blur(img)
    img = rand_noise(img)
    img = rand_jpeg(img)

    return img
