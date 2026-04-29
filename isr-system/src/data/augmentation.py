from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool = True
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.5
    rotate: bool = True
    seed: int = 42

class PairedAugmenter:
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def __call__(self, hr: np.ndarray, lr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.config.enabled:
            return hr, lr
        do_h = self.rng.random() < self.config.horizontal_flip_prob
        do_v = self.rng.random() < self.config.vertical_flip_prob
        k = int(self.rng.integers(0, 4)) if self.config.rotate else 0
        return self.apply_with_params(hr, lr, do_h=do_h, do_v=do_v, rot_k=k)

    @staticmethod
    def apply_with_params(hr: np.ndarray, lr: np.ndarray, *, do_h: bool = False, do_v: bool = False, rot_k: int = 0) -> tuple[np.ndarray, np.ndarray]:
        if do_h:
            hr = np.flip(hr, axis=1); lr = np.flip(lr, axis=1)
        if do_v:
            hr = np.flip(hr, axis=0); lr = np.flip(lr, axis=0)
        if rot_k:
            hr = np.rot90(hr, k=rot_k, axes=(0, 1)); lr = np.rot90(lr, k=rot_k, axes=(0, 1))
        return np.ascontiguousarray(hr), np.ascontiguousarray(lr)
