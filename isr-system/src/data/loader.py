from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal
import json
import numpy as np
from .augmentation import AugmentationConfig, PairedAugmenter

BackendName = Literal["numpy", "cupy"]

@dataclass(frozen=True)
class LoaderConfig:
    patch_index_path: Path
    batch_size: int = 8
    backend: BackendName = "numpy"
    shuffle: bool = True
    drop_last: bool = True
    augment: bool = True
    seed: int = 42

class SRPatchDataset:
    def __init__(self, patch_index_path: Path):
        self.patch_index_path = Path(patch_index_path)
        data = json.loads(self.patch_index_path.read_text(encoding="utf-8"))
        self.metadata = data
        self.records = data["patches"]
    def __len__(self) -> int:
        return len(self.records)
    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        rec = self.records[idx]
        return np.load(rec["hr_patch_path"]), np.load(rec["lr_patch_path"])

class SRBatchLoader:
    def __init__(self, config: LoaderConfig, augmentation_config: AugmentationConfig | None = None):
        if config.batch_size <= 0: raise ValueError("batch_size must be positive")
        self.config = config
        self.dataset = SRPatchDataset(config.patch_index_path)
        self.rng = np.random.default_rng(config.seed)
        self.augmenter = PairedAugmenter(augmentation_config or AugmentationConfig(enabled=config.augment, seed=config.seed))
        self.xp = self._resolve_backend(config.backend)

    def __iter__(self) -> Iterator[tuple[object, object]]:
        indices = np.arange(len(self.dataset))
        if self.config.shuffle: self.rng.shuffle(indices)
        bs = self.config.batch_size
        for start in range(0, len(indices), bs):
            batch_indices = indices[start:start+bs]
            if self.config.drop_last and len(batch_indices) < bs: continue
            yield self._make_batch(batch_indices)

    def __len__(self) -> int:
        n = len(self.dataset)
        return n // self.config.batch_size if self.config.drop_last else (n + self.config.batch_size - 1) // self.config.batch_size

    def _make_batch(self, indices: np.ndarray) -> tuple[object, object]:
        hrs, lrs = [], []
        for idx in indices:
            hr, lr = self.dataset[int(idx)]
            if self.config.augment: hr, lr = self.augmenter(hr, lr)
            hrs.append(self._normalize_hwc_to_chw(hr)); lrs.append(self._normalize_hwc_to_chw(lr))
        hr_batch = np.stack(hrs, axis=0).astype(np.float32, copy=False)
        lr_batch = np.stack(lrs, axis=0).astype(np.float32, copy=False)
        if self.config.backend == "cupy": return self.xp.asarray(hr_batch), self.xp.asarray(lr_batch)
        return hr_batch, lr_batch

    @staticmethod
    def _normalize_hwc_to_chw(arr: np.ndarray) -> np.ndarray:
        return np.transpose(arr.astype(np.float32) / 255.0, (2, 0, 1)).copy()

    @staticmethod
    def _resolve_backend(backend: BackendName):
        if backend == "numpy": return np
        if backend == "cupy":
            try:
                import cupy as cp
            except ImportError as exc:
                raise ImportError("CuPy backend requested but cupy is not installed") from exc
            return cp
        raise ValueError(f"Unsupported backend: {backend}")
