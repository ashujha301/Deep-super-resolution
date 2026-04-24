from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal
import json
import numpy as np
from PIL import Image

PatchStrategy = Literal["grid", "random", "importance"]

@dataclass(frozen=True)
class PatchConfig:
    output_patch_dir: Path
    scale: int = 2
    hr_patch_size: int = 48
    stride: int = 48
    strategy: PatchStrategy = "grid"
    random_patches_per_image: int = 50
    importance_patches_per_image: int = 50
    seed: int = 42
    force: bool = False
    @property
    def lr_patch_size(self) -> int:
        if self.hr_patch_size % self.scale != 0:
            raise ValueError("hr_patch_size must be divisible by scale")
        return self.hr_patch_size // self.scale

class PatchExtractor:
    def __init__(self, config: PatchConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.hr_dir = Path(config.output_patch_dir) / "hr"
        self.lr_dir = Path(config.output_patch_dir) / "lr"
        self.metadata_dir = Path(config.output_patch_dir).parent / "metadata"
        self.hr_dir.mkdir(parents=True, exist_ok=True)
        self.lr_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def process_pairs(self, pair_records: list[dict]) -> dict:
        patch_records, failed = [], []
        for pair in pair_records:
            try:
                patch_records.extend(self.extract_pair(Path(pair["hr_path"]), Path(pair["lr_path"])))
            except Exception as exc:
                failed.append({"hr_path": pair.get("hr_path"), "lr_path": pair.get("lr_path"), "reason": type(exc).__name__, "details": str(exc)})
        index = {
            "scale": self.config.scale,
            "hr_patch_size": self.config.hr_patch_size,
            "lr_patch_size": self.config.lr_patch_size,
            "strategy": self.config.strategy,
            "total_patches": len(patch_records),
            "patches": patch_records,
            "failed_records": failed,
            "config": {**asdict(self.config), "output_patch_dir": str(self.config.output_patch_dir)},
        }
        (self.metadata_dir / "patch_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
        return index

    def extract_pair(self, hr_path: Path, lr_path: Path) -> list[dict]:
        hr = np.asarray(Image.open(hr_path).convert("RGB"), dtype=np.uint8)
        lr = np.asarray(Image.open(lr_path).convert("RGB"), dtype=np.uint8)
        self._validate_pair_shapes(hr, lr, hr_path, lr_path)
        records = []
        for local_idx, (y, x) in enumerate(self._coordinates(hr)):
            ly, lx = y // self.config.scale, x // self.config.scale
            hr_patch = hr[y:y+self.config.hr_patch_size, x:x+self.config.hr_patch_size, :]
            lr_patch = lr[ly:ly+self.config.lr_patch_size, lx:lx+self.config.lr_patch_size, :]
            if hr_patch.shape != (self.config.hr_patch_size, self.config.hr_patch_size, 3):
                continue
            if lr_patch.shape != (self.config.lr_patch_size, self.config.lr_patch_size, 3):
                continue
            base = f"{hr_path.stem}_patch_{local_idx:06d}"
            hr_out = self.hr_dir / f"{base}_hr.npy"
            lr_out = self.lr_dir / f"{base}_lr.npy"
            if self.config.force or not hr_out.exists(): np.save(hr_out, hr_patch)
            if self.config.force or not lr_out.exists(): np.save(lr_out, lr_patch)
            records.append({
                "source_hr_path": str(hr_path), "source_lr_path": str(lr_path),
                "hr_patch_path": str(hr_out), "lr_patch_path": str(lr_out),
                "patch_index": local_idx, "hr_xy": [int(x), int(y)], "lr_xy": [int(lx), int(ly)],
                "hr_shape": list(hr_patch.shape), "lr_shape": list(lr_patch.shape), "strategy": self.config.strategy,
            })
        return records

    def _coordinates(self, hr: np.ndarray) -> list[tuple[int, int]]:
        if self.config.strategy == "grid": return self._grid_coordinates(hr)
        if self.config.strategy == "random": return self._random_coordinates(hr, self.config.random_patches_per_image)
        if self.config.strategy == "importance": return self._importance_coordinates(hr, self.config.importance_patches_per_image)
        raise ValueError(f"Unsupported patch strategy: {self.config.strategy}")

    def _grid_coordinates(self, hr: np.ndarray) -> list[tuple[int, int]]:
        h, w = hr.shape[:2]; size = self.config.hr_patch_size; stride = self.config.stride
        coords = [(y, x) for y in range(0, h-size+1, stride) for x in range(0, w-size+1, stride)]
        return coords or ([(0, 0)] if h >= size and w >= size else [])

    def _random_coordinates(self, hr: np.ndarray, count: int) -> list[tuple[int, int]]:
        h, w = hr.shape[:2]; size = self.config.hr_patch_size
        if h < size or w < size: return []
        ys = self.rng.integers(0, h-size+1, size=count); xs = self.rng.integers(0, w-size+1, size=count)
        return [(int(y), int(x)) for y, x in zip(ys, xs)]

    def _importance_coordinates(self, hr: np.ndarray, count: int) -> list[tuple[int, int]]:
        h, w = hr.shape[:2]; size = self.config.hr_patch_size
        if h < size or w < size: return []
        gray = hr.astype(np.float32).mean(axis=2)
        gy, gx = np.gradient(gray)
        score = np.sqrt(gx * gx + gy * gy)[:h-size+1, :w-size+1]
        flat = score.reshape(-1) + 1e-6
        probs = flat / flat.sum()
        idx = self.rng.choice(len(flat), size=count, replace=True, p=probs)
        ys, xs = np.unravel_index(idx, score.shape)
        return [(int(y), int(x)) for y, x in zip(ys, xs)]

    def _validate_pair_shapes(self, hr: np.ndarray, lr: np.ndarray, hr_path: Path, lr_path: Path) -> None:
        expected_lr_h = hr.shape[0] // self.config.scale
        expected_lr_w = hr.shape[1] // self.config.scale
        if lr.shape[0] != expected_lr_h or lr.shape[1] != expected_lr_w:
            raise ValueError(f"HR/LR shape mismatch: {hr_path.name} {hr.shape}, {lr_path.name} {lr.shape}, expected LR {expected_lr_h}x{expected_lr_w}")
