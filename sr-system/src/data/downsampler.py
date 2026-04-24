from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal
import io
import numpy as np
from PIL import Image

DegradationMode = Literal["bicubic", "bicubic_noise", "bicubic_jpeg"]

@dataclass(frozen=True)
class DownsampleConfig:
    output_lr_dir: Path
    scale: int = 2
    degradation: DegradationMode = "bicubic"
    noise_sigma: float = 2.0
    jpeg_quality: int = 85
    gamma: float = 2.2
    seed: int = 42
    force: bool = False

class Downsampler:
    def __init__(self, config: DownsampleConfig):
        if config.scale <= 0:
            raise ValueError("scale must be positive")
        if not 1 <= config.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.output_dir = Path(config.output_lr_dir) / config.degradation
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_records(self, valid_records: list[dict]) -> dict:
        generated, skipped, failed = [], [], []
        for record in valid_records:
            hr_path = Path(record["path"])
            try:
                out_path = self.output_path_for(hr_path)
                if out_path.exists() and not self.config.force:
                    skipped.append({"hr_path": str(hr_path), "lr_path": str(out_path), "reason": "exists"})
                    generated.append(self._record(hr_path, out_path))
                    continue
                self.downsample_file(hr_path, out_path)
                generated.append(self._record(hr_path, out_path))
            except Exception as exc:
                failed.append({"hr_path": str(hr_path), "reason": type(exc).__name__, "details": str(exc)})
        return {
            "degradation": self.config.degradation,
            "scale": self.config.scale,
            "lr_output_dir": str(self.output_dir),
            "generated_count": len(generated),
            "skipped_count": len(skipped),
            "failed_count": len(failed),
            "generated_records": generated,
            "skipped_records": skipped,
            "failed_records": failed,
            "config": {**asdict(self.config), "output_lr_dir": str(self.config.output_lr_dir)},
        }

    def downsample_file(self, hr_path: Path, out_path: Path) -> None:
        with Image.open(hr_path) as img:
            lr = self.downsample_image(img.convert("RGB"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lr.save(out_path)

    def downsample_image(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        lr_size = (width // self.config.scale, height // self.config.scale)
        if lr_size[0] <= 0 or lr_size[1] <= 0:
            raise ValueError(f"image too small for scale={self.config.scale}: {width}x{height}")
        arr = np.asarray(img).astype(np.float32) / 255.0
        linear = np.power(np.clip(arr, 0.0, 1.0), self.config.gamma)
        linear_img = Image.fromarray(np.uint8(np.clip(linear * 255.0, 0, 255)))
        resized_linear_img = linear_img.resize(lr_size, Image.Resampling.BICUBIC)
        resized_linear = np.asarray(resized_linear_img).astype(np.float32) / 255.0
        srgb = np.power(np.clip(resized_linear, 0.0, 1.0), 1.0 / self.config.gamma)
        out = np.clip(srgb * 255.0, 0, 255).astype(np.uint8)
        if self.config.degradation == "bicubic_noise":
            out = self._add_noise(out)
        elif self.config.degradation == "bicubic_jpeg":
            out = self._jpeg_roundtrip(out)
        elif self.config.degradation != "bicubic":
            raise ValueError(f"Unsupported degradation mode: {self.config.degradation}")
        return Image.fromarray(out, mode="RGB")

    def output_path_for(self, hr_path: Path) -> Path:
        return self.output_dir / f"{hr_path.stem}_x{self.config.scale}_{self.config.degradation}.png"

    def _add_noise(self, arr: np.ndarray) -> np.ndarray:
        noise = self.rng.normal(0.0, self.config.noise_sigma, size=arr.shape)
        return np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def _jpeg_roundtrip(self, arr: np.ndarray) -> np.ndarray:
        img = Image.fromarray(arr, mode="RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=self.config.jpeg_quality)
        buffer.seek(0)
        return np.asarray(Image.open(buffer).convert("RGB"), dtype=np.uint8)

    def _record(self, hr_path: Path, lr_path: Path) -> dict:
        with Image.open(hr_path) as hr_img, Image.open(lr_path) as lr_img:
            return {"hr_path": str(hr_path), "lr_path": str(lr_path), "hr_size": list(hr_img.size), "lr_size": list(lr_img.size)}
