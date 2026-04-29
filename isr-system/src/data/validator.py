from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Literal
import json
from PIL import Image, UnidentifiedImageError

SplitName = Literal["train", "valid", "test", "unknown"]

@dataclass(frozen=True)
class ValidationConfig:
    raw_dir: Path
    output_metadata_dir: Path
    min_width: int = 96
    min_height: int = 96
    supported_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    train_subdir: str = "train_hr"
    valid_subdir: str = "valid_hr"
    allow_unknown_layout: bool = True

@dataclass(frozen=True)
class ValidImageRecord:
    path: str
    split: SplitName
    width: int
    height: int
    mode: str
    filename: str

@dataclass(frozen=True)
class RejectedImageRecord:
    path: str
    split: SplitName
    reason: str
    details: str

class DatasetValidator:
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.config.output_metadata_dir.mkdir(parents=True, exist_ok=True)

    def discover_images(self) -> list[tuple[Path, SplitName]]:
        raw_dir = Path(self.config.raw_dir)
        candidates: list[tuple[Path, SplitName]] = []
        split_dirs = [(raw_dir / self.config.train_subdir, "train"), (raw_dir / self.config.valid_subdir, "valid")]
        found_known_layout = False
        for split_dir, split in split_dirs:
            if split_dir.exists():
                found_known_layout = True
                candidates.extend((p, split) for p in self._walk_files(split_dir))
        if not found_known_layout and self.config.allow_unknown_layout:
            candidates.extend((p, "unknown") for p in self._walk_files(raw_dir))
        return sorted(candidates, key=lambda item: str(item[0]))

    def validate(self, max_images: int | None = None) -> dict:
        valid: list[ValidImageRecord] = []
        rejected: list[RejectedImageRecord] = []
        discovered = self.discover_images()
        if max_images is not None:
            discovered = discovered[:max_images]
        for path, split in discovered:
            result = self._validate_one(path, split)
            if isinstance(result, ValidImageRecord):
                valid.append(result)
            else:
                rejected.append(result)
        summary = {
            "total_images_found": len(discovered),
            "valid_images": len(valid),
            "rejected_images": len(rejected),
            "valid_records": [asdict(x) for x in valid],
            "rejected_records": [asdict(x) for x in rejected],
        }
        self._write_outputs(summary)
        return summary

    def _validate_one(self, path: Path, split: SplitName) -> ValidImageRecord | RejectedImageRecord:
        if path.suffix.lower() not in self.config.supported_extensions:
            return RejectedImageRecord(str(path), split, "unsupported_extension", path.suffix)
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                width, height = img.size
                mode = img.mode
        except UnidentifiedImageError as exc:
            return RejectedImageRecord(str(path), split, "cannot_open", str(exc))
        except OSError as exc:
            return RejectedImageRecord(str(path), split, "corrupted_file", str(exc))
        if mode != "RGB":
            return RejectedImageRecord(str(path), split, "not_rgb", f"mode={mode}")
        if width < self.config.min_width or height < self.config.min_height:
            return RejectedImageRecord(str(path), split, "too_small", f"{width}x{height}")
        return ValidImageRecord(str(path), split, width, height, mode, path.name)

    def _walk_files(self, directory: Path) -> Iterable[Path]:
        if not directory.exists():
            return []
        return (p for p in directory.rglob("*") if p.is_file())

    def _write_outputs(self, summary: dict) -> None:
        metadata_dir = self.config.output_metadata_dir
        (metadata_dir / "valid_images.json").write_text(json.dumps(summary["valid_records"], indent=2), encoding="utf-8")
        (metadata_dir / "rejected_images.json").write_text(json.dumps(summary["rejected_records"], indent=2), encoding="utf-8")
