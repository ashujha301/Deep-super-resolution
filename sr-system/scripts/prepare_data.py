from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
from src.data.validator import DatasetValidator, ValidationConfig
from src.data.downsampler import Downsampler, DownsampleConfig
from src.data.patcher import PatchExtractor, PatchConfig
from src.data.loader import SRBatchLoader, LoaderConfig
from src.data.augmentation import AugmentationConfig

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare SR HR/LR training data.")
    p.add_argument("--dataset", default="div2k")
    p.add_argument("--raw-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--scale", default=2, type=int)
    p.add_argument("--hr-patch-size", default=48, type=int)
    p.add_argument("--stride", default=48, type=int)
    p.add_argument("--batch-size", default=8, type=int)
    p.add_argument("--degradation", default="bicubic", choices=["bicubic", "bicubic_noise", "bicubic_jpeg"])
    p.add_argument("--patch-strategy", default="grid", choices=["grid", "random", "importance"])
    p.add_argument("--random-patches-per-image", default=50, type=int)
    p.add_argument("--importance-patches-per-image", default=50, type=int)
    p.add_argument("--backend", default="numpy", choices=["numpy", "cupy"])
    p.add_argument("--max-images", default=None, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--report-path", default=Path("reports/pipeline_report.json"), type=Path)
    p.add_argument("--save-samples", action="store_true", default=True)
    p.add_argument("--force", action="store_true")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    metadata_dir = output_dir / "metadata"
    lr_dir = output_dir / "lr" 
    patch_dir = output_dir / "patches"
    samples_dir = args.report_path.parent / "samples"
    metadata_dir.mkdir(parents=True, exist_ok=True) 
    args.report_path.parent.mkdir(parents=True, exist_ok=True); 
    samples_dir.mkdir(parents=True, exist_ok=True)

    validation = DatasetValidator(ValidationConfig(
        raw_dir=args.raw_dir, 
        output_metadata_dir=metadata_dir, 
        min_width=args.hr_patch_size*2, 
        min_height=args.hr_patch_size*2)).validate(max_images=args.max_images)
    
    downsample_summary = Downsampler(DownsampleConfig(
        output_lr_dir=lr_dir, scale=args.scale, 
        degradation=args.degradation, 
        seed=args.seed, 
        force=args.force)).process_records(validation["valid_records"])
    
    patch_index = PatchExtractor(PatchConfig(
        output_patch_dir=patch_dir, 
        scale=args.scale, 
        hr_patch_size=args.hr_patch_size, 
        stride=args.stride, 
        strategy=args.patch_strategy, 
        random_patches_per_image=args.random_patches_per_image, 
        importance_patches_per_image=args.importance_patches_per_image, 
        seed=args.seed, force=args.force
        )).process_pairs(downsample_summary["generated_records"])
    
    loader_summary = inspect_loader(metadata_dir / "patch_index.json", args.batch_size, args.backend, args.seed)

    sample_summary = save_samples(patch_index, samples_dir) if args.save_samples else {}

    report = {
        "dataset": args.dataset, 
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(), 
        "raw_dir": str(args.raw_dir), 
        "output_dir": str(output_dir), 
        "seed": args.seed, 
        "scale_factor": args.scale, 
        "validation": compact_validation(validation), 
        "downsampling": compact_downsampling(downsample_summary), 
        "patching": compact_patching(patch_index), 
        "loader": loader_summary, 
        "samples": sample_summary
        }
    
    args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))

def inspect_loader(patch_index_path: Path, batch_size: int, backend: str, seed: int) -> dict:

    if not patch_index_path.exists(): return {"available": False, "reason": "patch_index_not_found"}

    loader = SRBatchLoader(LoaderConfig(
        patch_index_path=patch_index_path, 
        batch_size=batch_size, 
        backend=backend, 
        shuffle=False, 
        drop_last=True, 
        augment=False, 
        seed=seed), 
        AugmentationConfig(enabled=False, seed=seed))
    
    if len(loader) == 0: return {"available": False, "reason": "not_enough_patches_for_one_batch", "batch_size": batch_size}

    hr, lr = next(iter(loader))

    if backend == "cupy":
        return {
            "available": True, 
            "backend": backend, 
            "batch_size": batch_size, 
            "estimated_batches_per_epoch": len(loader), 
            "hr_batch_shape": list(hr.shape), 
            "lr_batch_shape": list(lr.shape), 
            "hr_dtype": str(hr.dtype), 
            "lr_dtype": str(lr.dtype), 
            "normalization_range_observed": {"hr": [float(hr.min().get()), float(hr.max().get())], "lr": [float(lr.min().get()), float(lr.max().get())]}
            }
    
    return {
        "available": True, 
        "backend": backend, 
        "batch_size": batch_size, 
        "estimated_batches_per_epoch": len(loader), 
        "hr_batch_shape": list(hr.shape), 
        "lr_batch_shape": list(lr.shape), 
        "hr_dtype": str(hr.dtype), 
        "lr_dtype": str(lr.dtype), 
        "normalization_range_observed": {"hr": [float(hr.min()), float(hr.max())], "lr": [float(lr.min()), float(lr.max())]}
        }

def save_samples(patch_index: dict, samples_dir: Path) -> dict:
    patches = patch_index.get("patches", [])
    if not patches: return {"saved": False, "reason": "no_patches"}
    import numpy as np
    first = patches[0]
    hr = np.load(first["hr_patch_path"]); 
    lr = np.load(first["lr_patch_path"])
    hr_img = Image.fromarray(hr); 
    lr_img = Image.fromarray(lr); 
    preview = lr_img.resize(hr_img.size, Image.Resampling.BICUBIC)
    hr_path = samples_dir / "sample_hr.png"; 
    lr_path = samples_dir / "sample_lr.png"; 
    preview_path = samples_dir / "sample_lr_upscaled_preview.png"
    hr_img.save(hr_path); 
    lr_img.save(lr_path); 
    preview.save(preview_path)

    return {"saved": True, "sample_hr": str(hr_path), "sample_lr": str(lr_path), "sample_lr_upscaled_preview": str(preview_path)}

def compact_validation(v: dict) -> dict:
    return {
        "total_images_found": v["total_images_found"], 
        "valid_images": v["valid_images"], 
        "rejected_images": v["rejected_images"], 
        "rejection_reasons": summarize_rejections(v["rejected_records"]), 
        "valid_images_path": "data/processed/metadata/valid_images.json", 
        "rejected_images_path": "data/processed/metadata/rejected_images.json"
        }

def compact_downsampling(s: dict) -> dict:
    return {
        "degradation": s["degradation"], 
        "scale": s["scale"], 
        "lr_output_dir": s["lr_output_dir"], 
        "generated_count": s["generated_count"], 
        "skipped_count": s["skipped_count"], 
        "failed_count": s["failed_count"]
        }

def compact_patching(i: dict) -> dict:
    total = i["total_patches"]; unique_sources = len({p["source_hr_path"] for p in i.get("patches", [])})
    return {
        "scale": i["scale"], 
        "hr_patch_size": [i["hr_patch_size"], i["hr_patch_size"]], 
        "lr_patch_size": [i["lr_patch_size"], i["lr_patch_size"]], 
        "strategy": i["strategy"], 
        "total_patches_extracted": total, 
        "average_patches_per_image": (total / unique_sources) if unique_sources else 0, 
        "patch_index_path": "data/processed/metadata/patch_index.json", 
        "failed_count": len(i.get("failed_records", []))
        }

def summarize_rejections(records: list[dict]) -> dict:
    counts = {}
    for rec in records:
        reason = rec.get("reason", "unknown"); counts[reason] = counts.get(reason, 0) + 1
    return counts

if __name__ == "__main__": main()
