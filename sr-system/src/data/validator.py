import os
import json
from PIL import Image, UnidentifiedImageError
import numpy as np
from pathlib import Path
from tqdm import tqdm


class DatasetValidator:
    def __init__(self, data_dir, patch_size=48, scale=2):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.scale = scale

        self.min_size = patch_size * scale * 4  # important rule

        self.valid_files = []
        self.invalid_files = []
        self.rejected_files = []
        self.converted_files = []

    # ---- Validate dataset
    def validate(self):
        files = list(self.data_dir.glob("*.png"))

        for file in tqdm(files, desc="Validating images"):
            try:
                img = Image.open(file)

                # ---- Channel handling
                if img.mode == "L":
                    img = img.convert("RGB")
                    self.converted_files.append(str(file))

                elif img.mode == "RGBA":
                    img = img.convert("RGB")
                    self.converted_files.append(str(file))

                elif img.mode != "RGB":
                    self.invalid_files.append(str(file))
                    continue

                w, h = img.size

                # ---- Resolution check
                if w < self.min_size or h < self.min_size:
                    self.rejected_files.append(str(file))
                    continue

                self.valid_files.append(str(file))

            except UnidentifiedImageError:
                self.invalid_files.append(str(file))

        return self._generate_report()

    # ---- Compute stats (sample 100 images)
    def compute_stats(self, sample_size=20):
        import random

        sample_files = random.sample(self.valid_files, min(sample_size, len(self.valid_files)))

        pixels = []

        for file in sample_files:
            img = Image.open(file).resize((256, 256))  # downscale for speed
            img = np.array(img).astype(np.float32)
            pixels.append(img.reshape(-1, 3))

        pixels = np.concatenate(pixels, axis=0)

        stats = {
            "mean": pixels.mean(axis=0).tolist(),
            "std": pixels.std(axis=0).tolist(),
        }

        return stats

    # ---- Generate report
    def _generate_report(self):
        report = {
            "valid_count": len(self.valid_files),
            "invalid_count": len(self.invalid_files),
            "rejected_count": len(self.rejected_files),
            "converted_count": len(self.converted_files),
        }

        return report

    # ---- Save report
    def save_report(self, output_path):
        report = self._generate_report()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=4)

        print("\n---- Validation Summary ----")
        for k, v in report.items():
            print(f"{k}: {v}")