from pathlib import Path
import json
import numpy as np
from src.data.loader import SRBatchLoader, LoaderConfig
from src.data.augmentation import AugmentationConfig

def test_loader_returns_normalized_chw_batches(tmp_path: Path):
    hr_dir = tmp_path / "patches" / "hr"; lr_dir = tmp_path / "patches" / "lr"; meta_dir = tmp_path / "metadata"
    hr_dir.mkdir(parents=True); lr_dir.mkdir(parents=True); meta_dir.mkdir(parents=True)
    records = []
    for i in range(8):
        hr_path = hr_dir / f"p{i}_hr.npy"; lr_path = lr_dir / f"p{i}_lr.npy"
        np.save(hr_path, np.full((48, 48, 3), 255, dtype=np.uint8))
        np.save(lr_path, np.full((24, 24, 3), 128, dtype=np.uint8))
        records.append({"hr_patch_path": str(hr_path), "lr_patch_path": str(lr_path)})
    index_path = meta_dir / "patch_index.json"
    index_path.write_text(json.dumps({"patches": records}), encoding="utf-8")
    loader = SRBatchLoader(LoaderConfig(patch_index_path=index_path, batch_size=8, shuffle=False, augment=False), AugmentationConfig(enabled=False))
    hr, lr = next(iter(loader))
    assert hr.shape == (8, 3, 48, 48)
    assert lr.shape == (8, 3, 24, 24)
    assert hr.dtype == np.float32 and lr.dtype == np.float32
    assert hr.max() == 1.0
