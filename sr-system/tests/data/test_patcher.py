from pathlib import Path
from PIL import Image
import numpy as np
from src.data.patcher import PatchExtractor, PatchConfig

def test_grid_patcher_extracts_aligned_pair(tmp_path: Path):
    hr_path = tmp_path / "hr.png"; lr_path = tmp_path / "lr.png"
    Image.fromarray(np.zeros((96, 96, 3), dtype=np.uint8)).save(hr_path)
    Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8)).save(lr_path)
    extractor = PatchExtractor(PatchConfig(output_patch_dir=tmp_path / "patches", hr_patch_size=48, stride=48, strategy="grid"))
    records = extractor.extract_pair(hr_path, lr_path)
    assert len(records) == 4
    assert np.load(records[0]["hr_patch_path"]).shape == (48, 48, 3)
    assert np.load(records[0]["lr_patch_path"]).shape == (24, 24, 3)
