from pathlib import Path
from PIL import Image
from src.data.validator import DatasetValidator, ValidationConfig

def test_validator_accepts_rgb_image(tmp_path: Path):
    raw = tmp_path / "raw" / "train_hr"; raw.mkdir(parents=True)
    Image.new("RGB", (128, 128), color=(10, 20, 30)).save(raw / "img.png")
    out = tmp_path / "metadata"
    result = DatasetValidator(ValidationConfig(raw_dir=tmp_path / "raw", output_metadata_dir=out)).validate()
    assert result["valid_images"] == 1
    assert result["rejected_images"] == 0
    assert (out / "valid_images.json").exists()

def test_validator_rejects_small_image(tmp_path: Path):
    raw = tmp_path / "raw" / "train_hr"; raw.mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(raw / "tiny.png")
    result = DatasetValidator(ValidationConfig(raw_dir=tmp_path / "raw", output_metadata_dir=tmp_path / "metadata")).validate()
    assert result["valid_images"] == 0
    assert result["rejected_records"][0]["reason"] == "too_small"
