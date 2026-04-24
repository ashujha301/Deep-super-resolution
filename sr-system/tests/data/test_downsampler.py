from pathlib import Path
from PIL import Image
from src.data.downsampler import Downsampler, DownsampleConfig

def test_downsampler_creates_half_size_image(tmp_path: Path):
    hr = tmp_path / "hr.png"
    Image.new("RGB", (96, 96), color=(100, 120, 130)).save(hr)
    ds = Downsampler(DownsampleConfig(output_lr_dir=tmp_path / "lr", scale=2, degradation="bicubic"))
    out = ds.output_path_for(hr); ds.downsample_file(hr, out)
    with Image.open(out) as img:
        assert img.size == (48, 48)
