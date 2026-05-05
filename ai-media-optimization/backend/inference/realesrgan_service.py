from pathlib import Path
import numpy as np
import torch
from PIL import Image
from RealESRGAN import RealESRGAN


class RealESRGANService:
    def __init__(
        self,
        weights_dir: str = "backend/models/realesrgan",
        default_scale: int = 4,
    ):
        self.weights_dir = Path(weights_dir)
        self.default_scale = default_scale
        self.device = self._get_device()

        if not self.weights_dir.exists():
            raise FileNotFoundError(f"Weights folder not found: {self.weights_dir}")

        self.models = {}

    def _get_device(self):
        if torch.cuda.is_available():
            try:
                torch.zeros(1).cuda()
                print("✅ Using GPU")
                return torch.device("cuda")
            except Exception as e:
                print("⚠️ CUDA unstable, falling back to CPU:", e)

        print("⚠️ Using CPU")
        return torch.device("cpu")

    def _get_weight_path(self, scale: int) -> Path:
        weight_path = self.weights_dir / f"RealESRGAN_x{scale}.pth"

        if not weight_path.exists():
            raise FileNotFoundError(f"Missing weights for scale {scale}: {weight_path}")

        return weight_path

    def _load_model(self, scale: int):
        if scale in self.models:
            return self.models[scale]

        weight_path = self._get_weight_path(scale)

        model = RealESRGAN(self.device, scale=scale)
        model.load_weights(str(weight_path), download=False)

        # Force eval + no grad globally
        model.model.float()
        model.model.eval()

        self.models[scale] = model
        return model

    def enhance_image(
        self,
        input_path: str,
        output_path: str,
        scale: int | None = None,
    ) -> str:
        scale = scale or self.default_scale

        if scale not in [2, 4, 8]:
            raise ValueError("Scale must be one of: 2, 4, 8")

        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        image = Image.open(input_path).convert("RGB")

        model = self._load_model(scale)
        with torch.no_grad():
            sr_image = model.predict(image)

        sr_np = np.array(sr_image).astype(np.float32)

        # Fix NaNs and inf
        if np.isnan(sr_np).any() or np.isinf(sr_np).any():
            print("⚠️ NaNs detected — fixing output")
            sr_np = np.nan_to_num(sr_np)

        # Clamp values
        sr_np = np.clip(sr_np, 0, 255)

        sr_image = Image.fromarray(sr_np.astype(np.uint8))

        sr_image.save(output_path)

        return str(output_path)