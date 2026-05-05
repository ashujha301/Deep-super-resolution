from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image


class BackgroundRemovalService:
    def __init__(
        self,
        model_path: str = "backend/models/rmbg/rmbg_fp16.onnx",
        input_size: tuple[int, int] = (1024, 1024),
    ):
        self.model_path = Path(model_path)
        self.input_size = input_size

        if not self.model_path.exists():
            raise FileNotFoundError(f"RMBG model not found: {self.model_path}")

        providers = ["CPUExecutionProvider"]

        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            print("✅ RMBG using ONNX CUDA")
        else:
            print("⚠️ RMBG using ONNX CPU")

        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB").resize(self.input_size, Image.LANCZOS)

        arr = np.array(image).astype(np.float32) / 255.0
        arr = arr - 0.5

        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, axis=0)

        return arr.astype(np.float32)

    def _postprocess(self, output: np.ndarray, original_size: tuple[int, int]) -> Image.Image:
        mask = np.squeeze(output)

        mask = mask - mask.min()
        if mask.max() > 0:
            mask = mask / mask.max()

        mask = (mask * 255).astype(np.uint8)

        return Image.fromarray(mask).resize(original_size, Image.LANCZOS)

    def extract_foreground(
        self,
        input_path: str,
        output_path: str,
        mask_output_path: str | None = None,
    ) -> dict:
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        image = Image.open(input_path).convert("RGB")
        original_size = image.size

        input_tensor = self._preprocess(image)
        outputs = self.session.run(None, {self.input_name: input_tensor})

        mask = self._postprocess(outputs[0], original_size)

        foreground = image.convert("RGBA")
        foreground.putalpha(mask)
        foreground.save(output_path)

        if mask_output_path:
            mask_path = Path(mask_output_path)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            mask.save(mask_path)

        return {
            "foreground_path": str(output_path),
            "mask_path": mask_output_path,
            "background_removed": True,
            "width": original_size[0],
            "height": original_size[1],
        }