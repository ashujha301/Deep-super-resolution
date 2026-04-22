import numpy as np
from PIL import Image
import io


class DownSampler:
    def __init__(
        self,
        scale_factor=2,
        mode="bicubic",
        noise_std=10,
        jpeg_quality=(65, 95),
        linear_space=False,
        seed=42
    ):
        self.scale = scale_factor
        self.mode = mode
        self.noise_std = noise_std
        self.jpeg_quality = jpeg_quality
        self.linear_space = linear_space
        self.rng = np.random.default_rng(seed)

    # ---- sRGB → Linear
    def _to_linear(self, img):
        return np.power(img / 255.0, 2.2)

    # ---- Linear → sRGB
    def _to_srgb(self, img):
        return np.clip(np.power(img, 1 / 2.2) * 255.0, 0, 255)

    # ---- Bicubic resize
    def _bicubic_resize(self, img):
        h, w = img.shape[:2]
        new_size = (w // self.scale, h // self.scale)

        pil_img = Image.fromarray(img.astype(np.uint8))
        lr_img = pil_img.resize(new_size, Image.BICUBIC)

        return np.array(lr_img).astype(np.float32)

    # ---- Gaussian noise
    def _add_noise(self, img):
        noise = self.rng.normal(0, self.noise_std, img.shape)
        return np.clip(img + noise, 0, 255)

    # ---- JPEG compression
    def _jpeg_compress(self, img):
        quality = self.rng.integers(self.jpeg_quality[0], self.jpeg_quality[1])

        buffer = io.BytesIO()
        Image.fromarray(img.astype(np.uint8)).save(buffer, format="JPEG", quality=int(quality))
        buffer.seek(0)

        return np.array(Image.open(buffer)).astype(np.float32)

    # ---- Main degradation
    def degrade(self, hr_image):
        img = hr_image.astype(np.float32)

        # ---- Linear space (optional)
        if self.linear_space:
            img = self._to_linear(img)

        # ---- Downsample
        if self.mode in ["bicubic", "bicubic_noise", "bicubic_jpeg"]:
            lr = self._bicubic_resize(img)
        elif self.mode == "nearest":
            h, w = img.shape[:2]
            new_size = (w // self.scale, h // self.scale)
            lr = np.array(
                Image.fromarray(img.astype(np.uint8)).resize(new_size, Image.NEAREST)
            ).astype(np.float32)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # ---- Convert back from linear
        if self.linear_space:
            lr = self._to_srgb(lr)

        # ---- Add noise
        if self.mode == "bicubic_noise":
            lr = self._add_noise(lr)

        # ---- JPEG artifacts
        if self.mode == "bicubic_jpeg":
            lr = self._jpeg_compress(lr)

        return lr

    # ---- Batch processing
    def degrade_batch(self, hr_images):
        return [self.degrade(img) for img in hr_images]