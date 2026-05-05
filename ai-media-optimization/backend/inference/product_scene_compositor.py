from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance, ImageStat


BACKGROUND_MAP = {
    "studio": "assets/backgrounds/studio-background.png",
    "marble_table": "assets/backgrounds/Marble-table.png",
    "luxury_bathroom": "assets/backgrounds/luxury-bathroom.png",
    "nature_background": "assets/backgrounds/nature-background.png",
}


BACKGROUND_SCENE_RULES = {
    "studio": {
        "canvas_size": (1600, 1600),
        "product_width_ratio": 0.48,
        "anchor_x": 0.50,
        "anchor_y": 0.70,
        "shadow_offset": (18, 36),
        "shadow_blur": 28,
        "shadow_strength": 0.28,
        "brightness": 1.05,
        "contrast": 1.04,
        "sharpness": 1.05,
    },
    "marble_table": {
        "canvas_size": (1600, 1600),
        "product_width_ratio": 0.34,
        "anchor_x": 0.58,
        "anchor_y": 0.63,
        "shadow_offset": (22, 24),
        "shadow_blur": 24,
        "shadow_strength": 0.34,
        "brightness": 0.90,
        "contrast": 0.96,
        "sharpness": 1.02,
    },
    "luxury_bathroom": {
        "canvas_size": (1200, 1600),
        "product_width_ratio": 0.28,
        "anchor_x": 0.32,
        "anchor_y": 0.78,
        "shadow_offset": (14, 20),
        "shadow_blur": 20,
        "shadow_strength": 0.30,
        "brightness": 1.03,
        "contrast": 1.00,
        "sharpness": 1.03,
    },
    "nature_background": {
        "canvas_size": (1200, 1600),
        "product_width_ratio": 0.40,
        "anchor_x": 0.50,
        "anchor_y": 0.70,
        "shadow_offset": (16, 28),
        "shadow_blur": 26,
        "shadow_strength": 0.36,
        "brightness": 0.92,
        "contrast": 0.98,
        "sharpness": 1.02,
    }
}


class ProductSceneCompositor:
    def _crop_foreground_content(self, fg: Image.Image) -> Image.Image:
        alpha = fg.getchannel("A")
        bbox = alpha.getbbox()
        return fg.crop(bbox) if bbox else fg

    def _fit_background(self, bg: Image.Image, canvas_size: tuple[int, int]) -> Image.Image:
        bg = bg.convert("RGB")

        bg_ratio = bg.width / bg.height
        canvas_ratio = canvas_size[0] / canvas_size[1]

        if bg_ratio > canvas_ratio:
            new_h = canvas_size[1]
            new_w = int(new_h * bg_ratio)
        else:
            new_w = canvas_size[0]
            new_h = int(new_w / bg_ratio)

        bg = bg.resize((new_w, new_h), Image.LANCZOS)

        left = (new_w - canvas_size[0]) // 2
        top = (new_h - canvas_size[1]) // 2

        return bg.crop((left, top, left + canvas_size[0], top + canvas_size[1])).convert("RGBA")

    def _create_contact_shadow(
        self,
        product: Image.Image,
        blur: int,
        strength: float,
    ) -> Image.Image:
        alpha = product.getchannel("A")

        shadow = Image.new("RGBA", product.size, (0, 0, 0, 0))
        shadow.putalpha(alpha)

        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur))

        r, g, b, a = shadow.split()
        a = ImageEnhance.Brightness(a).enhance(strength)
        shadow.putalpha(a)

        return shadow

    def _match_product_lighting(
        self,
        product: Image.Image,
        background: Image.Image,
        rule: dict,
        x: int,
        y: int,
    ) -> Image.Image:
        product = ImageEnhance.Brightness(product).enhance(rule["brightness"])
        product = ImageEnhance.Contrast(product).enhance(rule["contrast"])
        product = ImageEnhance.Sharpness(product).enhance(rule["sharpness"])

        # Simple local brightness matching
        bg_crop = background.crop((
            max(x, 0),
            max(y, 0),
            min(x + product.width, background.width),
            min(y + product.height, background.height),
        )).convert("RGB")

        bg_brightness = ImageStat.Stat(bg_crop.convert("L")).mean[0] / 255
        product_rgb = product.convert("RGB")
        product_brightness = ImageStat.Stat(product_rgb.convert("L")).mean[0] / 255

        if product_brightness > 0:
            factor = bg_brightness / product_brightness
            factor = max(0.82, min(1.18, factor))
            product = ImageEnhance.Brightness(product).enhance(factor)

        return product

    def compose(
        self,
        foreground_path: str,
        background_key: str,
        output_path: str,
    ) -> dict:
        if background_key not in BACKGROUND_MAP:
            raise ValueError(f"Unsupported background: {background_key}")

        rule = BACKGROUND_SCENE_RULES[background_key]
        canvas_size = rule["canvas_size"]

        bg_path = Path(BACKGROUND_MAP[background_key])
        if not bg_path.exists():
            raise FileNotFoundError(f"Background image missing: {bg_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        background = Image.open(bg_path)
        background = self._fit_background(background, canvas_size)

        product = Image.open(foreground_path).convert("RGBA")
        product = self._crop_foreground_content(product)

        target_product_w = int(canvas_size[0] * rule["product_width_ratio"])
        target_product_h = int(product.height * (target_product_w / product.width))

        product = product.resize((target_product_w, target_product_h), Image.LANCZOS)

        x = int(canvas_size[0] * rule["anchor_x"] - product.width / 2)
        y = int(canvas_size[1] * rule["anchor_y"] - product.height)

        product = self._match_product_lighting(product, background, rule, x, y)

        shadow = self._create_contact_shadow(
            product,
            blur=rule["shadow_blur"],
            strength=rule["shadow_strength"],
        )

        shadow_x = x + rule["shadow_offset"][0]
        shadow_y = y + rule["shadow_offset"][1]

        background.alpha_composite(shadow, (shadow_x, shadow_y))
        background.alpha_composite(product, (x, y))

        final = background.convert("RGB")
        final.save(output_path, format="JPEG", quality=94, optimize=True)

        return {
            "background": background_key,
            "output_path": str(output_path),
            "canvas_width": canvas_size[0],
            "canvas_height": canvas_size[1],
            "product_position": {
                "x": x,
                "y": y,
                "width": product.width,
                "height": product.height,
            },
            "actions": [
                "Removed product background",
                "Selected scene-specific product placement",
                "Matched product brightness to background lighting",
                "Added soft contact shadow",
                "Applied scene-specific scale and composition",
            ],
        }