from pathlib import Path
from PIL import Image, ImageOps, ImageStat
import numpy as np


PLATFORM_RULES = {
    "ecommerce_product": {
        "scale": 4,
        "mode": "center_product",
        "canvas_size": (2000, 2000),
        "background": (255, 255, 255),
        "output_format": "JPEG",
        "quality": 95,
    },
    "instagram_post": {
        "scale": 4,
        "mode": "fit_canvas",
        "canvas_size": (1080, 1350),
        "background": (255, 255, 255),
        "output_format": "JPEG",
        "quality": 90,
    },
    "instagram_story": {
        "scale": 4,
        "mode": "fit_canvas",
        "canvas_size": (1080, 1920),
        "background": (255, 255, 255),
        "output_format": "JPEG",
        "quality": 90,
    },
    "web_lcp": {
        "scale": 2,
        "mode": "resize_max",
        "max_size": (1200, 1200),
        "output_format": "WEBP",
        "quality": 75,
    },
    "mobile_lcp": {
        "scale": 2,
        "mode": "resize_max",
        "max_size": (800, 800),
        "output_format": "WEBP",
        "quality": 70,
    },
}


def estimate_brightness(image: Image.Image) -> float:
    grayscale = image.convert("L")
    stat = ImageStat.Stat(grayscale)
    return round(stat.mean[0] / 255, 4)


def detect_content_bbox(image: Image.Image):
    """
    Basic product/content detection.
    Works best for product images with plain/white background.
    """
    rgb = image.convert("RGB")
    arr = np.array(rgb)

    # Estimate background from image corners
    corners = np.array([
        arr[0, 0],
        arr[0, -1],
        arr[-1, 0],
        arr[-1, -1],
    ])

    bg = np.median(corners, axis=0)

    diff = np.linalg.norm(arr.astype(np.float32) - bg.astype(np.float32), axis=2)

    mask = diff > 30

    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return image.getbbox()

    padding = 20
    left = max(xs.min() - padding, 0)
    top = max(ys.min() - padding, 0)
    right = min(xs.max() + padding, image.width)
    bottom = min(ys.max() + padding, image.height)

    return (left, top, right, bottom)


def center_product_on_canvas(
    image: Image.Image,
    canvas_size: tuple[int, int],
    background: tuple[int, int, int],
) -> Image.Image:
    bbox = detect_content_bbox(image)
    product = image.crop(bbox)

    canvas_w, canvas_h = canvas_size

    # Product should occupy around 80% of canvas
    max_product_w = int(canvas_w * 0.8)
    max_product_h = int(canvas_h * 0.8)

    product.thumbnail((max_product_w, max_product_h), Image.LANCZOS)

    canvas = Image.new("RGB", canvas_size, background)

    x = (canvas_w - product.width) // 2
    y = (canvas_h - product.height) // 2

    canvas.paste(product, (x, y))

    return canvas


def fit_to_canvas(
    image: Image.Image,
    canvas_size: tuple[int, int],
    background: tuple[int, int, int],
) -> Image.Image:
    canvas = Image.new("RGB", canvas_size, background)

    fitted = ImageOps.contain(image, canvas_size, Image.LANCZOS)

    x = (canvas_size[0] - fitted.width) // 2
    y = (canvas_size[1] - fitted.height) // 2

    canvas.paste(fitted, (x, y))

    return canvas


def resize_max(image: Image.Image, max_size: tuple[int, int]) -> Image.Image:
    image = image.copy()
    image.thumbnail(max_size, Image.LANCZOS)
    return image


def optimize_platform_image(
    input_path: str,
    output_path: str,
    platform: str,
) -> dict:
    if platform not in PLATFORM_RULES:
        raise ValueError(f"Unsupported platform: {platform}")

    rule = PLATFORM_RULES[platform]

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(input_path).convert("RGB")

    original_size = image.size
    original_file_size = input_path.stat().st_size

    if rule["mode"] == "center_product":
        image = center_product_on_canvas(
            image=image,
            canvas_size=rule["canvas_size"],
            background=rule["background"],
        )
        actions = [
            "Detected visible product area",
            "Cropped unnecessary background",
            "Centered product on clean square canvas",
            "Preserved high quality for catalogue/product usage",
        ]

    elif rule["mode"] == "fit_canvas":
        image = fit_to_canvas(
            image=image,
            canvas_size=rule["canvas_size"],
            background=rule["background"],
        )
        actions = [
            "Resized image to platform-safe dimensions",
            "Preserved full image without cropping",
            "Added clean padding where required",
            "Optimized for social media upload quality",
        ]

    elif rule["mode"] == "resize_max":
        image = resize_max(
            image=image,
            max_size=rule["max_size"],
        )
        actions = [
            "Reduced dimensions for faster loading",
            "Converted to WebP",
            "Compressed image for LCP/page-speed optimization",
            "Kept visual quality balanced against file size",
        ]

    else:
        raise ValueError(f"Unknown optimization mode: {rule['mode']}")

    save_kwargs = {}

    if rule["output_format"] in ["JPEG", "WEBP"]:
        save_kwargs["quality"] = rule["quality"]
        save_kwargs["optimize"] = True

    image.save(output_path, format=rule["output_format"], **save_kwargs)

    final_file_size = output_path.stat().st_size

    return {
        "platform": platform,
        "original_width": original_size[0],
        "original_height": original_size[1],
        "optimized_width": image.width,
        "optimized_height": image.height,
        "original_file_size_kb": round(original_file_size / 1024, 2),
        "optimized_file_size_kb": round(final_file_size / 1024, 2),
        "size_reduction_percent": round(
            ((original_file_size - final_file_size) / original_file_size) * 100,
            2,
        ),
        "brightness": estimate_brightness(image),
        "output_format": rule["output_format"],
        "quality": rule["quality"],
        "actions": actions,
    }