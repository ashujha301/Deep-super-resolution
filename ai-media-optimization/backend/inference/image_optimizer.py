from pathlib import Path
from PIL import Image


def optimize_for_platform(
    input_path: str,
    output_path: str,
    rule: dict,
) -> dict:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(input_path).convert("RGB")

    original_width, original_height = image.size

    image.thumbnail(
        (rule["max_width"], rule["max_height"]),
        Image.LANCZOS,
    )

    optimized_width, optimized_height = image.size

    output_format = rule["output_format"]
    quality = rule["quality"]

    save_kwargs = {}

    if output_format in ["JPEG", "WEBP"]:
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True

    image.save(output_path, format=output_format, **save_kwargs)

    return {
        "original_width": original_width,
        "original_height": original_height,
        "optimized_width": optimized_width,
        "optimized_height": optimized_height,
        "output_format": output_format,
        "quality": quality,
        "output_path": str(output_path),
    }