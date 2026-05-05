from pathlib import Path

RAW_DIR = Path("storage/raw")
PROCESSED_DIR = Path("storage/processed")

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}