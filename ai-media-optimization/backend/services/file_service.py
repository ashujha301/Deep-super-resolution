from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from backend.core.constants import RAW_DIR


class FileService:
    @staticmethod
    async def save_upload_file(file: UploadFile) -> Path:
        RAW_DIR.mkdir(parents=True, exist_ok=True)

        file_ext = file.filename.split(".")[-1].lower()
        raw_filename = f"{uuid4()}.{file_ext}"
        raw_path = RAW_DIR / raw_filename

        with open(raw_path, "wb") as buffer:
            buffer.write(await file.read())

        return raw_path