from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

from backend.core.constants import ALLOWED_IMAGE_TYPES, PROCESSED_DIR
from backend.inference.platform_optimizer import PLATFORM_RULES
from backend.services.file_service import FileService
from backend.services.optimization_service import OptimizationService
from backend.schemas.optimization import OptimizationResponse


router = APIRouter(prefix="/optimize", tags=["Optimization"])

optimization_service = OptimizationService()


@router.post("", response_model=OptimizationResponse)
async def optimize_image(
    platform: str = Form(...),
    file: UploadFile = File(...),
):
    if platform not in PLATFORM_RULES:
        raise HTTPException(status_code=400, detail="Unsupported platform")

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, PNG, and WEBP images are allowed",
        )

    try:
        raw_path = await FileService.save_upload_file(file)

        result = optimization_service.optimize(
            input_path=str(raw_path),
            platform=platform,
        )

        final_filename = Path(result["final_output"]).name

        return {
            "status": "completed",
            "platform": platform,
            "download_url": f"/optimize/download/{final_filename}",
            "metadata": result["metadata"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{filename}")
def download_optimized_image(filename: str):
    file_path = PROCESSED_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        filename=filename,
    )