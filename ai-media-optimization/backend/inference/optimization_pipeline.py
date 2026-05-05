from pathlib import Path
from uuid import uuid4

from backend.inference.realesrgan_service import RealESRGANService
from backend.inference.platform_optimizer import PLATFORM_RULES, optimize_platform_image


class ImageOptimizationPipeline:
    def __init__(self):
        self.sr_service = RealESRGANService()

    def process(
        self,
        input_path: str,
        platform: str,
        output_dir: str = "storage/processed",
    ) -> dict:
        if platform not in PLATFORM_RULES:
            raise ValueError(f"Unsupported platform: {platform}")

        rule = PLATFORM_RULES[platform]
        job_id = str(uuid4())

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        temp_sr_path = output_dir / f"{job_id}_sr.png"

        output_format = rule["output_format"].lower()
        final_ext = "jpg" if output_format == "jpeg" else output_format

        final_path = output_dir / f"{job_id}_{platform}.{final_ext}"

        self.sr_service.enhance_image(
            input_path=input_path,
            output_path=str(temp_sr_path),
            scale=rule["scale"],
        )

        metadata = optimize_platform_image(
            input_path=str(temp_sr_path),
            output_path=str(final_path),
            platform=platform,
        )

        return {
            "job_id": job_id,
            "platform": platform,
            "sr_output": str(temp_sr_path),
            "final_output": str(final_path),
            "metadata": metadata,
        }