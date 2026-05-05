from pathlib import Path

from backend.core.constants import PROCESSED_DIR
from backend.inference.optimization_pipeline import ImageOptimizationPipeline


class OptimizationService:
    def __init__(self):
        self.pipeline = ImageOptimizationPipeline()

    def optimize(self, input_path: str, platform: str) -> dict:
        return self.pipeline.process(
            input_path=input_path,
            platform=platform,
            output_dir=str(PROCESSED_DIR),
        )