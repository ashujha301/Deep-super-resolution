from pydantic import BaseModel


class OptimizationResponse(BaseModel):
    status: str
    platform: str
    download_url: str
    metadata: dict