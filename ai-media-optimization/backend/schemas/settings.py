from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_env: str

    # Storage
    gcp_project_id: str
    gcs_bucket: str

    # Kafka
    kafka_broker: str
    kafka_topic: str

    # ML
    model_path: str

    # Monitoring
    prometheus_port: int

    #database
    database_url: str

    class Config:
        env_file = ".env"


@lru_cache
def get_settings():
    return Settings()