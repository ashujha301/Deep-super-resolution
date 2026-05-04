from backend.schemas.settings import get_settings
from backend.utils.config_loader import load_yaml_config

settings = get_settings()
yaml_config = load_yaml_config()

print("ENV:", settings.app_env)
print("Kafka:", settings.kafka_broker)
print("YAML max upload:", yaml_config["image"]["max_upload"])