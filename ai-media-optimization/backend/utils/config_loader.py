import yaml


def load_yaml_config(path: str = "configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)