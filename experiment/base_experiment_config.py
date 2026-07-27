import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    model_name: str = "openai-community/gpt2"
    dataset_path: str = "Salesforce/wikitext"
    dataset_name: str = "wikitext-2-raw-v1"
    iterations_intervals: list[int] | None = None
    learning_rate: float = 1e-5
    result_folder: str = "results"
    model_save_path: str | None = None
    model_id: str = "v1"
    training_output_save_path: str | None = None
    train_output_id: str = "v1"
    model_diffusion_steps: int = 1000
    corruption_diffusion_steps: int = 100
    max_length: int = 64
    batch_size: int = 95

    def __post_init__(self) -> None:
        if self.iterations_intervals is None:
            self.iterations_intervals = [5, 2, 2]
        if self.model_save_path is None:
            self.model_save_path = f"{self.result_folder}/models"
        if self.training_output_save_path is None:
            self.training_output_save_path = f"{self.result_folder}/training_output"

def load_config_values(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}

    with path.open() as config_file:
        raw_config = json.load(config_file)

    if not isinstance(raw_config, dict):
        raise ValueError("Config JSON must be an object.")

    if "overrides" in raw_config:
        raw_config = raw_config["overrides"]

    if not isinstance(raw_config, dict):
        raise ValueError("'overrides' must be an object.")

    config = {}
    allowed_keys = set(ExperimentConfig.__dataclass_fields__)
    for key, value in raw_config.items():
        if key not in allowed_keys:
            allowed = ", ".join(sorted(allowed_keys))
            raise ValueError(f"Unknown config key '{key}'. Allowed keys: {allowed}")
        config[key] = value

    return config


def parse_set_override(raw_override: str) -> tuple[str, Any]:
    if "=" not in raw_override:
        raise ValueError("--set values must use key=value format.")

    key, raw_value = raw_override.split("=", 1)
    if key not in ExperimentConfig.__dataclass_fields__:
        raise ValueError(f"Unknown --set key '{key}'.")

    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value

    return key, value


def print_config(config: ExperimentConfig) -> None:
    print("Experiment config:")
    for key in sorted(ExperimentConfig.__dataclass_fields__):
        print(f"  {key} = {getattr(config, key)!r}")
