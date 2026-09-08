import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from corruption_utils.factory import (
    CORRUPTION_METHOD_ORDER,
    validate_corruption_methods,
)

@dataclass
class TrainMultiExperimentConfig:
    mode: str = "train_multi"
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
    num_diffusion_steps: int = 100
    train_diffusion_steps: int = 1
    rollout_loss_decay: float = 0.5
    corruption_methods: list[str] = field(
        default_factory=lambda: list(CORRUPTION_METHOD_ORDER)
    )
    corruption_minimum_probability: float = 0.01
    corruption_maximum_probability: float = 0.95
    similar_number_of_neighbors: int = 20
    max_length: int = 64
    batch_size: int = 95

    def __post_init__(self) -> None:
        if self.mode != "train_multi" and self.mode != "train":
            raise ValueError("Training config mode must be 'train_multi' or 'train'.")
        if self.iterations_intervals is None:
            self.iterations_intervals = [5, 2, 2]
        if self.model_save_path is None:
            self.model_save_path = f"{self.result_folder}/models"
        if self.training_output_save_path is None:
            self.training_output_save_path = f"{self.result_folder}/training_output"
        self.corruption_methods = validate_corruption_methods(self.corruption_methods)
        if len(self.iterations_intervals) != len(self.corruption_methods):
            raise ValueError(
                "iterations_intervals must contain one value per corruption method."
            )
        if any(interval < 0 for interval in self.iterations_intervals):
            raise ValueError("iterations_intervals values must be positive.")
        if self.train_diffusion_steps < 1:
            raise ValueError("train_diffusion_steps must be greater than 0.")
        if self.num_diffusion_steps <= 1:
            raise ValueError("num_diffusion_steps must be greater than 1.")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not 0 < self.rollout_loss_decay <= 1:
            raise ValueError("rollout_loss_decay must be in (0, 1].")
        if not 0 < self.corruption_minimum_probability <= self.corruption_maximum_probability <= 1:
            raise ValueError(
                "corruption probabilities must satisfy "
                "0 < minimum <= maximum <= 1."
            )
        if self.similar_number_of_neighbors <= 0:
            raise ValueError("similar_number_of_neighbors must be positive.")


def set_config_value(config: dict[str, Any], key: str, value: Any) -> None:
    allowed_keys = set(TrainMultiExperimentConfig.__dataclass_fields__)

    if key not in allowed_keys:
        allowed = ", ".join(sorted(allowed_keys))
        raise ValueError(f"Unknown config key '{key}'. Allowed keys: {allowed}")

    if key in config and config[key] != value:
        raise ValueError(
            f"Conflicting values for '{key}': "
            f"{config[key]!r} and {value!r}."
        )

    config[key] = value


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
    for key, value in raw_config.items():
        set_config_value(config, key, value)

    return config


def parse_set_override(raw_override: str) -> tuple[str, Any]:
    if "=" not in raw_override:
        raise ValueError("--set values must use key=value format.")

    key, raw_value = raw_override.split("=", 1)
    if key not in TrainMultiExperimentConfig.__dataclass_fields__:
        allowed = ", ".join(sorted(TrainMultiExperimentConfig.__dataclass_fields__))
        raise ValueError(f"Unknown --set key '{key}'. Allowed keys: {allowed}")

    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value

    return key, value


def print_config(config: TrainMultiExperimentConfig) -> None:
    print("Experiment config:")
    for key in sorted(TrainMultiExperimentConfig.__dataclass_fields__):
        print(f"  {key} = {getattr(config, key)!r}")
