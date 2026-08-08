import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalExperimentConfig:
    mode: str = "eval"
    model_name: str = "openai-community/gpt2"
    models_path: list[str] = field(default_factory=list)
    dataset_path: str | None = None
    dataset_name: str = "cimec/lambada"
    dataset_split: str = "test"
    corruption_methods: list[str] = field(
        default_factory=lambda: ["similar", "mask", "random"]
    )
    corruption_rates: list[float] = field(
        default_factory=lambda: [0.25, 0.50, 0.75, 0.90, 0.95]
    )
    # Optional method-specific format. Each value may be a scalar or a list.
    corruptions: dict[str, float | list[float]] | None = None
    result_folder: str = "results/eval"
    output_file: str = "evaluation_results.json"
    num_diffusion_steps: int = 100
    max_length: int = 64
    batch_size: int = 32
    corruption_grid: list[tuple[str, float]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode != "eval":
            raise ValueError("Evaluation config mode must be 'eval'.")
        if not self.models_path:
            raise ValueError("models_path must contain at least one checkpoint.")
        allowed_methods = {"similar", "mask", "random"}
        if self.corruptions is not None:
            self.corruption_methods = list(self.corruptions)
            method_rates = {
                method: value if isinstance(value, list) else [value]
                for method, value in self.corruptions.items()
            }
            self.corruption_rates = sorted({rate for rates in method_rates.values() for rate in rates})
        else:
            method_rates = {
                method: self.corruption_rates
                for method in self.corruption_methods
            }

        unknown_methods = set(self.corruption_methods) - allowed_methods
        if unknown_methods:
            raise ValueError(f"Unknown corruption method(s): {sorted(unknown_methods)}")
        if not self.corruption_methods or not self.corruption_rates:
            raise ValueError("corruption_methods and corruption_rates cannot be empty.")
        if any(rate <= 0 or rate > 1 for rates in method_rates.values() for rate in rates):
            raise ValueError("corruption_rates must be in the interval (0, 1].")
        self.corruption_grid = [
            (method, rate)
            for method in self.corruption_methods
            for rate in method_rates[method]
        ]
        if self.num_diffusion_steps <= 1:
            raise ValueError("num_diffusion_steps must be greater than 1.")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")


def set_config_value(config: dict[str, Any], key: str, value: Any) -> None:
    allowed_keys = set(EvalExperimentConfig.__dataclass_fields__)

    if key not in allowed_keys:
        raise ValueError(f"Unknown config key '{key}'. Allowed keys: {allowed_keys}")

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
    if key not in EvalExperimentConfig.__dataclass_fields__:
        allowed = ", ".join(sorted(set(EvalExperimentConfig.__dataclass_fields__)))
        raise ValueError(f"Unknown --set key '{key}'. Allowed keys: {allowed}")

    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value

    return key, value


def print_config(config: EvalExperimentConfig) -> None:
    print("Experiment config:")
    for key in sorted(EvalExperimentConfig.__dataclass_fields__):
        print(f"  {key} = {getattr(config, key)!r}")
