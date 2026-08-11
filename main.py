import argparse
import json
from pathlib import Path
from experiment import (
    EvalExperimentConfig,
    ExperimentConfig,
    load_config_values as load_train_config_values,
    parse_set_override as parse_train_set_override,
    print_config as print_train_config,
    run_experiment,
    run_eval_experiment,
)
from experiment.eval_experiment_config import (
    load_config_values as load_eval_config_values,
    parse_set_override as parse_eval_set_override,
    print_config as print_eval_config,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a diffusion experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a JSON experiment config.",
    )
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        help="Override one config value, e.g. --set model_id=v2.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved config without running training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = "train"
    if args.config is not None:
        with args.config.open() as config_file:
            raw_config = json.load(config_file)
        if not isinstance(raw_config, dict):
            raise ValueError("Config JSON must be an object.")
        if "overrides" in raw_config:
            raw_config = raw_config["overrides"]
        if not isinstance(raw_config, dict):
            raise ValueError("'overrides' must be an object.")
        mode = raw_config.get("mode", "train")

    if mode == "eval":
        config_values = load_eval_config_values(args.config)
        for raw_override in args.set_overrides:
            key, value = parse_eval_set_override(raw_override)
            config_values[key] = value

        config = EvalExperimentConfig(**config_values)
        print_eval_config(config)
        
        if args.dry_run:
            print("Dry run only. Evaluation was not executed.")
            return
        run_eval_experiment(config)

    elif mode == "train":
        config_values = load_train_config_values(args.config)
        for raw_override in args.set_overrides:
            key, value = parse_train_set_override(raw_override)
            config_values[key] = value

        config = ExperimentConfig(**config_values)
        print_train_config(config)

        if args.dry_run:
            print("Dry run only. Training was not executed.")
            return

        run_experiment(config)
        
    else:
        raise ValueError("mode must be either 'train' or 'eval'.")


if __name__ == "__main__":
    main()
