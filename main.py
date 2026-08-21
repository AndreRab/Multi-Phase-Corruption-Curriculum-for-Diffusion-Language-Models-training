import argparse
import json
from pathlib import Path
from experiment import (
    load_eval_config_values,
    parse_eval_set_override,
    print_eval_config,
    EvalExperimentConfig,
    run_eval_experiment,
    load_train_multi_config_values,
    parse_train_multi_set_override,
    print_train_multi_config,
    run_multi_experiment,
    TrainMultiExperimentConfig,
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

    elif mode == "train" or mode == "train_multi":
        config_values = load_train_multi_config_values(args.config)
        for raw_override in args.set_overrides:
            key, value = parse_train_multi_set_override(raw_override)
            config_values[key] = value

        config = TrainMultiExperimentConfig(**config_values)
        print_train_multi_config(config)

        if args.dry_run:
            print("Dry run only. Training was not executed.")
            return

        run_multi_experiment(config)
    else:
        raise ValueError("mode must be either 'train', 'train_multi' or 'eval'.")


if __name__ == "__main__":
    main()
