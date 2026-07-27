import argparse
from pathlib import Path
from experiment import (
    ExperimentConfig,
    load_config_values,
    parse_set_override,
    print_config,
    run_experiment
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
    config_values = load_config_values(args.config)

    for raw_override in args.set_overrides:
        key, value = parse_set_override(raw_override)
        config_values[key] = value

    config = ExperimentConfig(**config_values)
    print_config(config)

    if args.dry_run:
        print("Dry run only. Training was not executed.")
        return

    run_experiment(config)


if __name__ == "__main__":
    main()
