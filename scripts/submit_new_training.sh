#!/bin/bash
# Usage: bash submit_new_training.sh
set -euo pipefail
cd "$(dirname "$0")"
project_dir="$PWD"

mkdir -p logs
# Keep a separate, immutable list for every submission: pending jobs use it later.
TRAIN_CONFIG_LIST="$(mktemp "$project_dir/logs/train_configs.XXXXXX")"
export TRAIN_CONFIG_LIST
find "$project_dir/experiments_config/train/new" -type f -name '*.json' | LC_ALL=C sort > "$TRAIN_CONFIG_LIST"
count=$(wc -l < "$TRAIN_CONFIG_LIST")
if (( count == 0 )); then
    echo "No training configs found." >&2
    rm "$TRAIN_CONFIG_LIST"
    exit 1
fi

echo "Submitting $count configs; Slurm determines how many run simultaneously."
echo "Config list: $TRAIN_CONFIG_LIST"
sbatch --chdir="$project_dir" --export=ALL \
    --array="0-$((count - 1))" "$project_dir/train_new.slurm"
