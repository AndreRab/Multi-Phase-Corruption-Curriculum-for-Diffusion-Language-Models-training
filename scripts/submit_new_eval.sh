#!/bin/bash
# Usage: bash scripts/submit_new_eval.sh [sbatch options]
set -euo pipefail
cd "$(dirname "$0")/.."
project_dir="$PWD"

mkdir -p logs
# Pending array jobs must keep the config list from their own submission.
EVAL_CONFIG_LIST="$(mktemp "$project_dir/logs/eval_configs.XXXXXX")"
export EVAL_CONFIG_LIST
find "$project_dir/experiments_config/eval/new" -type f -name '*.json' | LC_ALL=C sort > "$EVAL_CONFIG_LIST"
count=$(wc -l < "$EVAL_CONFIG_LIST")
if (( count == 0 )); then
    echo "No evaluation configs found." >&2
    rm "$EVAL_CONFIG_LIST"
    exit 1
fi

echo "Submitting $count configs; Slurm determines how many run simultaneously."
echo "Config list: $EVAL_CONFIG_LIST"
sbatch --chdir="$project_dir" --export=ALL \
    --array="0-$((count - 1))" "$@" "$project_dir/scripts/eval_new.slurm"
