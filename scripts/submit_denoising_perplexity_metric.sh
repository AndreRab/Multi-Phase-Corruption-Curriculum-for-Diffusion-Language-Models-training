#!/bin/bash
# Usage: bash scripts/submit_denoising_perplexity_metric.sh [sbatch options]
set -euo pipefail
export EVAL_SUITE=new_denoising_perplexity
export EVAL_JOB_PREFIX=eval_new_denoising_perplexity

for steps in 1 2 4 8 16 32 64 100; do
    case "$steps" in
        1|2) time_limit=00:30:00 ;;
        4) time_limit=00:45:00 ;;
        8) time_limit=01:00:00 ;;
        16) time_limit=01:30:00 ;;
        32) time_limit=02:30:00 ;;
        64|100) time_limit=04:00:00 ;;
    esac

    echo "Denoising steps: $steps, default time limit: $time_limit"
    EVAL_CONFIG_PATTERN="*/denoising_$steps/*.json" \
        bash "$(dirname "$0")/submit_new_eval.sh" --time="$time_limit" "$@"
done
