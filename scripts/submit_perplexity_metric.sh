#!/bin/bash
# Usage: bash scripts/submit_perplexity_metric.sh [sbatch options]
set -euo pipefail
export EVAL_SUITE=new_perplexity
export EVAL_JOB_PREFIX=eval_new_perplexity
exec bash "$(dirname "$0")/submit_new_eval.sh" "$@"
