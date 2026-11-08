#!/bin/bash
#SBATCH --job-name=eval_new
#SBATCH --account=plgdiffllms-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

: "${EVAL_CONFIG_LIST:?Submit using bash scripts/submit_new_eval.sh}"
: "${SLURM_ARRAY_TASK_ID:?Missing array task index}"
mapfile -t configs < "$EVAL_CONFIG_LIST"
config="${configs[$SLURM_ARRAY_TASK_ID]}"
test -f "$config"

phase="$(basename "$config" .json)"
denoising="$(basename "$(dirname "$config")")"
seed="$(basename "$(dirname "$(dirname "$config")")")"
task_name="eval_${seed}_${denoising}_${phase}"
scontrol update JobId="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" JobName="$task_name"

export HF_HOME="/net/tscratch/people/$USER/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

echo "Job=$SLURM_JOB_ID Task=$SLURM_ARRAY_TASK_ID Name=$task_name Config=$config"
nvidia-smi
torchrun --standalone --nproc_per_node=4 main.py --config "$config"
