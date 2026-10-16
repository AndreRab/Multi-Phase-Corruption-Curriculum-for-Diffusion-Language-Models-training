#!/bin/bash

#SBATCH --job-name=m_30r_30m_40s
#SBATCH --account=plgdiffllms-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

# Hugging Face cache -> tscratch
export HF_HOME="/net/tscratch/people/$USER/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

mkdir -p "$HF_HUB_CACHE"
mkdir -p "$HF_DATASETS_CACHE"

echo "HF_HOME=$HF_HOME"

nvidia-smi

torchrun \
  --standalone \
  --nproc_per_node=4 \
  main.py \
  --config experiments_config/train/one_step/mix_30r_30m_40s.json