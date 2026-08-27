#!/bin/bash
#SBATCH --job-name=pheq-eval-b0
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

# Frozen-pretrained-VAE (b0) reference row for the paper table.
# Usage: sbatch scripts/slurm/eval_b0.sbatch <data_dir>
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
DATA_DIR="${1:?usage: sbatch eval_b0.sbatch <data_dir>}"
export HF_HOME="${HF_HOME:-$HOME/scratch/photometric/hf_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/scratch/photometric/uv_cache}"
export TORCH_HOME="${TORCH_HOME:-$HOME/scratch/photometric/torch_cache}"
mkdir -p outputs/runs/b0

uv sync --frozen 2>/dev/null || uv sync
.venv/bin/python -m pheq.eval_battery \
    --frozen-vae sd --wfit outputs/wfit_sd_cluster.pt --images "$DATA_DIR" \
    --out outputs/runs/b0/eval.json --device cuda --rfid-n 1024 --refit-w
echo "done -> outputs/runs/b0/eval.json"
