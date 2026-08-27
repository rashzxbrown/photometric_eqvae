#!/bin/bash
#SBATCH --job-name=pheq-tests
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out

# Full offline test suite on a compute node (keeps login nodes clean).
# Usage: sbatch scripts/slurm/tests.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME="${HF_HOME:-$HOME/scratch/photometric/hf_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/scratch/photometric/uv_cache}"
export TORCH_HOME="${TORCH_HOME:-$HOME/scratch/photometric/torch_cache}"

uv sync --frozen 2>/dev/null || uv sync
.venv/bin/python -m pytest -q -m "not slow"
.venv/bin/python -m pytest -q -m slow
