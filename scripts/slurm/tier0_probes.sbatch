#!/bin/bash
#SBATCH --job-name=pheq-tier0
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
# (--output stays in the submit dir: SLURM opens it at launch, before this
#  script's mkdir -p outputs could ever run — outputs/ may not exist yet.)

# Tier-0 probes at thesis scale (plan §5.1). One GPU, ~30-60 min total.
# Usage:  sbatch scripts/slurm/tier0_probes.sbatch [IMAGE_DIR]
# IMAGE_DIR should hold >=1K real images (e.g. OpenImages val subset);
# defaults to data/samples for a smoke run.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

IMAGE_DIR="${1:-data/samples}"
DEVICE=cuda
mkdir -p outputs

# keep model/wheel caches in scratch, not the home quota (docs/cluster.md)
export HF_HOME="${HF_HOME:-$HOME/scratch/photometric/hf_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/scratch/photometric/uv_cache}"
export TORCH_HOME="${TORCH_HOME:-$HOME/scratch/photometric/torch_cache}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # reduce fragmentation OOMs

# uv is assumed on PATH (curl -LsSf https://astral.sh/uv/install.sh | sh); uv sync is idempotent.
uv sync --frozen 2>/dev/null || uv sync

echo "=== fit_w (thesis-grade: all images in $IMAGE_DIR) ==="
uv run python -m pheq.probes.fit_w --vae sd --images "$IMAGE_DIR" \
    --n-images 1024 --device $DEVICE --out outputs/wfit_sd_cluster.pt

echo "=== analytic probe ==="
uv run python -m pheq.probes.analytic_probe --vae sd --images "$IMAGE_DIR" \
    --n-images 64 --device $DEVICE --wfit outputs/wfit_sd_cluster.pt \
    --out outputs/analytic_probe_sd_cluster.csv

echo "=== oracle probe (RQ0 / gate G0) ==="
uv run python -m pheq.probes.oracle_probe --vae sd --images "$IMAGE_DIR" \
    --n-images 16 --steps 500 --device $DEVICE \
    --out outputs/oracle_probe_sd_cluster.csv

echo "done. Results in outputs/*_cluster.*"
