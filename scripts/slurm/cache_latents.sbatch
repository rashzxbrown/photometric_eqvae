#!/bin/bash
#SBATCH --job-name=pheq-cache
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
# (--output stays in the submit dir: SLURM opens it at launch, before this
#  script's mkdir -p outputs could ever run — outputs/ may not exist yet.)

# Cache latent shards + normalization stats for one run checkpoint
# (SPEC2 "scripts/slurm/" / cache_latents.py; plan §3.4 caching convention).
# Usage:  sbatch scripts/slurm/cache_latents.sbatch <ckpt> <data_dir> <out_dir>
# Writes shard_{i:04d}.pt files and the MANDATORY stats.json to <out_dir>
# (train_dit refuses to start without it).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

CKPT="${1:?usage: sbatch cache_latents.sbatch <ckpt> <data_dir> <out_dir>}"
DATA_DIR="${2:?usage: sbatch cache_latents.sbatch <ckpt> <data_dir> <out_dir>}"
OUT_DIR="${3:?usage: sbatch cache_latents.sbatch <ckpt> <data_dir> <out_dir>}"
mkdir -p outputs "$OUT_DIR"

# keep model/wheel caches in scratch, not the home quota (docs/cluster.md)
export HF_HOME="${HF_HOME:-$HOME/scratch/photometric/hf_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/scratch/photometric/uv_cache}"
export TORCH_HOME="${TORCH_HOME:-$HOME/scratch/photometric/torch_cache}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # reduce fragmentation OOMs

# uv is assumed on PATH (curl -LsSf https://astral.sh/uv/install.sh | sh); uv sync is idempotent.
uv sync --frozen 2>/dev/null || uv sync

CMD=(uv run python -m pheq.cache_latents
    --ckpt "$CKPT" --images "$DATA_DIR" --out "$OUT_DIR" --device cuda)
echo "${CMD[@]}"
"${CMD[@]}"

echo "done. Shards + stats.json in $OUT_DIR"
