#!/bin/bash
#SBATCH --job-name=pheq-dit
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --open-mode=append         # requeued jobs append to the same log
#SBATCH --signal=B:TERM@300        # SIGTERM to this shell 300 s before the limit
#SBATCH --requeue

# NOTE on --output: SLURM opens the stdout file at job LAUNCH, before any line
# of this script runs — pointing it into outputs/ would fail on a fresh clone
# (the mkdir below can never help its own log), so logs land in the submit dir.

# DiT proxy training on cached latent shards (SPEC2 "scripts/slurm/" /
# train_dit.py; plan §3.4). Same signal/save pattern as finetune_ae.sbatch:
# train_dit checkpoints on SIGTERM and exits 0, and resumes on requeue.
# Usage:  sbatch scripts/slurm/train_dit.sbatch <latent_dir> <vae_ckpt> [arch] [steps] [extra args...]
# Args past the 4th are passed through to pheq.train_dit verbatim, e.g.:
#   sbatch ... dit_s 100000 --fid-every 10000 --fid-ref-stats openimages_val_256 --batch 512

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

LATENT_DIR="${1:?usage: sbatch train_dit.sbatch <latent_dir> <vae_ckpt> [arch] [steps] [extra...]}"
VAE_CKPT="${2:?usage: sbatch train_dit.sbatch <latent_dir> <vae_ckpt> [arch] [steps] [extra...]}"
ARCH="${3:-dit_s}"
STEPS="${4:-}"
shift $(( $# < 4 ? $# : 4 ))
EXTRA=("$@")
# Key the run dir on the latent dir's FULL resolved path, not its basename:
# .../b1/latents and .../b2lite/latents would otherwise collide on 'latents'
# and clobber each other's ckpt_latest.pt. Array tasks get their own suffix.
LATENT_ID="$(cd "$LATENT_DIR" && pwd)"
LATENT_ID="${LATENT_ID#/}"; LATENT_ID="${LATENT_ID//\//_}"
# Seed repeats: SEED=1 sbatch ... gives a separate run dir and --seed.
OUT_DIR="outputs/dit/${LATENT_ID}_$ARCH${SEED:+_s$SEED}${SLURM_ARRAY_TASK_ID:+_a$SLURM_ARRAY_TASK_ID}"
mkdir -p outputs "$OUT_DIR"

# keep model/wheel caches in scratch, not the home quota (docs/cluster.md)
export HF_HOME="${HF_HOME:-$HOME/scratch/photometric/hf_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/scratch/photometric/uv_cache}"
export TORCH_HOME="${TORCH_HOME:-$HOME/scratch/photometric/torch_cache}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # reduce fragmentation OOMs

# uv is assumed on PATH (curl -LsSf https://astral.sh/uv/install.sh | sh); uv sync is idempotent.
uv sync --frozen 2>/dev/null || uv sync

# Invoke the venv python DIRECTLY (not `uv run`): the preemption chain relies
# on the trap's kill -TERM reaching the python trainer, and older uv releases
# did not forward termination signals to the grandchild.
CMD=(.venv/bin/python -m pheq.train_dit
    --latents "$LATENT_DIR" --out "$OUT_DIR" --arch "$ARCH"
    --vae-ckpt "$VAE_CKPT" --device cuda --max-hours 167)
if [[ -n "$STEPS" ]]; then CMD+=(--steps "$STEPS"); fi
if [[ -n "${SEED:-}" ]]; then CMD+=(--seed "$SEED"); fi
if [[ ${#EXTRA[@]} -gt 0 ]]; then CMD+=("${EXTRA[@]}"); fi
echo "${CMD[@]}"

# --signal=B:TERM@300 delivers SIGTERM to THIS shell (B:), not the python
# child — run the trainer in the background and forward the signal so its
# handler can checkpoint and exit 0 (SPEC2 preemption-save pattern).
"${CMD[@]}" &
CHILD=$!
trap 'echo "SIGTERM: forwarding to trainer (pid '"$CHILD"')"; kill -TERM "$CHILD" 2>/dev/null || true' TERM
set +e
wait "$CHILD"; RC=$?
if [[ $RC -gt 128 ]]; then wait "$CHILD"; RC=$?; fi  # first wait interrupted by the trap
set -e
trap - TERM
if [[ $RC -ne 0 ]]; then
    echo "train_dit exited with $RC" >&2
    exit "$RC"
fi

echo "done. Results in $OUT_DIR"
