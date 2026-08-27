#!/bin/bash
#SBATCH --job-name=pheq-finetune
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

# Fine-tune one v2 condition (SPEC2 "scripts/slurm/"; docs/plan-3month.md M2).
# Usage:  sbatch scripts/slurm/finetune_ae.sbatch <condition> <data_dir> [steps] [spectral_stats]
# (c1proxy REQUIRES spectral stats; the 4th arg — or $SPECTRAL_STATS — defaults
# to outputs/spectral_stats.json for that condition.)
# train_ae checkpoints on SIGTERM and exits 0 (preemption save); on requeue the
# run resumes from ckpt_latest.pt. eval_battery runs ONLY when training reached
# its full step budget (train_ae writes $OUT_DIR/DONE then — a zero exit alone
# also covers the preemption/max-hours early saves), with the gap-closed refs
# attached automatically when b1's eval.json exists AND parses as JSON.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

CONDITION="${1:?usage: sbatch finetune_ae.sbatch <condition> <data_dir> [steps] [spectral_stats]}"
DATA_DIR="${2:?usage: sbatch finetune_ae.sbatch <condition> <data_dir> [steps] [spectral_stats]}"
STEPS="${3:-}"
SPECTRAL_STATS="${4:-${SPECTRAL_STATS:-}}"

# Fail fast with a diagnosis if the data dir is empty AS SEEN FROM THIS NODE —
# symlinked prefixes like ~/scratch can resolve differently on compute nodes
# than on login nodes (observed on Oscar: /users/<u>/scratch empty on some GPU
# nodes while the canonical /oscar/scratch/<u> path is fine). Always submit
# with the canonical path (readlink -f on the login node).
N_ENTRIES=$(ls "$DATA_DIR" 2>/dev/null | wc -l)
echo "data dir: $DATA_DIR -> $(readlink -f "$DATA_DIR" 2>/dev/null || echo '<unresolvable>') ($N_ENTRIES entries from $(hostname))"
if [ "$N_ENTRIES" -eq 0 ]; then
    echo "ERROR: data dir is empty from this compute node; resubmit with the canonical path (login node: readlink -f <dir>)" >&2
    exit 1
fi
# Concurrent-run safety: array tasks get their own run dir (two trainers
# sharing one dir would interleave monitor.jsonl and clobber ckpt_latest.pt).
OUT_DIR="outputs/runs/$CONDITION${SEED:+_s$SEED}${SLURM_ARRAY_TASK_ID:+_a$SLURM_ARRAY_TASK_ID}"
mkdir -p outputs "$OUT_DIR"

# keep model/wheel caches in scratch, not the home quota (docs/cluster.md)
export HF_HOME="${HF_HOME:-$HOME/scratch/photometric/hf_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/scratch/photometric/uv_cache}"
export TORCH_HOME="${TORCH_HOME:-$HOME/scratch/photometric/torch_cache}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # reduce fragmentation OOMs

# uv is assumed on PATH (curl -LsSf https://astral.sh/uv/install.sh | sh); uv sync is idempotent.
uv sync --frozen 2>/dev/null || uv sync

# c1proxy cannot even build its config without spectral stats (get_config
# raises); default the path and fail fast here rather than at step 0.
if [[ "$CONDITION" == "c1proxy" && -z "$SPECTRAL_STATS" ]]; then
    SPECTRAL_STATS="outputs/spectral_stats.json"
fi
if [[ "$CONDITION" == "c1proxy" && ! -f "$SPECTRAL_STATS" ]]; then
    echo "c1proxy requires spectral stats; not found: $SPECTRAL_STATS" >&2
    exit 2
fi

# Invoke the venv python DIRECTLY (not `uv run`): the preemption chain relies
# on the trap's kill -TERM reaching the python trainer, and older uv releases
# did not forward termination signals to the grandchild.
TRAIN_CMD=(.venv/bin/python -m pheq.train_ae
    --condition "$CONDITION" --data "$DATA_DIR" --out "$OUT_DIR"
    --vae sd --wfit outputs/wfit_sd_cluster.pt --device cuda --max-hours 167)
if [[ -n "$STEPS" ]]; then TRAIN_CMD+=(--steps "$STEPS"); fi
# Seed repeats: SEED=1 sbatch ... gives a separate run dir and --seed.
if [[ -n "${SEED:-}" ]]; then TRAIN_CMD+=(--seed "$SEED"); fi
if [[ -n "$SPECTRAL_STATS" ]]; then TRAIN_CMD+=(--spectral-stats "$SPECTRAL_STATS"); fi
echo "${TRAIN_CMD[@]}"

# --signal=B:TERM@300 delivers SIGTERM to THIS shell (B:), not the python
# child — run the trainer in the background and forward the signal so its
# handler can checkpoint and exit 0 (SPEC2 preemption-save pattern).
"${TRAIN_CMD[@]}" &
CHILD=$!
trap 'echo "SIGTERM: forwarding to trainer (pid '"$CHILD"')"; kill -TERM "$CHILD" 2>/dev/null || true' TERM
set +e
wait "$CHILD"; RC=$?
if [[ $RC -gt 128 ]]; then wait "$CHILD"; RC=$?; fi  # first wait interrupted by the trap
set -e
trap - TERM
if [[ $RC -ne 0 ]]; then
    echo "train_ae exited with $RC" >&2
    exit "$RC"
fi

# train_ae exits 0 on the SIGTERM preemption save AND on the max-hours early
# return; only a $OUT_DIR/DONE sentinel means the full step budget was
# trained. Skip eval otherwise — the requeued job will finish training and
# run it (with <=300 s left, eval_battery would be SIGKILLed mid-write).
if [[ ! -f "$OUT_DIR/DONE" ]]; then
    echo "training not complete (preemption/max-hours save) — skipping eval; will resume on requeue"
    exit 0
fi

echo "=== eval battery on $OUT_DIR/ckpt_latest.pt ==="
EVAL_CMD=(.venv/bin/python -m pheq.eval_battery
    --ckpt "$OUT_DIR/ckpt_latest.pt" --images "$DATA_DIR"
    --out "$OUT_DIR/eval.json" --device cuda --rfid-n 1024 --refit-w)
B1_JSON=outputs/runs/b1/eval.json
ORACLE_CSV=outputs/oracle_probe_sd_cluster.csv
# Gate the gap-closed refs on the b1 JSON actually PARSING, not mere
# existence: a truncated/partial reference would silently poison every
# downstream gap-closed number.
if [[ "$CONDITION" != "b1" && -f "$B1_JSON" && -f "$ORACLE_CSV" ]] \
   && .venv/bin/python -c "import json,sys; json.load(open(sys.argv[1]))" "$B1_JSON" 2>/dev/null; then
    EVAL_CMD+=(--b1-json "$B1_JSON" --oracle-csv "$ORACLE_CSV")
fi
echo "${EVAL_CMD[@]}"
"${EVAL_CMD[@]}"

echo "done. Results in $OUT_DIR"
