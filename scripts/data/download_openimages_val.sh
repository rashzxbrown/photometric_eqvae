#!/bin/bash
# Download the OpenImages validation split (~41K images, ~12 GB) from the public
# S3 bucket (no account/registration needed) and build a deterministic N-image
# subset for Tier-0 probes (docs/cluster.md).
#
# Run on a cluster LOGIN/TRANSFER node (compute nodes often lack internet):
#   bash scripts/data/download_openimages_val.sh "$SCRATCH/data/openimages" 1024
#
# Then: sbatch scripts/slurm/tier0_probes.sbatch "$SCRATCH/data/openimages/val_1024"
set -euo pipefail

DEST="${1:?usage: download_openimages_val.sh DEST_DIR [N=1024]}"
N="${2:-1024}"
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

# aws cli: use system one if present, else zero-install via uvx
if command -v aws >/dev/null 2>&1; then AWS=aws; else AWS="uvx --from awscli aws"; fi

mkdir -p "$DEST/validation"
echo "syncing s3://open-images-dataset/validation -> $DEST/validation (~12 GB, resumable)"
$AWS s3 --no-sign-request sync s3://open-images-dataset/validation "$DEST/validation"

# CRITICAL on purged scratch filesystems: aws s3 sync preserves each S3 object's
# LastModified (2018 for OpenImages) as the local mtime, so scratch purgers that
# key on file age see freshly-downloaded files as years-stale and DELETE them on
# the next sweep (observed on Oscar: the full set vanished overnight, directories
# and fresh-mtime caches untouched). Reset mtimes to now. Side effect: a future
# re-sync sees local-newer-than-S3 and correctly skips re-downloading.
echo "resetting mtimes (purge protection): sample before -> $(stat -c %y "$(ls -d "$DEST"/validation/* | head -1)" 2>/dev/null | cut -d' ' -f1)"
find "$DEST/validation" -type f -exec touch {} +
echo "sample after -> $(stat -c %y "$(ls -d "$DEST"/validation/* | head -1)" 2>/dev/null | cut -d' ' -f1)"

TOTAL=$(ls "$DEST/validation" | wc -l)
echo "validation/: $TOTAL files"
if [ "$TOTAL" -lt "$N" ]; then
    echo "ERROR: sync produced only $TOTAL files (< $N requested)" >&2
    exit 1
fi

# Deterministic subset: lexicographically first N image files, COPIED into val_N/
# (real copies, not symlinks: ~300 MB buys a probe set that survives a purge of
# the parent tree and is safe to relocate; the manifest pins reproducibility).
# NOTE: awk (not head) — head's early pipe-close SIGPIPEs ls, which set -o pipefail
# turns into a silent mid-script abort. awk consumes the full stream.
SUBSET="$DEST/val_${N}"
mkdir -p "$SUBSET"
MANIFEST="$REPO_DIR/data/manifests/openimages_val_${N}.txt"
mkdir -p "$(dirname "$MANIFEST")"
ls "$DEST/validation" | sort | awk -v n="$N" 'NR<=n' > "$MANIFEST"
while read -r f; do cp -f "$DEST/validation/$f" "$SUBSET/$f"; done < "$MANIFEST"

COPIED=$(ls "$SUBSET" | wc -l)
if [ "$COPIED" -ne "$N" ]; then
    echo "ERROR: subset has $COPIED files, expected $N" >&2
    exit 1
fi
echo "subset:   $SUBSET ($COPIED images, real copies)"
echo "manifest: $MANIFEST  <- commit this (reproducibility, plan §5.1 / docs/cluster.md)"
