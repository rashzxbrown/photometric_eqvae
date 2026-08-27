"""Retroactively score already-sampled FID directories of a train_dit run.

The in-training FID hook writes its samples to ``<run>/fid/step_XXXXXXX/``
BEFORE scoring; when scoring fails (e.g. the cleanfid × Python-3.14
forkserver pickling crash) the PNGs survive. This script scans those dirs,
computes the missing FID scores, and appends ``{"level": "fid", ...,
"retro": true}`` records to the run's monitor.jsonl — steps that already
have a fid record are skipped.

Usage:
    .venv/bin/python scripts/score_fid_dirs.py --run outputs/dit/<run_dir> \
        --ref openimages_val_256 [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pheq.fid as fid_mod


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="train_dit run dir (has fid/ and monitor.jsonl)")
    p.add_argument("--ref", required=True,
                   help="cleanfid custom-stats name OR a reference image dir")
    args = p.parse_args()

    run = Path(args.run)
    monitor = run / "monitor.jsonl"
    scored: set[int] = set()
    if monitor.exists():
        for line in monitor.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("level") == "fid":
                scored.add(int(rec["step"]))

    step_dirs = sorted((run / "fid").glob("step_*")) if (run / "fid").is_dir() else []
    if not step_dirs:
        raise SystemExit(f"no fid/step_* dirs under {run}")

    for d in step_dirs:
        m = re.fullmatch(r"step_(\d+)", d.name)
        if m is None:
            continue
        step = int(m.group(1))
        if step in scored:
            print(f"step {step}: already scored — skip")
            continue
        n = len(list(d.glob("*.png")))
        if n == 0:
            print(f"step {step}: empty dir — skip")
            continue
        if Path(args.ref).is_dir():
            score = fid_mod.compute_fid(str(d), args.ref)
        else:
            score = fid_mod.fid_to_stats(str(d), args.ref)
        with open(monitor, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"level": "fid", "step": step, "fid": float(score),
                                 "n": n, "retro": True}) + "\n")
        print(f"step {step}: FID = {score:.3f}  (n={n})")


if __name__ == "__main__":
    main()
