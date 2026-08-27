"""Build the cleanfid reference statistics for generation FID (one-time).

Processes real images through the EXACT pipeline the models see
(pheq.data.ImageFolderDataset: shorter-side resize + center-crop to 256),
writes them as PNGs, and registers cleanfid custom stats under ``--name`` —
after which every FID evaluation (post-hoc scorer, train_dit --fid-ref-stats)
scores against the same reference without re-featurizing.

Usage (GPU node preferred for the featurization, ~5 min for 10K images):
    .venv/bin/python scripts/data/make_fid_ref.py \
        --images /oscar/scratch/<u>/photometric/openimages/validation \
        --name openimages_val_256 --n 10000 \
        --proc-dir /oscar/scratch/<u>/photometric/fid_ref_256
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from pheq.data import ImageFolderDataset
from pheq.fid import make_custom_stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", required=True, help="raw real-image directory")
    p.add_argument("--name", required=True, help="cleanfid custom-stats name")
    p.add_argument("--n", type=int, default=10000, help="number of reference images")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--proc-dir", required=True,
                   help="where processed reference PNGs are written")
    args = p.parse_args()

    ds = ImageFolderDataset(args.images, size=args.size)
    n = min(args.n, len(ds))
    proc = Path(args.proc_dir)
    proc.mkdir(parents=True, exist_ok=True)

    existing = len(list(proc.glob("*.png")))
    if existing >= n:
        print(f"processed dir already has {existing} PNGs — skipping conversion")
    else:
        for i in range(n):
            out = proc / f"{i:06d}.png"
            if out.exists():
                continue
            img, _ = ds[i]
            arr = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
            Image.fromarray(arr).save(out)
            if (i + 1) % 1000 == 0:
                print(f"processed {i + 1}/{n}", flush=True)

    make_custom_stats(args.name, str(proc))
    print(f"cleanfid custom stats registered: {args.name!r} "
          f"({n} images from {args.images}, size {args.size})")


if __name__ == "__main__":
    main()
