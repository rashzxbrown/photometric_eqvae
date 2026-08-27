"""Generate the c1proxy spectral-match target from a trained checkpoint's latents.

The C1-proxy control (docs/plan-3month.md M2) fine-tunes with a latent
spectral-smoothing regularizer matched to the BEST PHOTOMETRIC condition's
latent spectrum, with no equivariance — the intervention that separates
"equivariance helps" from "spectral smoothing helps". This script measures
that target spectrum.

Usage (cluster, GPU or CPU node):
    .venv/bin/python scripts/data/make_spectral_stats.py \
        --ckpt outputs/runs/p1_analytic/ckpt_latest.pt \
        --images /oscar/scratch/<user>/photometric/openimages/validation \
        --out outputs/spectral_stats.json [--n-images 256] [--device cuda]
"""

from __future__ import annotations

import argparse

import torch

from pheq.data import ImageFolderDataset
from pheq.eval_battery import load_run_checkpoint
from pheq.fid import _resolve_vae
from pheq.spectral import radial_power_spectrum, save_spectrum_stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, help="run checkpoint (.pt) providing the encoder")
    p.add_argument("--images", required=True, help="image directory")
    p.add_argument("--out", default="outputs/spectral_stats.json")
    p.add_argument("--n-images", type=int, default=256)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = load_run_checkpoint(args.ckpt)
    vae = _resolve_vae(ckpt, str(device))
    size = int(ckpt["config"].get("image_size", 256))
    ds = ImageFolderDataset(args.images, size=size)
    n = min(args.n_images, len(ds))

    mus = []
    with torch.no_grad():
        for i in range(0, n, args.batch):
            batch = torch.stack([ds[j][0] for j in range(i, min(i + args.batch, n))]).to(device)
            mu, _sigma = vae.encode_moments(batch)
            mus.append(mu.cpu())
    mu_all = torch.cat(mus)
    freqs, power = radial_power_spectrum(mu_all)
    save_spectrum_stats(args.out, freqs, power)
    print(
        f"spectral stats from {ckpt['condition']} (step {ckpt['step']}) on {n} images "
        f"-> {args.out} ({len(freqs)} radii)"
    )


if __name__ == "__main__":
    main()
