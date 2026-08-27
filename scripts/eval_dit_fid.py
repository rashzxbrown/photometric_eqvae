"""Post-hoc generation FID for a finished train_dit checkpoint.

Loads the DiT run checkpoint, swaps in the EMA weights, samples ``--n``
latents cfg-free (DDPM, same sampler as training hooks), un-normalizes with
the latent cache's stats.json, decodes with the AE run checkpoint, saves
PNGs, and scores against a cleanfid reference (a custom-stats name from
scripts/data/make_fid_ref.py, or a directory).

Usage:
    .venv/bin/python scripts/eval_dit_fid.py \
        --dit-ckpt outputs/dit/<run>/ckpt_latest.pt \
        --latents /oscar/scratch/<u>/photometric/latents/b1 \
        --ref openimages_val_256 [--n 5000] [--device cuda] \
        --out outputs/dit/<run>/fid_final.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import pheq.fid as fid_mod
from pheq.cache_latents import load_stats
from pheq.diffusion import EMA, GaussianDiffusion
from pheq.fid import _resolve_vae
from pheq.train_dit import _ARCHS


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dit-ckpt", required=True)
    p.add_argument("--latents", required=True, help="latent cache dir (stats.json)")
    p.add_argument("--ref", required=True,
                   help="cleanfid custom-stats name OR a reference image dir")
    p.add_argument("--vae-ckpt", default=None,
                   help="AE run checkpoint; default: the one recorded in the DiT config")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--sample-steps", type=int, default=None,
                   help="default: the run's sample_steps from its config")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, help="JSON output (default: alongside ckpt)")
    args = p.parse_args()

    device = torch.device(args.device)
    ck = torch.load(args.dit_ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    arch = ck["arch"]
    num_classes = int(cfg.get("num_classes", 0))
    c, h, w = cfg["latent_shape"]
    sample_steps = args.sample_steps or int(cfg.get("sample_steps", 250))
    timesteps = int(cfg.get("timesteps", 1000))

    model = _ARCHS[arch](num_classes=num_classes, input_size=h, in_channels=c).to(device)
    model.load_state_dict(ck["model"])
    ema = EMA(model, decay=float(cfg.get("ema_decay", 0.9999)))
    ema.load_state_dict(ck["ema"])
    ema.copy_to(model)  # EMA weights for sampling (standard eval convention)
    model.eval()

    stats = load_stats(args.latents)
    mean = torch.tensor(stats["mean"], dtype=torch.float32).view(-1, 1, 1).to(device)
    std = torch.tensor(stats["std"], dtype=torch.float32).view(-1, 1, 1).to(device)

    vae_ckpt = args.vae_ckpt or cfg.get("vae_ckpt")
    if vae_ckpt is None:
        raise SystemExit("no --vae-ckpt given and none recorded in the DiT config")
    vae = _resolve_vae(vae_ckpt, str(device))

    diffusion = GaussianDiffusion(timesteps=timesteps).to(device)
    gen_dir = Path(args.dit_ckpt).parent / "fid_final" / f"n{args.n}_s{args.seed}"
    gen_dir.mkdir(parents=True, exist_ok=True)

    written = len(list(gen_dir.glob("*.png")))
    if written >= args.n:
        print(f"{written} samples already present — skipping sampling")
    else:
        step_note = ck.get("step", "?")
        print(f"sampling {args.n} (batch {args.batch}, {sample_steps} DDPM steps, "
              f"ckpt step {step_note}) ...", flush=True)
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        with torch.no_grad():
            while written < args.n:
                nb = min(args.batch, args.n - written)
                y = None
                if num_classes > 0:
                    y = torch.arange(nb, device=device) % num_classes
                z = diffusion.ddpm_sample(
                    model, (nb, c, h, w), y=y, device=device,
                    steps=sample_steps, gen=gen,
                )
                z = z * std + mean  # un-normalize (plan §3.4)
                imgs = vae.decode_latents(z).clamp(0.0, 1.0)
                for j in range(nb):
                    arr = (imgs[j].permute(1, 2, 0).cpu().numpy() * 255.0)
                    Image.fromarray(arr.round().astype(np.uint8)).save(
                        gen_dir / f"{written + j:06d}.png"
                    )
                written += nb
                if written % 512 < args.batch:
                    print(f"  {written}/{args.n}", flush=True)

    if Path(args.ref).is_dir():
        score = fid_mod.compute_fid(str(gen_dir), args.ref)
    else:
        score = fid_mod.fid_to_stats(str(gen_dir), args.ref)

    result = {
        "fid": float(score), "n": args.n, "seed": args.seed,
        "sample_steps": sample_steps, "ref": args.ref,
        "dit_ckpt": str(args.dit_ckpt), "ckpt_step": int(ck.get("step", -1)),
        "arch": arch, "stats_condition": stats.get("condition"),
    }
    out_path = Path(args.out) if args.out else Path(args.dit_ckpt).parent / "fid_final.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"FID = {score:.3f}  (n={args.n}, ckpt step {result['ckpt_step']}, "
          f"condition {result['stats_condition']})")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
