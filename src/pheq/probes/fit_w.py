"""Tier-0 probe (b): fit the linear latent->RGB map ``x_p ~ W z_p + c`` (plan §3.2, §5.1(b)).

Encodes a batch of images with the chosen autoencoder, fits (W, c) by least
squares via :func:`pheq.analytic.fit_w` (the "latent RGB preview" map re-fit
per autoencoder, plan §3.2), prints R² per RGB channel, and saves the fit to
``wfit.pt`` for the downstream analytic probe.

Usage::

    uv run python -m pheq.probes.fit_w --images DIR [--vae sd|toy] [--device mps]
    uv run python -m pheq.probes.fit_w --vae toy        # fully offline smoke run

The least-squares solve always runs on CPU (torch.linalg.lstsq is not
available on every accelerator backend); only encoding uses ``--device``.
"""

from __future__ import annotations

import argparse

import torch

from pheq.analytic import fit_w
from pheq.probes._common import (
    add_common_args,
    build_vae,
    encode_latents,
    get_images,
    resolve_image_size,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fit the linear latent->RGB map (W, c) of an autoencoder (plan §3.2)."
    )
    add_common_args(parser, n_images=64)
    parser.add_argument("--out", type=str, default="wfit.pt",
                        help="output path for the saved fit (default wfit.pt)")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    size = resolve_image_size(args)

    images = get_images(args, size).to(device)
    model = build_vae(args.vae, device)
    latents = encode_latents(model, images)

    fit = fit_w(latents.cpu().float(), images.cpu().float())

    print(
        f"fit_w: vae={args.vae} n={images.shape[0]} image={size}x{size} "
        f"latent={tuple(latents.shape[1:])}"
    )
    for name, r2c in zip("RGB", fit.r2_per_channel.tolist()):
        print(f"  R2[{name}] = {r2c:.6f}")
    print(f"  R2 (mean) = {float(fit.r2):.6f}")

    payload = {
        "W": fit.W.detach().cpu(),
        "c": fit.c.detach().cpu(),
        "r2": float(fit.r2),
        "r2_per_channel": fit.r2_per_channel.detach().cpu(),
        "vae": args.vae,
        "n_images": int(images.shape[0]),
        "image_size": size,
    }
    torch.save(payload, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
