"""Shared plumbing for the Tier-0 CLI probes (plan §5.1).

Internal to ``pheq.probes`` — not part of the frozen SPEC surface. Provides
the common argument group, image sourcing (directory or deterministic
synthetic images so ``--vae toy`` runs fully offline), autoencoder
construction for ``--vae sd|toy``, and encode/decode adapters that work with
both the SD-VAE wrapper (``encode_moments`` / ``decode_latents``, SPEC
pheq/vae.py) and the toy autoencoders (``encode`` / ``decode``).
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

#: Photometric factor names in the plan's fixed order (plan §3.1).
FACTORS = ("brightness", "contrast", "saturation", "hue")

#: Per-factor magnitude grids for the analytic probe, from the plan's sampling
#: ranges (plan §3.1): beta, gamma in [0.6, 1.4], sat in [0.05, 1.5] with
#: s = 0 tested at eval, hue in [-pi/4, pi/4].
DEFAULT_GRIDS: dict[str, tuple[float, ...]] = {
    "brightness": (0.6, 0.8, 1.2, 1.4),
    "contrast": (0.6, 0.8, 1.2, 1.4),
    "saturation": (0.0, 0.05, 0.5, 1.5),
    "hue": (-math.pi / 4, -math.pi / 8, math.pi / 8, math.pi / 4),
}

#: Smaller grid for the (optimization-heavy) oracle probe: the range endpoints
#: per factor (plan §5.1(a): factor x magnitude on N images).
ORACLE_GRIDS: dict[str, tuple[float, ...]] = {
    "brightness": (0.6, 1.4),
    "contrast": (0.6, 1.4),
    "saturation": (0.05, 1.5),
    "hue": (-math.pi / 4, math.pi / 4),
}


def add_common_args(parser: argparse.ArgumentParser, n_images: int = 16) -> None:
    """Register the argument group shared by every Tier-0 probe."""
    parser.add_argument(
        "--images", type=str, default=None,
        help="directory of images; if omitted, deterministic synthetic images "
             "are generated (offline smoke path)")
    parser.add_argument(
        "--vae", choices=("sd", "toy"), default="sd",
        help="'sd' = stabilityai/sd-vae-ft-mse via pheq.vae.load_sd_vae "
             "(lazy diffusers import); 'toy' = pheq.vae.ToyLinearAE, fully offline")
    parser.add_argument("--device", type=str, default="cpu", help="cpu | cuda | mps")
    parser.add_argument("--n-images", type=int, default=n_images,
                        help=f"number of images to use (default {n_images})")
    parser.add_argument(
        "--image-size", type=int, default=None,
        help="square image resolution (default: 256 for --vae sd, 64 for --vae toy)")
    parser.add_argument("--seed", type=int, default=0)


def resolve_image_size(args: argparse.Namespace) -> int:
    """Default resolution: 256 for the SD-VAE (f=8), 64 for the toy AE (f=2)."""
    if args.image_size is not None:
        return int(args.image_size)
    return 256 if args.vae == "sd" else 64


def synthetic_images(n: int, size: int, seed: int = 0) -> torch.Tensor:
    """Deterministic smooth random RGB images in [0, 1], shape (n, 3, size, size).

    Low-frequency color fields plus mild texture noise — enough chromatic and
    spatial diversity for the W-fit design matrix to be well conditioned, so
    every probe can run fully offline (plan §5.1 Tier-0 smoke path).
    """
    gen = torch.Generator().manual_seed(seed)
    low = torch.rand((n, 3, max(size // 8, 2), max(size // 8, 2)), generator=gen)
    imgs = F.interpolate(low, size=(size, size), mode="bilinear", align_corners=False)
    imgs = imgs + 0.05 * torch.randn((n, 3, size, size), generator=gen)
    return imgs.clamp(0.0, 1.0)


def load_image_dir(directory: str, n: int, size: int) -> torch.Tensor:
    """Load up to ``n`` images from a directory as a float32 (n, 3, size, size) batch in [0, 1]."""
    import numpy as np
    from PIL import Image

    paths = sorted(
        p for p in Path(directory).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )[:n]
    if not paths:
        raise FileNotFoundError(f"no images with extensions {IMAGE_EXTENSIONS} in {directory!r}")
    tensors = []
    for path in paths:
        with Image.open(path) as im:
            im = im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(tensors)


def get_images(args: argparse.Namespace, size: int) -> torch.Tensor:
    """Image batch per CLI args: from ``--images DIR`` if given, else synthetic."""
    if args.images is not None:
        return load_image_dir(args.images, args.n_images, size)
    return synthetic_images(args.n_images, size, seed=args.seed)


def build_vae(name: str, device: torch.device) -> torch.nn.Module:
    """Construct the requested autoencoder in eval mode on ``device``.

    ``sd`` paths go through :func:`pheq.vae.load_sd_vae`, which imports
    diffusers lazily (SPEC pheq/vae.py) — so ``--vae toy`` never touches it.
    """
    if name == "sd":
        from pheq.vae import load_sd_vae

        return load_sd_vae(device=str(device))
    from pheq.vae import ToyLinearAE

    model = ToyLinearAE().to(device)
    model.eval()
    return model


def freeze(model: torch.nn.Module) -> None:
    """Disable gradients on all parameters (probes evaluate frozen checkpoints, plan §5.1)."""
    for p in model.parameters():
        p.requires_grad_(False)


@torch.no_grad()
def encode_latents(model: torch.nn.Module, images: torch.Tensor, chunk: int = 8) -> torch.Tensor:
    """Encode an image batch to latents (posterior mean for moment encoders)."""
    outs = []
    for i in range(0, images.shape[0], chunk):
        x = images[i : i + chunk]
        if hasattr(model, "encode_moments"):
            z = model.encode_moments(x)[0]
        elif hasattr(model, "encode"):
            z = model.encode(x)
            if isinstance(z, (tuple, list)):
                z = z[0]
        else:
            z = model.encoder(x)
        outs.append(z)
    return torch.cat(outs)


def make_decode_fn(model: torch.nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a ``z -> image`` callable for either AE interface."""
    if hasattr(model, "decode_latents"):
        return model.decode_latents
    if hasattr(model, "decode"):

        def _decode(z: torch.Tensor) -> torch.Tensor:
            out = model.decode(z)
            return out[0] if isinstance(out, (tuple, list)) else out

        return _decode
    return model.decoder


def factor_affine(name: str, magnitude: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(A, b) in Aff(3) for one named photometric factor at ``magnitude`` (plan §3.1)."""
    from pheq import color

    if name == "brightness":
        return color.brightness_affine(magnitude)
    if name == "contrast":
        return color.contrast_affine(magnitude)
    if name == "saturation":
        return color.saturation_affine(magnitude)
    if name == "hue":
        return color.hue_affine(magnitude)
    raise ValueError(f"unknown photometric factor {name!r}")


def write_csv(path: str, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    """Write a probe result table to ``path`` as CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(header))
        writer.writerows(rows)
