"""Tier-0 probe (c): decoder-side EE of the closed-form analytic operator (plan §3.2, §5.1(c)).

For each photometric factor x magnitude, builds the pixel-space affine
(A, b) in Aff(3) (plan §3.1), derives the closed-form latent operator

    M = W^+ A W + (I_C - W^+ W) K,   m = W^+ (A c + b - c)      (plan §3.2)

from a saved W-fit (``wfit.pt``), and reports the decoder-side equivariance
error d(D(M z + m), tau_a(x)) for d in {L2, CIEDE2000} against the
identity-operator floor d(D(z), tau_a(x)) (plan §5.3). Equivariance targets
are computed pre-clip (``apply_affine(..., clip=False)``, plan §3.1) and the
clipped-pixel fraction is measured per magnitude, not asserted. A final row
averages the same quantities over randomly sampled composed ``PhotoParams``.

Usage::

    uv run python -m pheq.probes.analytic_probe --wfit wfit.pt [--vae sd|toy] [--images DIR]
    uv run python -m pheq.probes.analytic_probe --vae toy   # offline smoke run
                                                            # (fits W on the fly if wfit.pt is absent)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable

import torch

from pheq.analytic import WFit, analytic_operator, fit_w
from pheq.color import PhotoParams, apply_affine, clipped_fraction
from pheq.metrics import ee_pix
from pheq.probes._common import (
    DEFAULT_GRIDS,
    add_common_args,
    build_vae,
    encode_latents,
    factor_affine,
    freeze,
    get_images,
    make_decode_fn,
    resolve_image_size,
    write_csv,
)

CSV_HEADER = ("factor", "magnitude", "ee_l2", "ee_ciede2000",
              "floor_l2", "floor_ciede2000", "clip_frac")


def _load_or_fit(path: str, latents: torch.Tensor, images: torch.Tensor) -> WFit:
    """Load a saved W-fit, or fit one on the fly (offline smoke path) if absent."""
    p = Path(path)
    if p.exists():
        d = torch.load(p, map_location="cpu", weights_only=True)
        return WFit(W=d["W"], c=d["c"], r2=float(d["r2"]),
                    r2_per_channel=d["r2_per_channel"])
    return fit_w(latents.cpu().float(), images.cpu().float())


def _evaluate(
    z: torch.Tensor,
    images: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
    fit: WFit,
    k: str,
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[float, float, float, float, float]:
    """(ee_l2, ee_de00, floor_l2, floor_de00, clip_frac) for one pixel affine (A, b).

    The floor is the identity latent operator (M = I, m = 0): the error of
    doing nothing to the latent, plan §5.1(c) "against the identity-operator
    floor".
    """
    device = z.device
    A_dev, b_dev = A.to(device), b.to(device)
    x_aug = apply_affine(images, A_dev, b_dev, clip=False)  # pre-clip target, plan §3.1
    clip = float(clipped_fraction(images, A_dev, b_dev))

    M, m = analytic_operator(fit, A.cpu().float(), b.cpu().float(), K=k)
    M, m = M.to(device), m.to(device)
    C = z.shape[1]
    eye = torch.eye(C, device=device)
    zero = torch.zeros(C, device=device)

    with torch.no_grad():
        ee_l2 = float(ee_pix(decode_fn, z, M, m, x_aug, metric="l2"))
        ee_de = float(ee_pix(decode_fn, z, M, m, x_aug, metric="ciede2000"))
        floor_l2 = float(ee_pix(decode_fn, z, eye, zero, x_aug, metric="l2"))
        floor_de = float(ee_pix(decode_fn, z, eye, zero, x_aug, metric="ciede2000"))
    return ee_l2, ee_de, floor_l2, floor_de, clip


def _print_table(rows: list[tuple]) -> None:
    header = (f"{'factor':<12}{'magnitude':>10}{'ee_l2':>12}{'ee_de00':>12}"
              f"{'floor_l2':>12}{'floor_de00':>12}{'clip_frac':>11}")
    print(header)
    print("-" * len(header))
    for factor, mag, ee_l2, ee_de, fl2, fde, clip in rows:
        mag_s = f"{mag:10.4f}" if mag == mag else f"{'sampled':>10}"  # nan -> composed row
        print(f"{factor:<12}{mag_s}{ee_l2:12.6f}{ee_de:12.4f}"
              f"{fl2:12.6f}{fde:12.4f}{clip:11.4f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Decoder-side equivariance error of the analytic latent operator (plan §5.1(c))."
    )
    add_common_args(parser, n_images=16)
    parser.add_argument("--wfit", type=str, default="wfit.pt",
                        help="path to the fit saved by pheq.probes.fit_w")
    parser.add_argument("--k", "--K", dest="k", choices=("I", "0"), default="I",
                        help="null-space treatment K in the closed form (plan §3.2)")
    parser.add_argument("--n-compositions", type=int, default=8,
                        help="number of sampled composed PhotoParams for the summary row")
    parser.add_argument("--out", type=str, default="analytic_probe.csv",
                        help="output CSV path")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    size = resolve_image_size(args)

    images = get_images(args, size).to(device)
    model = build_vae(args.vae, device)
    freeze(model)
    decode_fn = make_decode_fn(model)
    z = encode_latents(model, images)

    fit = _load_or_fit(args.wfit, z, images)
    print(f"analytic_probe: vae={args.vae} n={images.shape[0]} image={size}x{size} "
          f"K={args.k} wfit_r2={float(fit.r2):.4f}")

    rows: list[tuple] = []
    for factor, magnitudes in DEFAULT_GRIDS.items():
        for mag in magnitudes:
            A, b = factor_affine(factor, mag)
            rows.append((factor, float(mag)) + _evaluate(z, images, A, b, fit, args.k, decode_fn))

    # Composed transforms: sample PhotoParams from the plan's ranges (plan §3.1)
    # and average the same quantities over the sampled group elements.
    rng = torch.Generator().manual_seed(args.seed + 1)
    params = PhotoParams.sample(rng, args.n_compositions)
    stats = torch.zeros(5, dtype=torch.float64)
    for p in params:
        A, b = p.affine()
        stats += torch.tensor(_evaluate(z, images, A, b, fit, args.k, decode_fn),
                              dtype=torch.float64)
    stats /= max(len(params), 1)
    rows.append(("composed", math.nan) + tuple(float(v) for v in stats))

    _print_table(rows)
    write_csv(args.out, CSV_HEADER, rows)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
