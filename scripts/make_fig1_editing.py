"""Figure 1: qualitative latent color editing with the closed-form operator.

Loads an AE run checkpoint (SPEC2 format, ``pheq.train_ae``), encodes each of
``--n`` sample images ONCE, applies the checkpoint's analytic channel-affine
operator (``pheq.analytic``: ``M_a = W+ A_a W + (I - W+W) K`` built from the
ckpt's stored wfit) for a sweep of photometric transforms, decodes each edited
latent, and assembles a labeled grid (rows = images, cols = transforms).

Sweep (plan §3.1 factors): brightness β ∈ {0.7, 1.0, 1.3},
saturation s ∈ {0.05, 1.0, 1.5}, hue θ ∈ {−45°, 0, +45°}. The three identity
midpoints (β=1, s=1, θ=0) coincide — all are decode(encode(x)) — so the
identity is rendered once, as the "recon" column next to the input.

Everything happens in latent space: the pixel-space affine (A, b) is never
applied to the image, only mapped through the wfit into (M, m).

Intended to run ON THE CLUSTER against a real run checkpoint, e.g.::

    uv run python scripts/make_fig1_editing.py \
        --ckpt outputs/p1_analytic/ckpt_latest.pt \
        --images data/openimages/val --n 4 \
        --out paper/figures/fig1_editing

Local toy smoke (ToyConvAE checkpoint from ``pheq.train_ae --vae toy``)::

    uv run python scripts/make_fig1_editing.py \
        --ckpt /tmp/toyrun/ckpt_latest.pt --images /tmp/toyimgs --n 2 \
        --out /tmp/fig1_toy --device cpu

Writes ``<out>.png`` and ``<out>.pdf``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from pheq.analytic import analytic_operator, apply_channel_affine
from pheq.color import brightness_affine, hue_affine, saturation_affine
from pheq.data import ImageFolderDataset
from pheq.eval_battery import _wfit_from_ckpt, load_run_checkpoint
from pheq.fid import _resolve_vae


def build_columns() -> list[tuple[str, torch.Tensor, torch.Tensor] | tuple[str, None, None]]:
    """(header, A, b) per grid column; (header, None, None) = identity recon.

    The identity midpoints of the three sweeps (β=1, s=1, θ=0) all decode the
    unedited latent, so they collapse into the single "recon" column.
    """
    deg = math.pi / 180.0
    cols: list[tuple[str, torch.Tensor | None, torch.Tensor | None]] = [
        ("input", None, None),  # sentinel: raw image, no encode/decode
        ("recon\n(identity)", *(None, None)),
    ]
    for beta in (0.7, 1.3):
        cols.append((f"brightness\nβ = {beta}", *brightness_affine(beta)))
    for s in (0.05, 1.5):
        cols.append((f"saturation\ns = {s}", *saturation_affine(s)))
    for theta_deg in (-45, 45):
        cols.append((f"hue\nθ = {theta_deg:+d}°", *hue_affine(theta_deg * deg)))
    return cols


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Figure 1: closed-form latent color-editing grid."
    )
    parser.add_argument("--ckpt", type=str, required=True,
                        help="SPEC2 run checkpoint (pheq.train_ae ckpt_latest.pt)")
    parser.add_argument("--images", type=str, required=True,
                        help="image directory (recursive; first --n in sorted order)")
    parser.add_argument("--n", type=int, default=4, help="number of rows/images")
    parser.add_argument("--out", type=str, default="paper/figures/fig1_editing",
                        help="output path stem (writes <out>.png and <out>.pdf)")
    parser.add_argument("--device", type=str, default=None,
                        help="cpu | cuda | mps (default: auto-detect)")
    parser.add_argument("--image-size", type=int, default=None,
                        help="override the checkpoint config's image_size")
    args = parser.parse_args(argv)

    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    ckpt = load_run_checkpoint(args.ckpt)
    vae = _resolve_vae(ckpt, device)
    wfit = _wfit_from_ckpt(ckpt)
    image_size = args.image_size or int(ckpt.get("config", {}).get("image_size", 256))
    print(
        f"fig1: ckpt={args.ckpt} condition={ckpt.get('condition')} "
        f"step={ckpt.get('step')} image_size={image_size} device={device}"
    )

    ds = ImageFolderDataset(args.images, size=image_size)
    n = min(args.n, len(ds))
    imgs = torch.stack([ds[i][0] for i in range(n)]).to(device)  # (n, 3, H, W)

    cols = build_columns()
    with torch.no_grad():
        z = vae.encode_moments(imgs)[0]  # encode ONCE per image
        grid: list[list[torch.Tensor]] = [[] for _ in range(n)]
        for header, A, b in cols:
            if header == "input":
                out = imgs
            elif A is None:
                out = vae.decode_latents(z)
            else:
                M, m = analytic_operator(wfit, A, b, K="I")
                z_edit = apply_channel_affine(
                    z, M.to(dtype=z.dtype, device=z.device),
                    m.to(dtype=z.dtype, device=z.device),
                )
                out = vae.decode_latents(z_edit)
            out = out.clamp(0.0, 1.0).cpu()
            for row in range(n):
                grid[row].append(out[row])

    ncols = len(cols)
    cell = 1.35
    fig, axes = plt.subplots(
        n, ncols, figsize=(ncols * cell, n * cell + 0.42), squeeze=False
    )
    for row in range(n):
        for col in range(ncols):
            ax = axes[row][col]
            ax.imshow(grid[row][col].permute(1, 2, 0).numpy(),
                      interpolation="lanczos")
            ax.set_axis_off()
            if row == 0:
                ax.set_title(cols[col][0], fontsize=8)
    fig.tight_layout(pad=0.35)

    out_stem = Path(args.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        path = out_stem.with_suffix(suffix)
        fig.savefig(path, dpi=200)
        print(f"fig1: wrote {path} ({path.stat().st_size} bytes)")
    plt.close(fig)


if __name__ == "__main__":
    main()
