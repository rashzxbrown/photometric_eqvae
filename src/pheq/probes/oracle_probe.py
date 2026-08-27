"""Tier-0 probe (a): the decoder-inversion oracle, RQ0 (plan §1 RQ0, §5.1(a)).

For each photometric factor x magnitude on N images: optimize z' to minimize
L(tau_a(x), D(z')) starting from z_init = E(x) with the decoder frozen
(:func:`pheq.oracle.invert_latent`). Reports

- the oracle equivariance error (final inversion loss + CIEDE2000 of
  D(z_opt) vs tau_a(x)) — the decoder-expressivity ceiling, and
- the R² of a channel-affine fit z_orig -> z_opt
  (:func:`pheq.oracle.oracle_affine_fit`) — "is the oracle's latent edit
  channel-affine?", the analytic-form hypothesis test before any training,

per plan §5.1(a). Prints a table per factor x magnitude and saves a CSV.
The ``identity_l2`` column is the no-edit reference L2 of D(E(x)) vs
tau_a(x), so the improvement achieved by inversion is visible.

Usage::

    uv run python -m pheq.probes.oracle_probe [--vae sd|toy] [--images DIR] [--steps 300]
    uv run python -m pheq.probes.oracle_probe --vae toy    # offline smoke run (<1 min)
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from pheq.color import apply_affine
from pheq.metrics import mean_ciede2000
from pheq.oracle import invert_latent, oracle_affine_fit
from pheq.probes._common import (
    ORACLE_GRIDS,
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

CSV_HEADER = ("factor", "magnitude", "oracle_loss", "oracle_ciede2000",
              "affine_r2", "identity_l2")


def _print_table(rows: list[tuple]) -> None:
    header = (f"{'factor':<12}{'magnitude':>10}{'oracle_loss':>13}"
              f"{'oracle_de00':>13}{'affine_r2':>11}{'identity_l2':>13}")
    print(header)
    print("-" * len(header))
    for factor, mag, loss, de00, r2, id_l2 in rows:
        print(f"{factor:<12}{mag:10.4f}{loss:13.3e}{de00:13.4f}{r2:11.5f}{id_l2:13.3e}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="RQ0 decoder-inversion oracle: expressivity ceiling and affine-fit R2 (plan §5.1(a))."
    )
    add_common_args(parser, n_images=4)
    parser.add_argument("--steps", type=int, default=300,
                        help="Adam steps per inversion (pheq.oracle.invert_latent)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="images per inversion batch — memory-bound: the backward "
                             "pass through the decoder holds activations for the whole "
                             "batch (batch 16 at 256² OOMs a 24 GB GPU; 4 fits easily)")
    parser.add_argument("--lr", type=float, default=0.1, help="inversion learning rate")
    parser.add_argument("--loss", type=str, default="l2", help="inversion loss (default l2)")
    parser.add_argument("--out", type=str, default="oracle_probe.csv",
                        help="output CSV path")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    size = resolve_image_size(args)

    images = get_images(args, size).to(device)
    model = build_vae(args.vae, device)
    freeze(model)  # oracle optimizes z only; decoder params stay frozen (plan §1 RQ0)
    decode_fn = make_decode_fn(model)
    z0 = encode_latents(model, images)

    print(f"oracle_probe: vae={args.vae} n={images.shape[0]} image={size}x{size} "
          f"steps={args.steps} lr={args.lr}")

    total_rows = sum(len(m) for m in ORACLE_GRIDS.values())
    rows: list[tuple] = []
    for factor, magnitudes in ORACLE_GRIDS.items():
        for mag in magnitudes:
            A, b = factor_affine(factor, mag)
            # Pre-clip equivariance target, plan §3.1.
            x_target = apply_affine(images, A.to(device), b.to(device), clip=False)

            # Chunked inversion: backward through the decoder is memory-bound in
            # batch size, so invert args.batch_size images at a time and pool.
            n_img = images.shape[0]
            loss_sum = de00_sum = idl2_sum = 0.0
            pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
            for i in range(0, n_img, args.batch_size):
                zi = z0[i : i + args.batch_size]
                xt = x_target[i : i + args.batch_size]
                result = invert_latent(decode_fn, xt, zi,
                                       steps=args.steps, lr=args.lr, loss=args.loss)
                z_opt = result.z_opt.detach()
                with torch.no_grad():
                    de00_sum += float(mean_ciede2000(decode_fn(z_opt), xt)) * zi.shape[0]
                    idl2_sum += float(F.mse_loss(decode_fn(zi), xt)) * zi.shape[0]
                loss_sum += float(result.final_loss) * zi.shape[0]
                # Per-image pairs for the channel-affine fit (plan §5.1(a)).
                pairs.extend((zi[j : j + 1].cpu(), z_opt[j : j + 1].cpu())
                             for j in range(zi.shape[0]))
            final_loss = loss_sum / n_img
            de00 = de00_sum / n_img
            identity_l2 = idl2_sum / n_img

            affine_fit = oracle_affine_fit(pairs)
            r2 = float(getattr(affine_fit, "r2", affine_fit))

            rows.append((factor, float(mag), final_loss, de00, r2, identity_l2))
            print(f"[{len(rows)}/{total_rows}] {factor} mag={mag:+.4f} "
                  f"oracle_loss={final_loss:.3e} de00={de00:.2f} affine_r2={r2:.4f}",
                  flush=True)
            # incremental save: partial results survive interruption on long runs
            write_csv(args.out, CSV_HEADER, rows)

    _print_table(rows)
    write_csv(args.out, CSV_HEADER, rows)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
