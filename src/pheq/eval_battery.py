"""Per-checkpoint evaluation battery (SPEC2 "eval_battery.py"; plan §5.3, §3.5).

Consumes the SPEC2 run-checkpoint format exactly::

    {"vae": state_dict, "operator": state_dict | None, "operator_kind": str,
     "condition": str, "step": int, "wfit": {"W": (3, C), "c": (3,)},
     "config": dict}

and produces a single JSON report with:

- reconstruction metrics (PSNR, L1, LPIPS when available, rFID when
  ``rfid_n > 0`` via :func:`pheq.fid.rfid`);
- the decoder-side equivariance-error battery (L2 + CIEDE2000, plan §5.3)
  over the factor × magnitude grids REUSED from ``pheq.probes._common``
  (``DEFAULT_GRIDS`` ∪ ``ORACLE_GRIDS`` — imported, never duplicated), plus
  held-out magnitudes and sampled held-out compositions;
- the oracle-gap-closed fraction per factor — the sprint headline metric of
  docs/plan-3month.md — ONLY when BOTH ``b1_json`` AND ``oracle_csv`` are
  provided (the block is omitted otherwise);
- latent statistics (per-channel mu std, effective rank, spectral slope,
  high-frequency fraction — pheq.spectral);
- collapse diagnostics (plan §3.5 / risk R3): the cross-operator audit
  (EE with the FROZEN analytic operator built from the ckpt's wfit, flagging
  E/g co-adaptation) and the swapped-decoder test (EE through a freshly
  loaded PRETRAINED SD decoder; lazily loaded, skipped for toy checkpoints);
- the spatial-EE block ``ee_spatial`` (ALWAYS computed; docs/tier0-log.md
  WAVE-1 caveat 2): EE of the EXACT inherited spatial operator
  (pheq.spatial) for rot90 k∈{1,2} and scale ∈ {0.5, 0.25} — the b2lite
  known-good-anchor measurement;
- OPTIONALLY (``refit_w=True`` / ``--refit-w``; docs/tier0-log.md WAVE-1
  caveat 1): a second EE block ``ee_refit_w`` from an analytic operator whose
  (W, c) is REFIT on the checkpoint's own latents over the eval images, plus
  the refit R² under ``refit_w`` — the fair b1 reference (the stored-wfit EE
  of a plainly fine-tuned run conflates latent drift with lack of
  equivariance).

The checkpoint may also be built IN MEMORY by :func:`make_frozen_checkpoint`
(``--frozen-vae {sd,toy} --wfit PATH``; docs/tier0-log.md WAVE-1 caveat 3):
the PRETRAINED, never-fine-tuned VAE evaluated as reference row "b0" at the
same eval settings.

Spec resolutions (documented per SPEC2 ground rules):

- "held-out magnitudes (midpoints of the training ranges)": the literal
  range midpoint is the identity for beta/gamma/hue (e.g. (0.7+1.3)/2 = 1),
  so we evaluate the two midpoints between the identity value and each end
  of the checkpoint's training range — e.g. beta ∈ [0.7, 1.3] → {0.85, 1.15},
  genuinely off the probe grids.
- The battery always evaluates ALL FOUR factors, also for single-factor
  conditions (cross-factor transfer is part of the measurement).
- Gap-closed uses CIEDE2000 (the oracle CSV's ``oracle_ciede2000`` column is
  the only EE there in comparable units), averaged over the magnitudes
  shared by this battery, the b1 reference battery, and the oracle CSV.
  Numerator and denominator must share the W convention: when the b1
  reference is its ``ee_refit_w`` block and this checkpoint's operator is
  the analytic one from its STORED wfit ('none'/'analytic'), the numerator
  uses this report's own ``ee_refit_w`` grid (computed with ``--refit-w``)
  rather than the stored-wfit ``ee`` grid — otherwise the checkpoint's own
  latent drift is silently counted against it (b1 vs its own refit JSON
  must close exactly 0% of its own gap). The block used is recorded as
  ``ckpt_ee_block``; 'lie'/'conv' always use ``ee`` (the co-trained
  operator is the object under evaluation).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from pheq import fid as fid_mod
from pheq.analytic import WFit, analytic_operator, apply_channel_affine, fit_w
from pheq.color import PhotoParams, apply_affine, clipped_fraction
from pheq.conditions import DEFAULT_PHOTO_RANGES, FACTOR_FIELDS, TrainConfig
from pheq.metrics import ee_pix
from pheq.probes._common import DEFAULT_GRIDS, FACTORS, ORACLE_GRIDS
from pheq.spatial import SpatialParams, apply_spatial
from pheq.spectral import effective_rank, high_freq_fraction, spectral_slope

__all__ = ["evaluate", "gap_closed_fraction", "make_frozen_checkpoint", "main"]

#: Identity value per photometric factor (plan §3.1: beta = gamma = sat = 1,
#: hue = 0), keyed by factor name.
_FACTOR_IDENTITY: dict[str, float] = {
    "brightness": 1.0,
    "contrast": 1.0,
    "saturation": 1.0,
    "hue": 0.0,
}

#: Decimal places used to match float magnitudes across JSON/CSV round-trips.
_MAG_DECIMALS = 6

_REQUIRED_CKPT_KEYS = ("vae", "operator", "operator_kind", "condition", "step", "wfit")

#: The spatial-EE op set (docs/tier0-log.md WAVE-1 caveat 2): rot90 k ∈ {1, 2}
#: and isotropic scale ∈ {0.5, 0.25} — cheap, exact latent operators
#: (pheq.spatial inherits them from the grid, plan §2).
_SPATIAL_EE_OPS: tuple[tuple[str, SpatialParams], ...] = (
    ("rot90_k1", SpatialParams(rot90=1)),
    ("rot90_k2", SpatialParams(rot90=2)),
    ("scale_0.5", SpatialParams(scale=0.5)),
    ("scale_0.25", SpatialParams(scale=0.25)),
)


# ---------------------------------------------------------------------------
# Run-checkpoint consumption
# ---------------------------------------------------------------------------


def load_run_checkpoint(path: str | dict) -> dict:
    """Load and validate a SPEC2 run checkpoint (trusted local artifact).

    ``weights_only=False`` because checkpoints carry a config dict (and, for
    ``ckpt_latest.pt``, optimizer/RNG state) — they are produced locally by
    ``pheq.train_ae``. Extra keys (optimizer, RNG) are tolerated; the SPEC2
    required keys must be present.

    An already-built run-checkpoint ``dict`` (e.g. from
    :func:`make_frozen_checkpoint`, the in-memory b0 reference of
    docs/tier0-log.md WAVE-1 caveat 3) is validated and returned as-is.
    """
    if isinstance(path, dict):
        ckpt: Any = path
    else:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"run checkpoint must be a dict, got {type(ckpt).__name__}")
    missing = [k for k in _REQUIRED_CKPT_KEYS if k not in ckpt]
    if missing:
        name = "<in-memory>" if isinstance(path, dict) else repr(str(path))
        raise KeyError(
            f"run checkpoint {name} missing key(s) {missing} "
            "(SPEC2 run-checkpoint format)"
        )
    return ckpt


def make_frozen_checkpoint(
    arch: str,
    wfit_path: str,
    image_size: int | None = None,
) -> dict:
    """Build the in-memory "b0" run checkpoint of the PRETRAINED (never
    fine-tuned) VAE — reference row b0 (docs/tier0-log.md WAVE-1 caveat 3:
    "no frozen-VAE (B0) row at the same eval settings for reference").

    The dict is a complete SPEC2 run checkpoint (condition "b0",
    operator_kind "none", step 0, wfit loaded from ``wfit_path``), so
    everything downstream — the EE battery, latent stats, rFID — works
    unchanged via :func:`evaluate`. The ``config`` sub-dict is built from a
    real :class:`pheq.conditions.TrainConfig` (``asdict`` + the ``"vae"``
    key), so its schema matches ``pheq.train_ae._save_ckpt``'s writer
    key-for-key — a future consumer reading e.g. ``config["condition"]`` or
    ``config["seed"]`` behaves identically in frozen-b0 mode.

    Args:
        arch: 'sd' (pretrained ``stabilityai/sd-vae-ft-mse`` via
            :func:`pheq.vae.load_sd_vae`, lazy diffusers import) or 'toy'
            (:class:`pheq.vae.ToyConvAE` at its deterministic defaults —
            channels=4, hidden=16, seed=0).
        wfit_path: ``wfit.pt`` payload (``pheq.probes.fit_w`` convention:
            a dict with at least ``{"W": (3, C), "c": (3,)}``) fit on the
            FROZEN VAE's latents.
        image_size: eval resolution stored in ``config["image_size"]``;
            defaults to the probe conventions (256 for 'sd', 64 for 'toy').

    Returns:
        A SPEC2 run-checkpoint dict, entirely in memory.
    """
    if arch == "toy":
        from pheq.vae import ToyConvAE

        vae: torch.nn.Module = ToyConvAE()
        size = 64 if image_size is None else int(image_size)
    elif arch == "sd":
        from pheq.vae import load_sd_vae  # lazy: pretrained weights

        vae = load_sd_vae(device="cpu")
        size = 256 if image_size is None else int(image_size)
    else:
        raise ValueError(f"frozen-vae arch must be 'sd' or 'toy', got {arch!r}")

    payload = torch.load(str(wfit_path), map_location="cpu", weights_only=False)
    return {
        "vae": {k: v.detach().cpu() for k, v in vae.state_dict().items()},
        "operator": None,
        "operator_kind": "none",
        "condition": "b0",
        "step": 0,
        "wfit": {
            "W": torch.as_tensor(payload["W"], dtype=torch.float32).cpu(),
            "c": torch.as_tensor(payload["c"], dtype=torch.float32).cpu(),
        },
        # Same config schema as train_ae's writer (asdict(TrainConfig) +
        # "vae"): b0 is the never-trained condition, so steps=0 and the
        # remaining fields at their registry defaults describe it exactly.
        "config": {
            **asdict(
                TrainConfig(
                    condition="b0",
                    operator_kind="none",
                    tau_photo=False,
                    tau_spatial=False,
                    image_size=size,
                    steps=0,
                )
            ),
            "vae": arch,
        },
    }


def _wfit_from_ckpt(ckpt: dict) -> WFit:
    """FROZEN analytic fit from the checkpoint's ``wfit`` block (SPEC2 care point).

    The run checkpoint stores only ``{"W", "c"}``; R² fields (unused by
    :func:`pheq.analytic.analytic_operator`) are filled with NaN.
    """
    w = torch.as_tensor(ckpt["wfit"]["W"], dtype=torch.float32).cpu()
    c = torch.as_tensor(ckpt["wfit"]["c"], dtype=torch.float32).cpu()
    return WFit(W=w, c=c, r2=float("nan"), r2_per_channel=torch.full((3,), float("nan")))


def _load_operator(ckpt: dict, device: str) -> torch.nn.Module:
    """Rebuild the checkpoint's learned operator ('lie' | 'conv') from its state dict.

    Constructor hyper-parameters are inferred from the state-dict shapes
    (channels/hidden/n_freq), so non-default operators round-trip.
    """
    kind = str(ckpt["operator_kind"])
    state = ckpt["operator"]
    if state is None:
        raise ValueError(f"operator_kind={kind!r} but checkpoint has operator=None")
    if kind == "lie":
        from pheq.lie_operator import LieAffineOperator

        op: torch.nn.Module = LieAffineOperator(
            channels=int(state["G"].shape[1]),
            hidden=int(state["mlp.0.weight"].shape[0]),
            n_freq=int(state["freqs"].numel()),
        )
    elif kind == "conv":
        from pheq.conv_operator import ConvResidualOperator

        op = ConvResidualOperator(
            _wfit_from_ckpt(ckpt),
            hidden=int(state["conv_in.weight"].shape[0]),
            n_freq=int(state["freqs"].numel()),
        )
    else:
        raise ValueError(f"unknown operator_kind {kind!r} (expected 'lie' or 'conv')")
    op.load_state_dict(state)
    op.to(device).eval()
    for p in op.parameters():
        p.requires_grad_(False)
    return op


def _analytic_apply_fn(fit: WFit) -> Callable[[torch.Tensor, PhotoParams], torch.Tensor]:
    """``(z, params) -> M z + m`` with the closed-form operator (plan §3.2, K='I').

    (M, m) are computed on CPU float32 (deterministic pinv) and moved to the
    latent's device per call.
    """

    def apply_op(z: torch.Tensor, params: PhotoParams) -> torch.Tensor:
        a_mat, b_vec = params.affine()
        m_mat, m_vec = analytic_operator(fit, a_mat, b_vec, K="I")
        return apply_channel_affine(
            z, m_mat.to(device=z.device, dtype=z.dtype), m_vec.to(device=z.device, dtype=z.dtype)
        )

    return apply_op


def _make_apply_op(ckpt: dict, device: str) -> Callable[[torch.Tensor, PhotoParams], torch.Tensor]:
    """THE CHECKPOINT'S operator as a uniform ``(z, params) -> z'`` callable.

    SPEC2: analytic from the ckpt's wfit when ``operator_kind`` is 'none' or
    'analytic'; the co-trained module (state dict in the ckpt) for
    'lie'/'conv' (applied via canonical coordinates ``params.phi()``,
    plan §3.3).
    """
    kind = str(ckpt["operator_kind"])
    if kind in ("none", "analytic"):
        return _analytic_apply_fn(_wfit_from_ckpt(ckpt))
    op = _load_operator(ckpt, device)

    def apply_op(z: torch.Tensor, params: PhotoParams) -> torch.Tensor:
        return op(z, params.phi().to(z.device))

    return apply_op


# ---------------------------------------------------------------------------
# EE battery
# ---------------------------------------------------------------------------


def _grid_for(factor: str) -> tuple[float, ...]:
    """Eval magnitudes: union of the analytic-probe and oracle grids (imported)."""
    return tuple(sorted(set(DEFAULT_GRIDS[factor]) | set(ORACLE_GRIDS[factor])))


def _single_factor_params(factor: str, magnitude: float) -> PhotoParams:
    """PhotoParams with one factor at ``magnitude``, others at identity."""
    return PhotoParams(**{FACTOR_FIELDS[factor]: float(magnitude)})


def _ee_pair(
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    z_op: torch.Tensor,
    x_aug: torch.Tensor,
    batch: int,
) -> tuple[float, float]:
    """(ee_l2, ee_ciede2000) of decoding ``z_op`` against ``x_aug``, chunked.

    Reuses :func:`pheq.metrics.ee_pix` with the identity channel affine
    (M = I, m = 0) applied to the ALREADY-transformed latent, so the metric
    reductions (safe-sqrt L2, mean CIEDE2000) stay in one place. Chunk means
    are weighted by chunk size (all images share a pixel count, so this
    equals the full-batch mean).
    """
    c_lat = z_op.shape[1]
    eye = torch.eye(c_lat, device=z_op.device, dtype=z_op.dtype)
    zero = torch.zeros(c_lat, device=z_op.device, dtype=z_op.dtype)
    tot_l2 = tot_de = 0.0
    n = 0
    with torch.no_grad():
        for i in range(0, z_op.shape[0], batch):
            zc, xc = z_op[i : i + batch], x_aug[i : i + batch]
            k = int(zc.shape[0])
            tot_l2 += float(ee_pix(decode_fn, zc, eye, zero, xc, metric="l2")) * k
            tot_de += float(ee_pix(decode_fn, zc, eye, zero, xc, metric="ciede2000")) * k
            n += k
    return tot_l2 / n, tot_de / n


def _battery_rows(
    items: Sequence[tuple[str, float]],
    images: torch.Tensor,
    z: torch.Tensor,
    apply_op: Callable[[torch.Tensor, PhotoParams], torch.Tensor],
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    batch: int,
) -> list[dict]:
    """One EE row per (factor, magnitude): pre-clip target (plan §3.1), operator
    on the latent, decoder-side L2 + CIEDE2000 (plan §5.3), measured clip fraction."""
    rows: list[dict] = []
    with torch.no_grad():
        for factor, mag in items:
            params = _single_factor_params(factor, mag)
            a_mat, b_vec = params.affine()
            x_aug = apply_affine(images, a_mat, b_vec, clip=False)
            clip = float(clipped_fraction(images, a_mat, b_vec))
            z_op = apply_op(z, params)
            ee_l2, ee_de = _ee_pair(decode_fn, z_op, x_aug, batch)
            rows.append(
                {
                    "factor": factor,
                    "magnitude": float(mag),
                    "ee_l2": ee_l2,
                    "ee_ciede2000": ee_de,
                    "clip_frac": clip,
                }
            )
    return rows


def _held_out_items(photo_ranges: dict) -> list[tuple[str, float]]:
    """Held-out magnitudes: midpoints between identity and each training-range end.

    See the module docstring for why the literal range midpoint (the identity
    for three of four factors) is not used.
    """
    items: list[tuple[str, float]] = []
    for factor in FACTORS:
        lo, hi = (float(v) for v in photo_ranges[FACTOR_FIELDS[factor]])
        ident = _FACTOR_IDENTITY[factor]
        for mag in ((lo + ident) / 2.0, (ident + hi) / 2.0):
            if not math.isclose(mag, ident, abs_tol=1e-12):
                items.append((factor, mag))
    return items


def _composition_stats(
    params_list: Sequence[PhotoParams],
    images: torch.Tensor,
    z: torch.Tensor,
    apply_op: Callable[[torch.Tensor, PhotoParams], torch.Tensor],
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    batch: int,
) -> dict:
    """Mean EE over sampled composed PhotoParams (plan §3.1 composed elements)."""
    tot = torch.zeros(3, dtype=torch.float64)
    with torch.no_grad():
        for p in params_list:
            a_mat, b_vec = p.affine()
            x_aug = apply_affine(images, a_mat, b_vec, clip=False)
            clip = float(clipped_fraction(images, a_mat, b_vec))
            ee_l2, ee_de = _ee_pair(decode_fn, apply_op(z, p), x_aug, batch)
            tot += torch.tensor([ee_l2, ee_de, clip], dtype=torch.float64)
    tot /= max(len(params_list), 1)
    return {
        "n": len(params_list),
        "ee_l2": float(tot[0]),
        "ee_ciede2000": float(tot[1]),
        "clip_frac": float(tot[2]),
    }


def _per_factor_mean(rows: Sequence[dict]) -> dict[str, dict[str, float]]:
    """Per-factor means of ee_l2 / ee_ciede2000 over a list of battery rows."""
    acc: dict[str, list[dict]] = {}
    for r in rows:
        acc.setdefault(r["factor"], []).append(r)
    return {
        f: {
            "ee_l2": sum(r["ee_l2"] for r in rs) / len(rs),
            "ee_ciede2000": sum(r["ee_ciede2000"] for r in rs) / len(rs),
        }
        for f, rs in acc.items()
    }


# ---------------------------------------------------------------------------
# Spatial-EE block (docs/tier0-log.md WAVE-1 caveat 2)
# ---------------------------------------------------------------------------


def _spatial_ee_block(
    images: torch.Tensor,
    z: torch.Tensor,
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    batch: int,
    f_img: int,
) -> dict:
    """EE of the EXACT inherited spatial operator, per op and mean.

    Resolves docs/tier0-log.md WAVE-1 caveat 2: "the eval battery measures
    photometric EE only — b2lite's known-good-anchor claim (spatial EE drops
    vs b1) needs a spatial-EE block added to the battery." For each op in
    :data:`_SPATIAL_EE_OPS`,

        EE = metric(D(apply_spatial(z, op)), apply_spatial(x, op))

    for L2 + CIEDE2000, averaged over the eval images. The spatial operator
    is exact (plan §2: inherited from the grid; rot90 is a site permutation,
    bilinear interpolation a convex combination), so this measures how well
    the LATENT respects spatial structure — no operator fit is involved.

    f-alignment (pheq.spatial conventions): the latent side uses ``f = 1``
    (target ``round(h*s)``) and the image side ``f = f_img`` (the AE
    downsampling factor; target ``round(H*s/f)*f``), keeping decoded and
    target sizes equal for the scale ops.

    Args:
        images: (B, 3, H, W) eval images in [0, 1].
        z: (B, C, h, w) latents (posterior means) of ``images``.
        decode_fn: latent → image decoder.
        batch: decode chunk size.
        f_img: AE downsampling factor (H // h; 8 for the SD-VAE, 2 for toys).

    Returns:
        ``{"ops": [{"op", "rot90", "scale", "ee_l2", "ee_ciede2000"}, ...],
        "mean": {"ee_l2", "ee_ciede2000"}}``.
    """
    rows: list[dict] = []
    with torch.no_grad():
        for name, params in _SPATIAL_EE_OPS:
            z_op = apply_spatial(z, params, antialias=True, f=1)
            x_aug = apply_spatial(images, params, antialias=True, f=f_img)
            ee_l2, ee_de = _ee_pair(decode_fn, z_op, x_aug, batch)
            rows.append(
                {
                    "op": name,
                    "rot90": int(params.rot90),
                    "scale": float(params.scale),
                    "ee_l2": ee_l2,
                    "ee_ciede2000": ee_de,
                }
            )
    mean = {
        "ee_l2": sum(r["ee_l2"] for r in rows) / len(rows),
        "ee_ciede2000": sum(r["ee_ciede2000"] for r in rows) / len(rows),
    }
    return {"ops": rows, "mean": mean}


# ---------------------------------------------------------------------------
# Oracle-gap-closed (sprint headline metric, docs/plan-3month.md)
# ---------------------------------------------------------------------------


def gap_closed_fraction(
    ee_ckpt: dict[str, dict[float, float]],
    ee_b1: dict[str, dict[float, float]],
    ee_oracle: dict[str, dict[float, float]],
) -> dict[str, dict]:
    """``(EE_b1 − EE_ckpt) / (EE_b1 − EE_oracle)`` per factor (plan-3month.md).

    Each argument maps ``factor -> {magnitude: EE}`` (CIEDE2000). Per factor,
    EEs are averaged over the magnitudes present in ALL THREE maps before the
    fraction is formed; factors with no shared magnitude are omitted, and a
    degenerate gap (``EE_b1 ≈ EE_oracle``) yields ``gap_closed = None``.

    Pure math on plain dicts so tests can hit it with synthetic numbers.
    """
    out: dict[str, dict] = {}
    for factor in ee_ckpt:
        if factor not in ee_b1 or factor not in ee_oracle:
            continue
        shared = sorted(set(ee_ckpt[factor]) & set(ee_b1[factor]) & set(ee_oracle[factor]))
        if not shared:
            continue
        mean_c = sum(ee_ckpt[factor][m] for m in shared) / len(shared)
        mean_b = sum(ee_b1[factor][m] for m in shared) / len(shared)
        mean_o = sum(ee_oracle[factor][m] for m in shared) / len(shared)
        denom = mean_b - mean_o
        gap = (mean_b - mean_c) / denom if abs(denom) > 1e-12 else None
        out[factor] = {
            "ee_ckpt": mean_c,
            "ee_b1": mean_b,
            "ee_oracle": mean_o,
            "n_magnitudes": len(shared),
            "gap_closed": gap,
        }
    return out


def _de00_map_from_rows(rows: Sequence[dict]) -> dict[str, dict[float, float]]:
    """``factor -> {rounded magnitude: ee_ciede2000}`` from battery grid rows."""
    out: dict[str, dict[float, float]] = {}
    for r in rows:
        out.setdefault(r["factor"], {})[round(float(r["magnitude"]), _MAG_DECIMALS)] = float(
            r["ee_ciede2000"]
        )
    return out


def _de00_map_from_oracle_csv(path: str) -> dict[str, dict[float, float]]:
    """Oracle EE map from an oracle-probe CSV (columns per pheq.probes.oracle_probe:
    ``factor, magnitude, oracle_loss, oracle_ciede2000, affine_r2, identity_l2``)."""
    out: dict[str, dict[float, float]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.setdefault(row["factor"], {})[
                round(float(row["magnitude"]), _MAG_DECIMALS)
            ] = float(row["oracle_ciede2000"])
    return out


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------


def evaluate(
    ckpt_path: str | dict,
    image_dir: str,
    out_json: str,
    device: str = "cpu",
    n_images: int = 64,
    rfid_n: int = 0,
    b1_json: str | None = None,
    oracle_csv: str | None = None,
    batch: int = 8,
    seed: int = 0,
    use_lpips: bool = True,
    refit_w: bool = False,
) -> dict:
    """Run the full per-checkpoint battery; return the report dict and write it
    to ``out_json`` (SPEC2 "eval_battery.py").

    Args:
        ckpt_path: SPEC2 run checkpoint (``ckpt_latest.pt``; extra
            optimizer/RNG keys tolerated), or an already-built run-checkpoint
            dict (e.g. :func:`make_frozen_checkpoint`'s in-memory b0
            reference, docs/tier0-log.md WAVE-1 caveat 3).
        image_dir: eval images, loaded via :class:`pheq.data.ImageFolderDataset`
            at the checkpoint's ``config["image_size"]`` (default 256) in
            deterministic sorted order; the first ``n_images`` are used for
            reconstruction, the EE battery, and latent stats.
        out_json: report path (parents created). rFID scratch PNGs go to
            ``<out_json stem>_rfid/`` next to it.
        device: torch device string.
        n_images: eval-batch size (capped at the dataset size).
        rfid_n: images for :func:`pheq.fid.rfid`; 0 disables rFID.
        b1_json: eval-battery JSON of the b1 reference run. The
            oracle-gap-closed block appears ONLY when BOTH ``b1_json`` AND
            ``oracle_csv`` are given (SPEC2), and is omitted otherwise. Its
            ``ee_refit_w`` block is preferred as the reference; when the
            reference IS the refit block and this checkpoint's operator is
            analytic-from-stored-wfit ('none'/'analytic'), the numerator uses
            this report's own ``ee_refit_w`` grid too (requires
            ``refit_w=True``) so both sides share the W convention — the
            block fed to the numerator is recorded as ``ckpt_ee_block``.
        oracle_csv: Tier-0 oracle-probe CSV (oracle EE ceilings).
        batch: encode/decode chunk size.
        seed: seed for the held-out composition sampling (independent of the
            training RNG stream — the draws are held out by construction).
        use_lpips: attempt LPIPS (net='vgg') for the reconstruction block;
            on failure (package/weights unavailable) the value is reported as
            None. Pass False to skip the attempt entirely (offline tests).
        refit_w: when True, refit (W, c) on the CHECKPOINT'S OWN latents over
            the eval images (:func:`pheq.analytic.fit_w` on the encoded mu vs
            the same images) and compute a SECOND EE block ``ee_refit_w``
            with the analytic operator built from the refit, alongside the
            stored-wfit ``ee`` block; the refit R² is recorded under
            ``refit_w``. This is the fair b1 reference of docs/tier0-log.md
            WAVE-1 caveat 1: b1's stored-wfit EE conflates the latent-drift
            of plain fine-tuning with lack of equivariance.

    Returns:
        The report dict (same content as the JSON on disk).
    """
    from pheq.data import ImageFolderDataset  # sibling v2 module; lazy per fid.py convention

    ckpt = load_run_checkpoint(ckpt_path)
    config = dict(ckpt.get("config") or {})
    image_size = int(config.get("image_size", 256))
    photo_ranges = {
        k: tuple(float(x) for x in v)
        for k, v in dict(config.get("photo_ranges") or DEFAULT_PHOTO_RANGES).items()
    }
    is_toy = str(config.get("vae", "sd")) == "toy"

    # VAE resolution is shared with pheq.fid (same run-checkpoint consumer);
    # _resolve_vae is package-internal, imported deliberately (do not duplicate).
    vae = fid_mod._resolve_vae(ckpt, device)
    decode_fn = vae.decode_latents
    apply_op = _make_apply_op(ckpt, device)

    # ----- eval batch: images, moments, reconstruction ---------------------
    dataset = ImageFolderDataset(image_dir, size=image_size)
    n_eff = min(int(n_images), len(dataset))
    if n_eff == 0:
        raise ValueError(f"no images found under {image_dir!r}")

    recon_loss = None
    if use_lpips:
        try:
            from pheq.losses import ReconLoss

            rl = ReconLoss(require_lpips=False)
            recon_loss = rl if rl.lpips_active else None
            if recon_loss is not None:
                recon_loss.to(device)
        except Exception:
            recon_loss = None  # reported as null; CLIs surface this

    images_chunks: list[torch.Tensor] = []
    mu_chunks: list[torch.Tensor] = []
    sq_err_sum = 0.0
    abs_err_sum = 0.0
    lpips_sum = 0.0
    px_count = 0
    with torch.no_grad():
        for start in range(0, n_eff, batch):
            x = torch.stack(
                [dataset[i][0] for i in range(start, min(start + batch, n_eff))]
            ).to(device)
            mu, _sigma = vae.encode_moments(x)
            recon = decode_fn(mu)
            # PSNR on [0,1]-clamped output (display metric); L1 pre-clip
            # unclamped, matching the training loss convention (plan §3.1).
            sq_err_sum += float((recon.clamp(0.0, 1.0) - x).pow(2).sum())
            abs_err_sum += float((recon - x).abs().sum())
            px_count += int(x.numel())
            if recon_loss is not None:
                lpips_sum += float(recon_loss.lpips_term(recon, x)) * x.shape[0]
            images_chunks.append(x)
            mu_chunks.append(mu)
    images = torch.cat(images_chunks)
    z = torch.cat(mu_chunks)  # posterior mean, the probes' latent convention

    mse = max(sq_err_sum / px_count, 1e-12)  # cap PSNR at 120 dB (JSON-safe)
    reconstruction: dict[str, Any] = {
        "psnr": 10.0 * math.log10(1.0 / mse),
        "l1": abs_err_sum / px_count,
        "lpips": (lpips_sum / n_eff) if recon_loss is not None else None,
        "rfid": None,
    }
    if rfid_n > 0:
        out_path = Path(out_json)
        reconstruction["rfid"] = fid_mod.rfid(
            vae,
            image_dir,
            str(out_path.parent / f"{out_path.stem}_rfid"),
            n=int(rfid_n),
            device=device,
            batch=batch,
            size=image_size,
        )

    # ----- EE battery ------------------------------------------------------
    grid_items = [(f, m) for f in FACTORS for m in _grid_for(f)]
    held_items = _held_out_items(photo_ranges)
    grid_rows = _battery_rows(grid_items, images, z, apply_op, decode_fn, batch)
    held_rows = _battery_rows(held_items, images, z, apply_op, decode_fn, batch)
    comp_gen = torch.Generator().manual_seed(int(seed) + 7919)  # eval-only stream
    comp_params = PhotoParams.sample(comp_gen, 8, ranges=photo_ranges)
    compositions = _composition_stats(comp_params, images, z, apply_op, decode_fn, batch)
    compositions["seed"] = int(seed)

    # ----- spatial-EE block (always computed; tier0-log WAVE-1 caveat 2) ----
    f_img = images.shape[-1] // z.shape[-1]  # AE downsampling factor (8 sd, 2 toy)
    ee_spatial = _spatial_ee_block(images, z, decode_fn, batch, f_img)

    # ----- refit-W reference (opt-in; tier0-log WAVE-1 caveat 1) -----------
    refit_block: dict | None = None
    refit_stats: dict | None = None
    if refit_w:
        # Refit (W, c) on the CHECKPOINT'S OWN latents over the SAME eval
        # images (CPU float32, pheq.analytic.fit_w conventions), then rerun
        # the photometric battery with the analytic operator built from the
        # refit. Compositions reuse the identical sampled params so the
        # ee / ee_refit_w blocks are comparable row for row.
        refit = fit_w(z.detach().float().cpu(), images.detach().float().cpu())
        refit_apply = _analytic_apply_fn(refit)
        refit_comp = _composition_stats(
            comp_params, images, z, refit_apply, decode_fn, batch
        )
        refit_comp["seed"] = int(seed)
        refit_block = {
            "grid": _battery_rows(grid_items, images, z, refit_apply, decode_fn, batch),
            "held_out": _battery_rows(held_items, images, z, refit_apply, decode_fn, batch),
            "compositions": refit_comp,
        }
        refit_stats = {
            "r2": float(refit.r2),
            "r2_per_channel": [float(v) for v in refit.r2_per_channel],
        }

    # ----- latent stats ----------------------------------------------------
    latent = {
        "mu_std_per_channel": [float(v) for v in z.std(dim=(0, 2, 3))],
        "effective_rank": float(effective_rank(z)),
        "spectral_slope": float(spectral_slope(z)),
        "high_freq_fraction": float(high_freq_fraction(z)),
    }

    # ----- collapse diagnostics (plan §3.5, risk R3) -----------------------
    kind = str(ckpt["operator_kind"])
    if kind in ("none", "analytic"):
        # The checkpoint operator IS the frozen analytic operator: the audit
        # coincides with the main battery, so reuse its rows.
        audit_rows = grid_rows
        identical = True
    else:
        analytic_apply = _analytic_apply_fn(_wfit_from_ckpt(ckpt))
        audit_rows = _battery_rows(grid_items, images, z, analytic_apply, decode_fn, batch)
        identical = False
    cross_operator = {
        "identical_to_main": identical,
        "per_factor": _per_factor_mean(audit_rows),
    }

    swapped_decoder: dict[str, Any] = {"skipped": True, "reason": None, "per_factor": None}
    if is_toy:
        swapped_decoder["reason"] = "toy checkpoint (no pretrained reference decoder)"
    else:
        try:
            from pheq.vae import load_sd_vae  # lazy: downloads/loads pretrained weights

            fresh = load_sd_vae(device=device)  # PRETRAINED, not the fine-tune
            for p in fresh.parameters():
                p.requires_grad_(False)
            swapped_rows = _battery_rows(
                grid_items, images, z, apply_op, fresh.decode_latents, batch
            )
            swapped_decoder = {
                "skipped": False,
                "reason": None,
                "per_factor": _per_factor_mean(swapped_rows),
            }
        except Exception as exc:  # diffusers/weights unavailable: degrade, don't die
            swapped_decoder["reason"] = f"pretrained SD decoder unavailable: {exc}"

    # ----- assemble --------------------------------------------------------
    path_repr = (
        str(ckpt_path)
        if isinstance(ckpt_path, (str, Path))
        else f"<in-memory:{ckpt['condition']}>"
    )
    result: dict[str, Any] = {
        "ckpt": {
            "path": path_repr,
            "condition": str(ckpt["condition"]),
            "operator_kind": kind,
            "step": int(ckpt["step"]),
        },
        "eval": {
            "n_images": n_eff,
            "image_size": image_size,
            "batch": int(batch),
            "device": str(device),
            "seed": int(seed),
        },
        "reconstruction": reconstruction,
        "ee": {"grid": grid_rows, "held_out": held_rows, "compositions": compositions},
        "ee_spatial": ee_spatial,
        "latent": latent,
        "collapse": {"cross_operator": cross_operator, "swapped_decoder": swapped_decoder},
    }
    if refit_block is not None:
        result["ee_refit_w"] = refit_block
        result["refit_w"] = refit_stats

    # Oracle-gap-closed block ONLY when BOTH references are provided (SPEC2).
    if b1_json is not None and oracle_csv is not None:
        with open(b1_json) as f:
            b1 = json.load(f)
        # tier0-log WAVE-1 caveat 1: prefer the b1 JSON's refit-W EE block
        # (fair reference — b1's stored-wfit EE conflates latent drift with
        # lack of equivariance); fall back to the stored-wfit "ee" block with
        # an explicit stale-reference marker.
        b1_ee = b1.get("ee_refit_w")
        stale_reference = b1_ee is None
        if stale_reference:
            b1_ee = b1["ee"]
        # W-convention for the NUMERATOR — which EE grid represents "this
        # checkpoint's operator":
        # - 'none' (b1, b2lite, c1proxy, b0): NO operator was trained; the
        #   stored-wfit grid conflates the checkpoint's own latent drift with
        #   lack of equivariance (caveat-1 bias — e.g. b1 against its own
        #   refit JSON must close exactly 0% of its own gap). Its fairest
        #   analytic steering is the REFIT grid; use ee_refit_w when computed.
        # - 'analytic' (p1): the stored wfit IS the shipped operator — training
        #   optimized the autoencoder to make exactly that (M, m) equivariant.
        #   Its stored-wfit EE is the artifact's true performance; substituting
        #   the refit grid would score a different operator than the one under
        #   evaluation (measured: p1 refit 8.4 vs shipped 4.6 ΔE00 brightness)
        #   and systematically UNDERSTATE analytic conditions.
        # - 'lie'/'conv': the co-trained operator is the object under
        #   evaluation; the main "ee" grid already measures it.
        ckpt_rows = grid_rows
        ckpt_ee_block = "ee"
        if not stale_reference and kind == "none" and refit_block is not None:
            ckpt_rows = refit_block["grid"]
            ckpt_ee_block = "ee_refit_w"
        result["oracle_gap_closed"] = {
            "metric": "ciede2000",
            "stale_w_reference": stale_reference,
            "ckpt_ee_block": ckpt_ee_block,
            "per_factor": gap_closed_fraction(
                _de00_map_from_rows(ckpt_rows),
                _de00_map_from_rows(b1_ee["grid"]),
                _de00_map_from_oracle_csv(oracle_csv),
            ),
        }

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write (tmp + os.replace): a SIGKILL mid-dump (e.g. the SLURM
    # hard limit after a preemption save) must never leave a TRUNCATED
    # eval.json — downstream jobs consume b1's eval.json as the gap-closed
    # reference and only check for its existence.
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp_path, out_path)
    return result


# ---------------------------------------------------------------------------
# CLI (mirrors the train_ae/probe conventions; printing allowed here)
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Per-checkpoint evaluation battery (SPEC2 eval_battery.py)."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ckpt", type=str, default=None, help="run checkpoint (.pt)")
    source.add_argument("--frozen-vae", choices=("sd", "toy"), default=None,
                        help="evaluate the PRETRAINED (never fine-tuned) VAE as "
                             "reference row b0 (requires --wfit; tier0-log W1 caveat 3)")
    parser.add_argument("--wfit", type=str, default=None,
                        help="wfit .pt for --frozen-vae (frozen-VAE-era W, c)")
    parser.add_argument("--image-size", type=int, default=None,
                        help="eval resolution for --frozen-vae "
                             "(default 256 for sd, 64 for toy)")
    parser.add_argument("--images", type=str, required=True, help="eval image directory")
    parser.add_argument("--out", type=str, required=True, help="output JSON path")
    parser.add_argument("--device", type=str, default="cpu", help="cpu | cuda | mps")
    parser.add_argument("--n-images", type=int, default=64)
    parser.add_argument("--rfid-n", type=int, default=0,
                        help="images for rFID (0 disables)")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--b1-json", type=str, default=None,
                        help="b1 reference eval JSON (gap-closed needs BOTH refs; "
                             "its ee_refit_w block is preferred when present)")
    parser.add_argument("--oracle-csv", type=str, default=None,
                        help="Tier-0 oracle probe CSV (gap-closed needs BOTH refs)")
    parser.add_argument("--no-lpips", action="store_true",
                        help="skip the LPIPS reconstruction metric")
    parser.add_argument("--refit-w", action="store_true",
                        help="add the ee_refit_w block: (W, c) refit on the "
                             "checkpoint's own latents over the eval images "
                             "(fair b1 reference, tier0-log W1 caveat 1)")
    args = parser.parse_args(argv)

    if args.frozen_vae is not None and args.wfit is None:
        parser.error("--frozen-vae requires --wfit")
    if args.frozen_vae is None and args.wfit is not None:
        parser.error("--wfit is only valid with --frozen-vae")
    if args.frozen_vae is None and args.image_size is not None:
        parser.error("--image-size is only valid with --frozen-vae "
                     "(checkpoints carry their own image_size)")

    target: str | dict = (
        args.ckpt
        if args.ckpt is not None
        else make_frozen_checkpoint(args.frozen_vae, args.wfit, image_size=args.image_size)
    )

    result = evaluate(
        target,
        args.images,
        args.out,
        device=args.device,
        n_images=args.n_images,
        rfid_n=args.rfid_n,
        b1_json=args.b1_json,
        oracle_csv=args.oracle_csv,
        batch=args.batch,
        seed=args.seed,
        use_lpips=not args.no_lpips,
        refit_w=args.refit_w,
    )

    ck, rec = result["ckpt"], result["reconstruction"]
    print(f"eval_battery: {ck['condition']} (operator={ck['operator_kind']}, "
          f"step={ck['step']}) on {result['eval']['n_images']} images")
    lp = "n/a" if rec["lpips"] is None else f"{rec['lpips']:.4f}"
    rf = "n/a" if rec["rfid"] is None else f"{rec['rfid']:.3f}"
    print(f"recon: PSNR {rec['psnr']:.2f} dB  L1 {rec['l1']:.4f}  LPIPS {lp}  rFID {rf}")
    if rec["lpips"] is None and not args.no_lpips:
        print("WARNING: LPIPS unavailable (package/weights missing) — reported as null")

    refit_means = (
        _per_factor_mean(result["ee_refit_w"]["grid"]) if "ee_refit_w" in result else None
    )
    header = f"{'factor':<12}{'mean ee_l2':>12}{'mean ee_de00':>14}"
    print(header + (f"{'refit ee_de00':>15}" if refit_means else ""))
    for factor, stats in _per_factor_mean(result["ee"]["grid"]).items():
        line = f"{factor:<12}{stats['ee_l2']:12.6f}{stats['ee_ciede2000']:14.4f}"
        if refit_means:
            line += f"{refit_means[factor]['ee_ciede2000']:15.4f}"
        print(line)
    comp = result["ee"]["compositions"]
    line = f"{'composed':<12}{comp['ee_l2']:12.6f}{comp['ee_ciede2000']:14.4f}"
    if refit_means:
        line += f"{result['ee_refit_w']['compositions']['ee_ciede2000']:15.4f}"
    print(line)
    if "refit_w" in result:
        rw = result["refit_w"]
        per_ch = ", ".join(f"{v:.3f}" for v in rw["r2_per_channel"])
        print(f"refit-w: R2 {rw['r2']:.4f} (per-channel {per_ch})")

    print(f"{'spatial op':<12}{'ee_l2':>12}{'ee_de00':>14}")
    for row in result["ee_spatial"]["ops"]:
        print(f"{row['op']:<12}{row['ee_l2']:12.6f}{row['ee_ciede2000']:14.4f}")
    sp_mean = result["ee_spatial"]["mean"]
    print(f"{'sp-mean':<12}{sp_mean['ee_l2']:12.6f}{sp_mean['ee_ciede2000']:14.4f}")

    lat = result["latent"]
    print(f"latent: eff_rank {lat['effective_rank']:.3f}  "
          f"slope {lat['spectral_slope']:.3f}  hff {lat['high_freq_fraction']:.4f}")
    sw = result["collapse"]["swapped_decoder"]
    if sw["skipped"]:
        print(f"swapped-decoder: skipped ({sw['reason']})")
    if "oracle_gap_closed" in result:
        gap_block = result["oracle_gap_closed"]
        if gap_block["stale_w_reference"]:
            print("WARNING: gap-closed b1 reference is the stored-wfit 'ee' block "
                  "(STALE for drifted b1 latents; rerun b1 with --refit-w)")
        elif (gap_block["ckpt_ee_block"] == "ee"
              and result["ckpt"]["operator_kind"] == "none"):
            print("WARNING: gap-closed numerator is this checkpoint's stored-wfit "
                  "'ee' block while the b1 reference is refit-W (mixed W "
                  "conventions; rerun this eval with --refit-w)")
        for factor, g in gap_block["per_factor"].items():
            gc = "n/a" if g["gap_closed"] is None else f"{g['gap_closed']:.3f}"
            print(f"gap-closed[{factor}]: {gc}  "
                  f"(b1 {g['ee_b1']:.3f} -> ckpt {g['ee_ckpt']:.3f}, "
                  f"oracle {g['ee_oracle']:.3f}, n={g['n_magnitudes']})")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
