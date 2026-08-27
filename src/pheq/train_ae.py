"""THE fine-tune loop: GAN-free equivariance fine-tuning of the autoencoder.

Implements SPEC2 "train_ae.py" (critical path), following plan §3.5 adapted
GAN-free (docs/plan-3month.md M1): ``loss = L1 + λ_lpips·LPIPS + λ_kl·KL``,
with the equivariance branch taken with probability ``p_eq`` per step.

Binding math (plan §3.4–3.5, SPEC2 care points):

- The KL is ALWAYS computed on the UNTRANSFORMED posterior ``(mu, sigma)``
  returned by ``encode_moments`` — never on pushed-forward moments.
- The 'analytic' operator acts as a push-forward on posterior MOMENTS
  (:func:`pheq.analytic.push_forward_posterior`); the sample is drawn AFTER
  the push-forward: ``z' = mu' + sigma'·eps``.
- The 'lie'/'conv' operators act on the REALIZED sample
  ``z' = g(mu + sigma·eps, phi)`` — the moment-push-forward shortcut is exact
  only for the affine analytic operator, so learned operators are applied to
  samples (documented spec decision).
- NO latent-matching loss anywhere: the equivariance branch is decoder-routed,
  ``L = ReconLoss(D(z'), τ(x))`` (EQ-VAE's collapse defense, plan §3.5.1).
  ``τ(x)`` is never encoded.

Engineering (SPEC2): monitors every 200 steps to ``out_dir/monitor.jsonl``
(step-0 snapshot; ALERT lines on >25% drift of channel std / effective rank),
checkpoint/resume via ``out_dir/ckpt_latest.pt`` every 500 steps, SIGTERM
handler that saves and exits 0 (SLURM ``--signal=B:TERM``), ``max_hours``
clean save+exit, eval hook (PSNR + L1 on a fixed val split) every 2000 steps.

Run-checkpoint format (SPEC2 design decisions — consumed by eval_battery,
cache_latents, train_dit)::

    {"vae": state_dict, "operator": state_dict | None, "operator_kind": str,
     "condition": str, "step": int, "wfit": {"W": (3,C), "c": (3,)},
     "config": dict}

plus resume-only extras (``optimizer``, ``torch_rng``, ``aug_rng``,
``cuda_rng``, ``monitor_ref``) that consumers ignore.

Spec ambiguities resolved (documented here, asserted in tests):

- SPEC2 prints ``wfit_path: str | None`` without a default after a defaulted
  arg (invalid Python); it gets ``= None``. When None and no resume
  checkpoint exists, ``(W, c)`` is fitted on the fly from up to 64 training
  images with the INITIAL encoder (the frozen-encoder calibration of plan
  §3.2); on resume the checkpoint's wfit is reused verbatim so the operator
  is identical across preemptions.
- The train() signature carries two trailing keyword-only args not printed
  in SPEC2 but required by its CLI: ``vae`` ('sd' | 'toy', the CLI's
  ``--vae``; 'toy' = ToyConvAE per SPEC2) and ``workers`` (loader workers).
- Deterministic toy encoders report ``sigma = 0`` (``encode_moments`` of
  SPEC pheq/vae.py); the KL of that degenerate posterior is infinite, so the
  KL input is floored at ``sigma >= 1e-4``. For the SD-VAE this is inactive
  in practice; for the toys it turns the KL into ``0.5·mean(mu²) + const``
  (gradients w.r.t. mu unchanged).
- Monitor scalars (L1/LPIPS/KL/latent stats/EE) are computed on a FIXED
  batch of up to 16 val images with the posterior mean (deterministic given
  the weights — comparable across steps and the basis of the step-0
  snapshot), not on running training-loss averages.
- The EE quick probe uses per-active-factor reference magnitudes
  (brightness/contrast 1.25, saturation 0.5, hue π/8 — all interior to the
  tightened SPEC2 ranges) with the FROZEN analytic operator from the run's
  wfit, CIEDE2000, averaged over factors; ``clip_fraction`` is the mean
  clipped fraction of those probe targets (same fixed transforms every
  monitor, hence comparable across steps and runs).
- ``lambda_lpips == 0`` skips LPIPS construction entirely (offline tests
  never touch VGG weights); otherwise ``ReconLoss(require_lpips=False)`` is
  used and a ``warnings.warn`` is emitted if LPIPS is unavailable (the CLI
  surfaces it loudly; library code never prints).
- Effective equivariance-constraint rate: the eq branch fires with prob
  ``p_eq``, but ``PhotoParams.sample`` gates each factor active at prob 0.5
  INTERNALLY, so inside the eq branch the sampled transform is the exact
  identity with prob ``0.5 ** len(active_factors)``. The NONTRIVIAL
  photometric constraint therefore fires at rate
  ``p_eq * (1 - 0.5 ** len(active_factors))`` — ~46.9% of steps for the
  4-factor conditions at p_eq = 0.5, but only 25% for single_* conditions
  (half their eq steps apply the identity). Per-factor dose parity across
  conditions still holds (every factor is nontrivially constrained at rate
  ``p_eq * 0.5`` in both full and single conditions); the any-factor rate
  is what differs. Documented, not "fixed": the shared activity gate keeps
  the factor-marginal transform distribution identical across conditions
  (revisit post-sprint alongside the per-batch parameter sharing).
- Completion sentinel: when training actually reaches ``config.steps`` the
  loop writes ``out_dir/DONE`` (the step count as text) next to the
  checkpoint. The SIGTERM-preemption and ``max_hours`` early saves do NOT
  write it — SLURM scripts gate post-training work (eval_battery) on this
  file, since train_ae deliberately exits 0 on those early-save paths.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import threading
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from pheq.analytic import WFit, analytic_operator, fit_w, push_forward_posterior
from pheq.color import PhotoParams, apply_affine, clipped_fraction
from pheq.conditions import FACTOR_FIELDS, TrainConfig, get_config
from pheq.conv_operator import ConvResidualOperator
from pheq.data import ImageFolderDataset, make_loader, train_val_split
from pheq.lie_operator import LieAffineOperator
from pheq.losses import ReconLoss, kl_loss
from pheq.metrics import ee_pix
from pheq.probes._common import factor_affine
from pheq.spatial import SpatialParams, apply_spatial
from pheq.spectral import SpectralMatchLoss, effective_rank, load_spectrum_stats, spectral_slope

__all__ = ["train", "main", "MONITOR_EVERY", "CKPT_EVERY", "EVAL_EVERY"]

#: Cadences (SPEC2 engineering requirements).
MONITOR_EVERY: int = 200
CKPT_EVERY: int = 500
EVAL_EVERY: int = 2000

#: Floor applied to sigma before the KL only (see module docstring).
_SIGMA_FLOOR: float = 1e-4

#: Relative drift on channel std / effective rank that triggers an ALERT line
#: (plan §3.5 defense 5: ±25%-of-snapshot thresholds).
_ALERT_REL_DRIFT: float = 0.25

#: Fixed EE-quick-probe magnitudes per factor (interior to the tightened
#: SPEC2 ranges beta, gamma ∈ [0.7, 1.3], sat ∈ [0.05, 1.5], hue ∈ [-π/4, π/4]).
_PROBE_MAGNITUDES: dict[str, float] = {
    "brightness": 1.25,
    "contrast": 1.25,
    "saturation": 0.5,
    "hue": math.pi / 8,
}


class _L1Recon(nn.Module):
    """L1-only stand-in for :class:`pheq.losses.ReconLoss` when λ_lpips = 0.

    Constructing ``ReconLoss`` always attempts to build the LPIPS VGG (its
    lazy import lives in ``__init__``); with ``lambda_lpips == 0`` that work
    (and any weight download) is pointless, so this shim provides the same
    ``forward`` / ``lpips_term`` / ``lpips_active`` surface with LPIPS
    permanently inactive. Offline tests run through this path.
    """

    lpips_active: bool = False

    def lpips_term(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return pred.new_zeros(())

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)


def _append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON line to ``path`` (monitor.jsonl convention)."""
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _restrict_params(params: PhotoParams, active: tuple[str, ...]) -> PhotoParams:
    """Reset factors outside ``active_factors`` to their identity values.

    ``PhotoParams.sample`` draws all four factors; single-factor conditions
    (SPEC2 conditions.py) restrict the pool by forcing inactive factors to
    the group identity (beta = gamma = sat = 1, hue = 0).
    """
    vals = {"beta": params.beta, "gamma": params.gamma, "sat": params.sat, "hue": params.hue}
    for factor, field in FACTOR_FIELDS.items():
        if factor not in active:
            vals[field] = 0.0 if field == "hue" else 1.0
    return PhotoParams(**vals)


def _sample_photo(gen: torch.Generator, cfg: TrainConfig, n: int = 1) -> list[PhotoParams]:
    """Sample ``n`` PhotoParams from the config's ranges/active factors."""
    return [
        _restrict_params(p, cfg.active_factors)
        for p in PhotoParams.sample(gen, n, ranges=cfg.photo_ranges)
    ]


def _build_probes(
    wfit: WFit, active: tuple[str, ...]
) -> list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Fixed (factor, A, b, M, m) probe transforms for the EE quick probe.

    One reference magnitude per ACTIVE factor (:data:`_PROBE_MAGNITUDES`);
    (M, m) is the FROZEN analytic operator (K = "I") built from the run's
    wfit — the collapse instrumentation of plan §3.5 (cross-operator audit
    flavor: learned-operator conditions are still probed analytically).
    """
    probes = []
    for factor in active:
        a_mat, b_vec = factor_affine(factor, _PROBE_MAGNITUDES[factor])
        m_mat, m_vec = analytic_operator(wfit, a_mat, b_vec, K="I")
        probes.append((factor, a_mat, b_vec, m_mat, m_vec))
    return probes


@torch.no_grad()
def _monitor_stats(
    encode: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    decode: Callable[[torch.Tensor], torch.Tensor],
    recon: nn.Module,
    x_mon: torch.Tensor,
    probes: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] | None,
) -> dict:
    """Deterministic monitor scalars on the fixed val batch (SPEC2 monitors).

    Keys: l1, lpips, kl, mu_std (per channel), eff_rank, spectral_slope
    (None when the latent is too small for the [2, h/2] fit), and — photo
    conditions only — clip_fraction and ee_quick (else None).
    """
    mu, sigma = encode(x_mon)
    rec = decode(mu)
    stats: dict = {
        "l1": float(F.l1_loss(rec, x_mon).item()),
        "lpips": float(recon.lpips_term(rec, x_mon).item()),
        "kl": float(kl_loss(mu, sigma.clamp_min(_SIGMA_FLOOR)).item()),
        "mu_std": [float(v) for v in mu.std(dim=(0, 2, 3)).tolist()],
        "eff_rank": float(effective_rank(mu)),
        "spectral_slope": (
            float(spectral_slope(mu)) if min(mu.shape[-2:]) // 2 >= 3 else None
        ),
        "clip_fraction": None,
        "ee_quick": None,
    }
    if probes:
        ees, clips = [], []
        for _, a_mat, b_vec, m_mat, m_vec in probes:
            x_aug = apply_affine(x_mon, a_mat, b_vec, clip=False)
            m_dev = m_mat.to(mu.device)
            v_dev = m_vec.to(mu.device)
            ees.append(float(ee_pix(decode, mu, m_dev, v_dev, x_aug, metric="ciede2000").item()))
            clips.append(clipped_fraction(x_mon, a_mat, b_vec))
        stats["ee_quick"] = sum(ees) / len(ees)
        stats["clip_fraction"] = sum(clips) / len(clips)
    return stats


def _drift_alert(cur: dict, ref: dict, rel: float = _ALERT_REL_DRIFT) -> list[str]:
    """Reasons for an ALERT line: >``rel`` relative drift of channel std or
    effective rank vs the step-0 snapshot (plan §3.5 defense 5)."""
    reasons = []
    for i, (c, r) in enumerate(zip(cur["mu_std"], ref["mu_std"])):
        if abs(c - r) > rel * max(abs(r), 1e-8):
            reasons.append(f"mu_std[{i}] drifted {c:.4g} vs snapshot {r:.4g}")
    c, r = cur["eff_rank"], ref["eff_rank"]
    if abs(c - r) > rel * max(abs(r), 1e-8):
        reasons.append(f"eff_rank drifted {c:.4g} vs snapshot {r:.4g}")
    return reasons


def _build_vae(vae: str, device: torch.device, seed: int) -> nn.Module:
    """Construct the trainable autoencoder ('sd' | 'toy' = ToyConvAE, SPEC2)."""
    if vae == "sd":
        from pheq.vae import load_sd_vae  # lazy diffusers import (SPEC pheq/vae.py)

        model = load_sd_vae(device=str(device))
    elif vae == "toy":
        from pheq.vae import ToyConvAE

        model = ToyConvAE(seed=seed).to(device)
    else:
        raise ValueError(f"vae must be 'sd' or 'toy', got {vae!r}")
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()
    return model


def _load_wfit(path: str) -> WFit:
    """Load a wfit payload saved by ``pheq.probes.fit_w`` (or any {"W","c"} dict)."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return WFit(
        W=payload["W"].float(),
        c=payload["c"].float(),
        r2=float(payload.get("r2", float("nan"))),
        r2_per_channel=payload.get("r2_per_channel", torch.full((3,), float("nan"))),
    )


@torch.no_grad()
def _fit_wfit_on_the_fly(
    encode: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    train_set: torch.utils.data.Dataset,
    device: torch.device,
    n_fit: int = 64,
    chunk: int = 8,
) -> WFit:
    """Fit (W, c) on up to ``n_fit`` training images with the initial encoder.

    The frozen-encoder calibration of plan §3.2 — used when no ``wfit_path``
    is supplied (the offline ``--vae toy`` path).
    """
    n = min(n_fit, len(train_set))  # type: ignore[arg-type]
    imgs = torch.stack([train_set[i][0] for i in range(n)]).to(device)
    mus = [encode(imgs[i : i + chunk])[0] for i in range(0, n, chunk)]
    return fit_w(torch.cat(mus).cpu().float(), imgs.cpu().float())


def _save_ckpt(
    path: Path,
    vae: nn.Module,
    operator: nn.Module | None,
    cfg: TrainConfig,
    step: int,
    wfit: WFit,
    optimizer: torch.optim.Optimizer,
    monitor_ref: dict | None,
    aug_gen: torch.Generator,
    vae_arch: str = "sd",
) -> Path:
    """Atomically write the run checkpoint (SPEC2 format + resume extras).

    ``config`` is ``asdict(cfg)`` plus a ``"vae"`` key ('sd' | 'toy'): the
    downstream resolver (``pheq.fid._resolve_vae``, shared by cache_latents /
    eval_battery / train_dit) reads ``config["vae"]`` to pick the checkpoint
    architecture and defaults to 'sd' when absent — omitting it would make
    every toy run checkpoint unloadable by its consumers.
    """
    payload = {
        # --- the consumer contract (eval_battery / cache_latents / train_dit) ---
        "vae": {k: v.detach().cpu() for k, v in vae.state_dict().items()},
        "operator": (
            {k: v.detach().cpu() for k, v in operator.state_dict().items()}
            if operator is not None
            else None
        ),
        "operator_kind": cfg.operator_kind,
        "condition": cfg.condition,
        "step": int(step),
        "wfit": {"W": wfit.W.detach().cpu(), "c": wfit.c.detach().cpu()},
        "config": {**asdict(cfg), "vae": str(vae_arch)},
        # --- resume-only extras (consumers ignore) ---
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "aug_rng": aug_gen.get_state(),
        "monitor_ref": monitor_ref,
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic on POSIX: never leaves a torn ckpt_latest
    return path


def train(
    config: TrainConfig,
    data_dir: str,
    out_dir: str,
    device: str | torch.device,
    resume: bool = True,
    wfit_path: str | None = None,
    max_hours: float | None = None,
    *,
    vae: str = "sd",
    workers: int = 4,
) -> Path:
    """Run one fine-tune condition; returns the final checkpoint path.

    Per step (plan §3.5 adapted GAN-free; SPEC2 train_ae.py):

    1. Batch ``x``; ``mu, sigma = encode_moments(x)``; KL on the
       UNTRANSFORMED posterior (floored, see module docstring).
    2. With prob ``1 − p_eq`` (or when the condition has no τ): standard
       branch ``loss = ReconLoss(D(mu + sigma·eps), x) + λ_kl·KL``.
    3. Else equivariance branch (decoder-routed, NO latent matching):
       photometric τ sampled ONE params per batch (deliberate
       simplification — simpler operator batching; revisit post-sprint),
       target ``x_t = τ(x)`` pre-clip; 'analytic' pushes MOMENTS
       (``z' = mu' + sigma'·eps``), 'lie'/'conv' act on the REALIZED sample
       (``z' = g(mu + sigma·eps, φ)``, co-trained in the joint optimizer);
       spatial τ (if any) applied identically to image (f-aligned) and
       latent; ``loss = ReconLoss(D(z'), x_t) + λ_kl·KL``
       (+ ``λ_comp · composition_loss`` for 'lie', fresh params pair,
       z detached inside).
    4. c1proxy: no branch split; adds ``λ_spectral · SpectralMatchLoss(mu)``
       to the standard loss every step.
    5. AdamW (betas 0.9/0.999; wd 0.01 on VAE, 0 on operator; same lr),
       grad clip 1.0, bf16 autocast when CUDA.

    Args:
        config: a :class:`pheq.conditions.TrainConfig` (via ``get_config``).
        data_dir: image folder (recursive; ``pheq.data.ImageFolderDataset``).
        out_dir: run directory (``ckpt_latest.pt``, ``monitor.jsonl``).
        device: torch device (string or ``torch.device``).
        resume: auto-load ``out_dir/ckpt_latest.pt`` when present.
        wfit_path: optional ``wfit.pt`` (``pheq.probes.fit_w`` payload);
            fitted on the fly when None (see module docstring).
        max_hours: clean save+return before a SLURM wall clock.
        vae: 'sd' (pretrained SD-VAE) or 'toy' (ToyConvAE, offline).
        workers: DataLoader workers (0 = main process; tests use 0).

    Returns:
        Path to the final run checkpoint (``out_dir/ckpt_latest.pt``).
    """
    cfg = config
    t0 = time.monotonic()
    device = torch.device(device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "ckpt_latest.pt"
    monitor_path = out / "monitor.jsonl"
    done_path = out / "DONE"  # written ONLY when config.steps is reached

    if cfg.condition == "c1proxy" and cfg.spectral_stats is None:
        raise ValueError("c1proxy requires config.spectral_stats (spectrum-stats JSON)")

    torch.manual_seed(cfg.seed)
    # Dedicated CPU generator for augmentation/branch sampling; its state is
    # checkpointed. The split/loader generators are re-seeded from cfg.seed so
    # the val split is IDENTICAL across runs and resumes (SPEC2 fixed val set).
    aug_gen = torch.Generator().manual_seed(cfg.seed + 1)

    dataset = ImageFolderDataset(data_dir, size=cfg.image_size)
    val_n = min(64, max(1, len(dataset) // 5))
    split_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = train_val_split(dataset, val_n, split_gen)
    if len(train_set) < cfg.batch_size:
        raise ValueError(
            f"train split has {len(train_set)} images < batch_size {cfg.batch_size}"
        )

    model = _build_vae(vae, device, cfg.seed)
    encode = model.encode_moments
    decode = model.decode_latents

    # ---- resume state (loaded early: the wfit must be the ORIGINAL run's) ----
    ck = None
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ck["condition"] != cfg.condition:
            raise ValueError(
                f"checkpoint at {ckpt_path} is condition {ck['condition']!r}, "
                f"not {cfg.condition!r}"
            )
        if ck["operator_kind"] != cfg.operator_kind:
            raise ValueError(
                f"checkpoint operator_kind {ck['operator_kind']!r} != config "
                f"{cfg.operator_kind!r}"
            )

    if ck is not None:
        wfit = WFit(
            W=ck["wfit"]["W"].float(),
            c=ck["wfit"]["c"].float(),
            r2=float("nan"),
            r2_per_channel=torch.full((3,), float("nan")),
        )
    elif wfit_path is not None:
        wfit = _load_wfit(wfit_path)
    else:
        wfit = _fit_wfit_on_the_fly(encode, train_set, device)
    wfit_dev = WFit(
        W=wfit.W.to(device), c=wfit.c.to(device), r2=wfit.r2,
        r2_per_channel=wfit.r2_per_channel,
    )

    # ---- operator (co-trained for 'lie'/'conv'; None for 'none'/'analytic') ----
    with torch.no_grad():
        probe_img, _ = train_set[0]
        probe_mu, _ = encode(probe_img[None].to(device))
    channels = probe_mu.shape[1]
    f_img = probe_img.shape[-1] // probe_mu.shape[-1]  # AE downsampling factor

    operator: nn.Module | None = None
    if cfg.operator_kind == "lie":
        operator = LieAffineOperator(channels=channels)
        if cfg.analytic_init:
            operator.init_from_analytic(wfit)
        operator = operator.to(device)
    elif cfg.operator_kind == "conv":
        operator = ConvResidualOperator(wfit).to(device)
    elif cfg.operator_kind not in ("none", "analytic"):
        raise ValueError(f"unknown operator_kind {cfg.operator_kind!r}")

    # ---- losses ----
    if cfg.lambda_lpips > 0:
        recon: nn.Module = ReconLoss(lambda_lpips=cfg.lambda_lpips, require_lpips=False)
        if not recon.lpips_active:  # type: ignore[union-attr]
            warnings.warn(
                "LPIPS unavailable (package or VGG weights missing); training "
                "with L1-only reconstruction. CLIs should surface this loudly.",
                stacklevel=2,
            )
    else:
        recon = _L1Recon()
    recon = recon.to(device)

    spectral_loss: SpectralMatchLoss | None = None
    if cfg.condition == "c1proxy":
        spectral_loss = SpectralMatchLoss(*load_spectrum_stats(cfg.spectral_stats)).to(device)

    # ---- optimizer: AdamW, wd 0.01 on VAE only, operator wd 0, same lr ----
    groups = [{"params": list(model.parameters()), "weight_decay": 0.01}]
    if operator is not None:
        groups.append({"params": list(operator.parameters()), "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.999))
    clip_params = [p for g in groups for p in g["params"]]

    # ---- restore resume state ----
    start_step = 0
    monitor_ref: dict | None = None
    if ck is not None:
        model.load_state_dict(ck["vae"])
        if operator is not None and ck["operator"] is not None:
            operator.load_state_dict(ck["operator"])
        optimizer.load_state_dict(ck["optimizer"])
        torch.set_rng_state(ck["torch_rng"])
        if ck.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        aug_gen.set_state(ck["aug_rng"])
        start_step = int(ck["step"])
        monitor_ref = ck.get("monitor_ref")

    # ---- fixed val batches (deterministic split => stable across resumes) ----
    x_mon = torch.stack(
        [val_set[i][0] for i in range(min(16, len(val_set)))]
    ).to(device)
    x_eval = torch.stack(
        [val_set[i][0] for i in range(min(64, len(val_set)))]
    ).to(device)
    probes = _build_probes(wfit_dev, cfg.active_factors) if cfg.tau_photo else None

    # ---- step-0 snapshot (fresh runs only; resumes reuse the saved ref) ----
    if start_step == 0:
        monitor_ref = _monitor_stats(encode, decode, recon, x_mon, probes)
        _append_jsonl(
            monitor_path,
            {"level": "monitor", "step": 0, "lpips_active": bool(recon.lpips_active), **monitor_ref},
        )
    elif monitor_ref is None:  # legacy ckpt without a snapshot: re-baseline
        monitor_ref = _monitor_stats(encode, decode, recon, x_mon, probes)

    def save(step: int) -> Path:
        return _save_ckpt(
            ckpt_path, model, operator, cfg, step, wfit, optimizer, monitor_ref,
            aug_gen, vae_arch=vae,
        )

    if start_step >= cfg.steps:
        if not ckpt_path.exists():
            save(start_step)
        done_path.write_text(f"{start_step}\n")  # budget already met = complete
        return ckpt_path

    if ck is not None:
        # Segment marker for monitor.jsonl consumers: after a HARD crash
        # (SIGKILL/OOM) the file may hold lines from the dead run with steps
        # BEYOND the checkpoint we resume from; readers should dedupe by
        # keeping the last occurrence after the final resume marker.
        _append_jsonl(monitor_path, {"level": "resume", "step": start_step})

    # ---- SIGTERM: save and exit 0 (SLURM preemption, SPEC2) ----
    stop_requested = False
    prev_handler = None

    def _on_sigterm(signum, frame):  # noqa: ANN001 - signal API
        nonlocal stop_requested
        stop_requested = True

    in_main_thread = threading.current_thread() is threading.main_thread()
    if in_main_thread:
        prev_handler = signal.signal(signal.SIGTERM, _on_sigterm)

    loader_gen = torch.Generator().manual_seed(cfg.seed + 2 + start_step)
    loader = make_loader(
        train_set, cfg.batch_size, gen=loader_gen, workers=workers, shuffle=True
    )

    def batches():
        while True:
            yield from loader

    has_tau = cfg.tau_photo or cfg.tau_spatial
    use_amp = device.type == "cuda"
    step = start_step
    try:
        batch_iter = batches()
        while step < cfg.steps:
            if stop_requested:
                save(step)
                raise SystemExit(0)
            if max_hours is not None and (time.monotonic() - t0) / 3600.0 >= max_hours:
                save(step)
                return ckpt_path

            x, _ = next(batch_iter)
            x = x.to(device)
            take_eq = has_tau and bool(
                torch.rand(1, generator=aug_gen).item() < cfg.p_eq
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                mu, sigma = encode(x)
                # KL ALWAYS on the untransformed posterior (plan §3.4).
                kl = kl_loss(mu, sigma.clamp_min(_SIGMA_FLOOR))

                if not take_eq:
                    # Standard branch (also every step for b1 / c1proxy).
                    z = mu + sigma * torch.randn_like(mu)
                    loss = recon(decode(z), x) + cfg.lambda_kl * kl
                    if spectral_loss is not None:
                        loss = loss + cfg.lambda_spectral * spectral_loss(mu)
                else:
                    x_t = x
                    comp_term = None
                    if cfg.tau_photo:
                        params = _sample_photo(aug_gen, cfg)[0]
                        a_mat, b_vec = params.affine()
                        # Target is PRE-CLIP (plan §3.1): clip=False.
                        x_t = apply_affine(x_t, a_mat, b_vec, clip=False)
                        if cfg.operator_kind == "analytic":
                            m_mat, m_vec = analytic_operator(wfit_dev, a_mat, b_vec, K="I")
                            mu_p, sigma_p = push_forward_posterior(mu, sigma, m_mat, m_vec)
                            z_t = mu_p + sigma_p * torch.randn_like(mu_p)
                        else:  # 'lie' / 'conv': operator on the REALIZED sample
                            z = mu + sigma * torch.randn_like(mu)
                            phi = params.phi().to(device)
                            z_t = operator(z, phi)
                            if cfg.operator_kind == "lie" and cfg.lambda_comp > 0:
                                pa, pb = _sample_photo(aug_gen, cfg, n=2)
                                comp_term = operator.composition_loss(
                                    z.detach(), pa.phi().to(device), pb.phi().to(device)
                                )
                    else:
                        z_t = mu + sigma * torch.randn_like(mu)
                    if cfg.tau_spatial:
                        sp = SpatialParams.sample(aug_gen)
                        # SAME op on image (f-aligned) and latent (plan §2).
                        x_t = apply_spatial(x_t, sp, antialias=True, f=f_img)
                        z_t = apply_spatial(z_t, sp, antialias=True, f=1)
                    loss = recon(decode(z_t), x_t) + cfg.lambda_kl * kl
                    if comp_term is not None:
                        loss = loss + cfg.lambda_comp * comp_term

            loss.backward()
            torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
            optimizer.step()
            step += 1

            if step % MONITOR_EVERY == 0:
                stats = _monitor_stats(encode, decode, recon, x_mon, probes)
                _append_jsonl(monitor_path, {"level": "monitor", "step": step, **stats})
                assert monitor_ref is not None
                reasons = _drift_alert(stats, monitor_ref)
                if reasons:
                    _append_jsonl(
                        monitor_path,
                        {
                            "level": "alert",
                            "step": step,
                            "reasons": reasons,
                            "mu_std": stats["mu_std"],
                            "eff_rank": stats["eff_rank"],
                            "ref_mu_std": monitor_ref["mu_std"],
                            "ref_eff_rank": monitor_ref["eff_rank"],
                        },
                    )
            if step % EVAL_EVERY == 0:
                _append_jsonl(
                    monitor_path,
                    {"level": "eval", "step": step, **_eval_stats(encode, decode, x_eval)},
                )
            if step % CKPT_EVERY == 0:
                save(step)

        save(step)
        done_path.write_text(f"{step}\n")  # full budget trained: completion sentinel
        return ckpt_path
    finally:
        if in_main_thread:
            signal.signal(signal.SIGTERM, prev_handler)


@torch.no_grad()
def _eval_stats(
    encode: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    decode: Callable[[torch.Tensor], torch.Tensor],
    x_eval: torch.Tensor,
    chunk: int = 8,
) -> dict:
    """PSNR + L1 of the mean-path reconstruction on the fixed val split."""
    l1_sum, mse_sum, n = 0.0, 0.0, 0
    for i in range(0, x_eval.shape[0], chunk):
        x = x_eval[i : i + chunk]
        rec = decode(encode(x)[0])
        l1_sum += float(F.l1_loss(rec, x, reduction="sum").item())
        mse_sum += float(F.mse_loss(rec, x, reduction="sum").item())
        n += x.numel()
    mse = max(mse_sum / n, 1e-12)
    return {"psnr": float(10.0 * math.log10(1.0 / mse)), "l1": l1_sum / n}


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m pheq.train_ae --condition p2_lie --data DIR --out DIR ...``

    SPEC2 flags: ``[--vae sd|toy] [--steps N] [--wfit PATH] [--device]
    [--seed] [--max-hours H] [--spectral-stats PATH]`` (+ convenience
    overrides ``--image-size``, ``--batch-size``, ``--workers``,
    ``--no-resume``). ``--vae toy`` runs a full tiny loop offline in <60 s.
    """
    parser = argparse.ArgumentParser(
        description="GAN-free equivariance fine-tune of the autoencoder (plan §3.5)."
    )
    parser.add_argument("--condition", type=str, required=True,
                        help="registry name from pheq.conditions.CONDITIONS")
    parser.add_argument("--data", type=str, required=True, help="image directory")
    parser.add_argument("--out", type=str, required=True, help="run/output directory")
    parser.add_argument("--vae", choices=("sd", "toy"), default="sd",
                        help="'sd' = pretrained SD-VAE; 'toy' = ToyConvAE (offline)")
    parser.add_argument("--steps", type=int, default=None, help="override config.steps")
    parser.add_argument("--wfit", type=str, default=None,
                        help="wfit.pt from pheq.probes.fit_w (fitted on the fly if omitted)")
    parser.add_argument("--device", type=str, default=None,
                        help="cpu | cuda | mps (default: auto-detect)")
    parser.add_argument("--seed", type=int, default=None, help="override config.seed")
    parser.add_argument("--image-size", type=int, default=None,
                        help="override config.image_size (e.g. 32 for --vae toy smoke runs)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="override config.batch_size")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="clean save+exit before the SLURM wall clock")
    parser.add_argument("--spectral-stats", type=str, default=None,
                        help="spectrum-stats JSON (required for c1proxy)")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore an existing ckpt_latest.pt")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    args = parser.parse_args(argv)

    overrides: dict = {}
    if args.steps is not None:
        overrides["steps"] = args.steps
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.image_size is not None:
        overrides["image_size"] = args.image_size
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.spectral_stats is not None:
        overrides["spectral_stats"] = args.spectral_stats
    cfg = get_config(args.condition, **overrides)

    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(
        f"train_ae: condition={cfg.condition} operator={cfg.operator_kind} "
        f"vae={args.vae} steps={cfg.steps} device={device} out={args.out}"
    )
    if cfg.lambda_lpips == 0:
        print("train_ae: lambda_lpips=0 — LPIPS disabled, L1-only reconstruction")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ckpt = train(
            cfg,
            data_dir=args.data,
            out_dir=args.out,
            device=device,
            resume=not args.no_resume,
            wfit_path=args.wfit,
            max_hours=args.max_hours,
            vae=args.vae,
            workers=args.workers,
        )
        for w in caught:
            if "LPIPS unavailable" in str(w.message):
                print(f"train_ae: WARNING — {w.message}")  # loud CLI log (SPEC2)
    print(f"train_ae: done — checkpoint at {ckpt}")


if __name__ == "__main__":
    main()
