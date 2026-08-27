"""Experiment registry — the single source of truth for v2 training conditions.

Implements SPEC2 "conditions.py"; the condition matrix follows
docs/plan-3month.md M2 and docs/research-plan.md §5. Photometric DEFAULT
ranges are TIGHTENED vs v1 per SPEC2's binding design decisions (Tier-0
measured 20–27% clipped pixels at beta/gamma = 1.4 on real images):
``beta, gamma ∈ [0.7, 1.3]``; ``sat ∈ [0.05, 1.5]`` and ``hue ∈ [−π/4, π/4]``
unchanged. Conditions pass these ranges EXPLICITLY to
``PhotoParams.sample(..., ranges=...)`` — v1 defaults in color.py are
untouched apart from the optional ``ranges`` argument.

Validation of a config against the environment (files exist, device, etc.)
happens in train_ae; this module only owns the shape of the registry, plus the
one registry-level demand SPEC2 states: ``c1proxy`` requires ``spectral_stats``
at :func:`get_config` time.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, fields, replace

__all__ = [
    "DEFAULT_PHOTO_RANGES",
    "FACTOR_FIELDS",
    "TrainConfig",
    "CONDITIONS",
    "get_config",
]

#: Tightened default photometric sampling ranges (SPEC2 design decisions).
#: Keys are the ``PhotoParams`` FIELD names — this dict is passed verbatim as
#: ``PhotoParams.sample(..., ranges=photo_ranges)``.
DEFAULT_PHOTO_RANGES: dict[str, tuple[float, float]] = {
    "beta": (0.7, 1.3),
    "gamma": (0.7, 1.3),
    "sat": (0.05, 1.5),
    "hue": (-math.pi / 4, math.pi / 4),
}

#: Map from ``active_factors`` names (plan §3.1 factor names) to the
#: corresponding ``PhotoParams`` field / ``photo_ranges`` key.
FACTOR_FIELDS: dict[str, str] = {
    "brightness": "beta",
    "contrast": "gamma",
    "saturation": "sat",
    "hue": "hue",
}

_ALL_FACTORS: tuple[str, ...] = ("brightness", "contrast", "saturation", "hue")


@dataclass
class TrainConfig:
    """One fine-tune condition (SPEC2 "conditions.py"; plan-3month.md M2).

    Fields:
        condition: registry name (also stamped into run checkpoints).
        operator_kind: 'none' | 'analytic' | 'lie' | 'conv' (plan §3.2–3.3).
        tau_photo: photometric equivariance branch active (plan §3.5).
        tau_spatial: EQ-VAE-style spatial branch active (B2-lite).
        photo_ranges: sampling ranges for ``PhotoParams.sample`` — TIGHTENED
            defaults (:data:`DEFAULT_PHOTO_RANGES`), keyed by PhotoParams
            field names.
        active_factors: which photometric factors are sampled (single-factor
            conditions restrict this; names per :data:`FACTOR_FIELDS`).
        p_eq: probability of the equivariance branch per step. NOTE:
            ``PhotoParams.sample`` gates each factor active at prob 0.5
            internally, so the sampled transform inside the eq branch is the
            exact identity with prob ``0.5 ** len(active_factors)`` — the
            NONTRIVIAL photometric constraint fires at effective rate
            ``p_eq * (1 - 0.5 ** len(active_factors))`` (~46.9% of steps for
            4-factor conditions at p_eq = 0.5, 25% for single_* conditions).
            Per-factor dose parity (``p_eq * 0.5`` per factor) holds across
            all conditions; see the pheq.train_ae module docstring.
        lambda_comp: composition-loss weight (lie operator only).
        analytic_init: init the lie operator from the analytic fit (lie only).
        spectral_stats: path to spectrum-stats JSON (c1proxy only; REQUIRED
            at :func:`get_config` time for c1proxy).
        lambda_spectral: SpectralMatchLoss weight (c1proxy).
        lr / batch_size / steps / image_size: optimization + data settings.
        lambda_kl / lambda_lpips: loss weights (GAN-free recipe, SPEC2).
        seed: RNG seed for the run.
    """

    condition: str
    operator_kind: str  # 'none' | 'analytic' | 'lie' | 'conv'
    tau_photo: bool
    tau_spatial: bool
    photo_ranges: dict = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_PHOTO_RANGES)
    )
    active_factors: tuple[str, ...] = _ALL_FACTORS
    p_eq: float = 0.5
    lambda_comp: float = 0.1  # lie only
    analytic_init: bool = True  # lie only
    spectral_stats: str | None = None  # c1proxy only
    lambda_spectral: float = 1.0
    lr: float = 1e-4
    batch_size: int = 8
    steps: int = 20_000
    image_size: int = 256
    lambda_kl: float = 1e-6
    lambda_lpips: float = 1.0
    seed: int = 0


#: The experiment registry — exactly the 12 SPEC2 conditions (plan-3month M2).
CONDITIONS: dict[str, TrainConfig] = {
    # Baselines
    "b1": TrainConfig(  # no equivariance at all
        condition="b1", operator_kind="none", tau_photo=False, tau_spatial=False
    ),
    "b2lite": TrainConfig(  # spatial only; spatial ops ARE the latent op
        condition="b2lite", operator_kind="none", tau_photo=False, tau_spatial=True
    ),
    # Photometric conditions
    "p1_analytic": TrainConfig(
        condition="p1_analytic",
        operator_kind="analytic",
        tau_photo=True,
        tau_spatial=False,
    ),
    "p2_lie": TrainConfig(
        condition="p2_lie", operator_kind="lie", tau_photo=True, tau_spatial=False
    ),
    "p2_nocomp": TrainConfig(  # p2 ablation: no composition loss
        condition="p2_nocomp",
        operator_kind="lie",
        tau_photo=True,
        tau_spatial=False,
        lambda_comp=0.0,
    ),
    "p2_noinit": TrainConfig(  # p2 ablation: no analytic init
        condition="p2_noinit",
        operator_kind="lie",
        tau_photo=True,
        tau_spatial=False,
        analytic_init=False,
    ),
    "p3_conv": TrainConfig(
        condition="p3_conv", operator_kind="conv", tau_photo=True, tau_spatial=False
    ),
    # Spectral-matched control (RQ5 mediation; plan §5 C1)
    "c1proxy": TrainConfig(  # no tau; spectral_stats REQUIRED at runtime
        condition="c1proxy", operator_kind="none", tau_photo=False, tau_spatial=False
    ),
    # Single-factor conditions (lie operator, one active factor)
    "single_brightness": TrainConfig(
        condition="single_brightness",
        operator_kind="lie",
        tau_photo=True,
        tau_spatial=False,
        active_factors=("brightness",),
    ),
    "single_contrast": TrainConfig(
        condition="single_contrast",
        operator_kind="lie",
        tau_photo=True,
        tau_spatial=False,
        active_factors=("contrast",),
    ),
    "single_saturation": TrainConfig(
        condition="single_saturation",
        operator_kind="lie",
        tau_photo=True,
        tau_spatial=False,
        active_factors=("saturation",),
    ),
    "single_hue": TrainConfig(
        condition="single_hue",
        operator_kind="lie",
        tau_photo=True,
        tau_spatial=False,
        active_factors=("hue",),
    ),
}


def get_config(name: str, **overrides) -> TrainConfig:
    """Fetch a fresh (deep-copied) registry config with overrides applied.

    Unknown ``name`` → ValueError listing valid condition names; unknown
    override field → ValueError listing valid fields. SPEC2 registry-level
    demand: ``c1proxy`` raises unless ``spectral_stats`` is provided (either
    as an override here or already set on the returned config); all deeper
    validation lives in train_ae. The returned config owns its own
    ``photo_ranges`` dict — mutating it never touches the registry.
    """
    if name not in CONDITIONS:
        raise ValueError(
            f"unknown condition {name!r}; valid conditions: {sorted(CONDITIONS)}"
        )
    valid_fields = {f.name for f in fields(TrainConfig)}
    unknown = set(overrides) - valid_fields
    if unknown:
        raise ValueError(
            f"unknown TrainConfig field(s) {sorted(unknown)}; "
            f"valid fields: {sorted(valid_fields)}"
        )
    cfg = replace(copy.deepcopy(CONDITIONS[name]), **overrides)
    if cfg.condition == "c1proxy" and cfg.spectral_stats is None:
        raise ValueError(
            "condition 'c1proxy' requires spectral_stats "
            "(path to a spectrum-stats JSON from pheq.spectral.save_spectrum_stats); "
            "pass get_config('c1proxy', spectral_stats=...)"
        )
    return cfg
