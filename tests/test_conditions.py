"""Tests for pheq.conditions (SPEC2 "conditions.py") + the permitted v1 edit
(PhotoParams.sample ranges argument). Offline, CPU, deterministic."""

from __future__ import annotations

import math

import pytest
import torch

from pheq.color import PhotoParams
from pheq.conditions import (
    CONDITIONS,
    DEFAULT_PHOTO_RANGES,
    FACTOR_FIELDS,
    TrainConfig,
    get_config,
)

EXPECTED_NAMES = {
    "b1",
    "b2lite",
    "p1_analytic",
    "p2_lie",
    "p2_nocomp",
    "p2_noinit",
    "p3_conv",
    "c1proxy",
    "single_brightness",
    "single_contrast",
    "single_saturation",
    "single_hue",
}


# ---------------------------------------------------------------- registry


def test_registry_contains_exactly_the_12_conditions() -> None:
    assert set(CONDITIONS) == EXPECTED_NAMES
    assert len(CONDITIONS) == 12
    for name, cfg in CONDITIONS.items():
        assert isinstance(cfg, TrainConfig)
        assert cfg.condition == name
        assert cfg.operator_kind in ("none", "analytic", "lie", "conv")


def test_default_photo_ranges_are_tightened() -> None:
    assert DEFAULT_PHOTO_RANGES["beta"] == (0.7, 1.3)
    assert DEFAULT_PHOTO_RANGES["gamma"] == (0.7, 1.3)
    assert DEFAULT_PHOTO_RANGES["sat"] == (0.05, 1.5)
    lo, hi = DEFAULT_PHOTO_RANGES["hue"]
    assert lo == pytest.approx(-math.pi / 4) and hi == pytest.approx(math.pi / 4)
    for cfg in CONDITIONS.values():
        assert cfg.photo_ranges == DEFAULT_PHOTO_RANGES


def test_baselines() -> None:
    b1 = CONDITIONS["b1"]
    assert not b1.tau_photo and not b1.tau_spatial  # b1 has NO tau
    assert b1.operator_kind == "none"
    b2 = CONDITIONS["b2lite"]
    assert b2.tau_spatial and not b2.tau_photo  # spatial only
    assert b2.operator_kind == "none"  # spatial ops ARE the latent op


def test_photometric_conditions() -> None:
    assert CONDITIONS["p1_analytic"].operator_kind == "analytic"
    assert CONDITIONS["p1_analytic"].tau_photo
    assert CONDITIONS["p2_lie"].operator_kind == "lie"
    assert CONDITIONS["p2_nocomp"].operator_kind == "lie"
    assert CONDITIONS["p2_nocomp"].lambda_comp == 0.0
    assert CONDITIONS["p2_lie"].lambda_comp == 0.1
    assert CONDITIONS["p2_noinit"].analytic_init is False
    assert CONDITIONS["p2_lie"].analytic_init is True
    assert CONDITIONS["p3_conv"].operator_kind == "conv"
    for name in ("p1_analytic", "p2_lie", "p2_nocomp", "p2_noinit", "p3_conv"):
        cfg = CONDITIONS[name]
        assert cfg.tau_photo and not cfg.tau_spatial
        assert cfg.active_factors == ("brightness", "contrast", "saturation", "hue")


def test_c1proxy_shape() -> None:
    cfg = CONDITIONS["c1proxy"]
    assert not cfg.tau_photo and not cfg.tau_spatial  # no tau at all
    assert cfg.operator_kind == "none"
    assert cfg.spectral_stats is None  # must be supplied at get_config time
    assert cfg.lambda_spectral == 1.0


def test_single_factor_conditions_restrict_active_factors() -> None:
    for factor in ("brightness", "contrast", "saturation", "hue"):
        cfg = CONDITIONS[f"single_{factor}"]
        assert cfg.operator_kind == "lie"
        assert cfg.tau_photo and not cfg.tau_spatial
        assert cfg.active_factors == (factor,)
        assert FACTOR_FIELDS[factor] in cfg.photo_ranges


# ---------------------------------------------------------------- get_config


def test_get_config_returns_copy_with_overrides() -> None:
    cfg = get_config("b1", steps=100, lr=3e-4, seed=7)
    assert cfg.steps == 100 and cfg.lr == 3e-4 and cfg.seed == 7
    assert cfg.condition == "b1"
    # registry untouched
    assert CONDITIONS["b1"].steps == 20_000
    assert CONDITIONS["b1"].lr == 1e-4


def test_get_config_photo_ranges_isolated_from_registry() -> None:
    cfg = get_config("p1_analytic")
    cfg.photo_ranges["beta"] = (0.0, 9.0)
    assert CONDITIONS["p1_analytic"].photo_ranges["beta"] == (0.7, 1.3)
    assert get_config("p1_analytic").photo_ranges["beta"] == (0.7, 1.3)


def test_get_config_unknown_name_lists_valid_names() -> None:
    with pytest.raises(ValueError) as exc:
        get_config("p9_nonsense")
    for name in EXPECTED_NAMES:
        assert name in str(exc.value)


def test_get_config_unknown_override_raises() -> None:
    with pytest.raises(ValueError):
        get_config("b1", not_a_field=1)


def test_c1proxy_demands_spectral_stats_at_get_config_time() -> None:
    with pytest.raises(ValueError, match="spectral_stats"):
        get_config("c1proxy")
    cfg = get_config("c1proxy", spectral_stats="outputs/spectrum_b1.json")
    assert cfg.spectral_stats == "outputs/spectrum_b1.json"


def test_get_config_defaults_match_spec() -> None:
    cfg = get_config("p2_lie")
    assert cfg.p_eq == 0.5
    assert cfg.batch_size == 8
    assert cfg.steps == 20_000
    assert cfg.image_size == 256
    assert cfg.lambda_kl == 1e-6
    assert cfg.lambda_lpips == 1.0
    assert cfg.seed == 0


# ------------------------------------------- PhotoParams.sample(ranges=...)
# The single permitted v1 edit (SPEC2 design decisions).


def test_sample_ranges_none_matches_v1_bit_exact() -> None:
    r1 = torch.Generator().manual_seed(7)
    r2 = torch.Generator().manual_seed(7)
    assert PhotoParams.sample(r1, 16) == PhotoParams.sample(r2, 16, ranges=None)


def test_sample_with_tightened_ranges() -> None:
    rng = torch.Generator().manual_seed(0)
    ps = PhotoParams.sample(rng, 256, ranges=DEFAULT_PHOTO_RANGES)
    for p in ps:
        for val, ident, (lo, hi) in (
            (p.beta, 1.0, DEFAULT_PHOTO_RANGES["beta"]),
            (p.gamma, 1.0, DEFAULT_PHOTO_RANGES["gamma"]),
            (p.sat, 1.0, DEFAULT_PHOTO_RANGES["sat"]),
            (p.hue, 0.0, DEFAULT_PHOTO_RANGES["hue"]),
        ):
            if val != ident:  # inactive factors sit exactly at identity
                assert lo - 1e-9 <= val <= hi + 1e-9


def test_sample_partial_ranges_fall_back_to_v1() -> None:
    rng = torch.Generator().manual_seed(1)
    ps = PhotoParams.sample(rng, 256, ranges={"beta": (0.9, 1.1)})
    saw_wide_gamma = False
    for p in ps:
        if p.beta != 1.0:
            assert 0.9 - 1e-9 <= p.beta <= 1.1 + 1e-9
        if p.gamma != 1.0:
            assert 0.6 - 1e-9 <= p.gamma <= 1.4 + 1e-9  # v1 default retained
            saw_wide_gamma = saw_wide_gamma or p.gamma < 0.9 or p.gamma > 1.1
    assert saw_wide_gamma  # gamma really still spans the wide v1 range


def test_sample_unknown_range_key_raises() -> None:
    rng = torch.Generator().manual_seed(2)
    with pytest.raises(ValueError):
        PhotoParams.sample(rng, 4, ranges={"brightness": (0.9, 1.1)})
