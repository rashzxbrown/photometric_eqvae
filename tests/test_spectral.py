"""Tests for pheq.spectral (SPEC2 "spectral.py"). Offline, CPU, deterministic."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from pheq.spectral import (
    SpectralMatchLoss,
    effective_rank,
    high_freq_fraction,
    load_spectrum_stats,
    radial_power_spectrum,
    save_spectrum_stats,
    spectral_slope,
)


def _randn(*shape: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen)


def _blur(z: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Depthwise box blur (padding='same') — a crude low-pass."""
    c = z.shape[1]
    kernel = torch.full((c, 1, k, k), 1.0 / (k * k))
    return F.conv2d(z, kernel, padding=k // 2, groups=c)


# ---------------------------------------------------------------- spectrum


def test_radial_power_spectrum_basic_properties() -> None:
    z = _randn(2, 4, 32, 32, seed=0)
    freqs, power = radial_power_spectrum(z)
    assert freqs.shape == power.shape
    assert freqs.dtype == torch.float32 and power.dtype == torch.float32
    # integer radius axis starting at 0, ascending by 1
    assert torch.equal(freqs, torch.arange(freqs.numel(), dtype=torch.float32))
    # covers the full FFT plane incl. corners: r_max = round(sqrt(2) * 16)
    assert freqs[-1].item() == round(math.sqrt(2.0) * 16)
    # normalized to sum 1, nonnegative
    assert torch.all(power >= 0)
    assert abs(power.sum().item() - 1.0) < 1e-5


def test_radial_power_spectrum_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        radial_power_spectrum(torch.zeros(3, 8, 8))


def test_pure_tone_peaks_at_correct_radius() -> None:
    # cos(2*pi*5*x/32) along width, constant along height: all power must land
    # at integer radius 5 — pins the frequency axis (cycles per image).
    w = 32
    x = torch.arange(w, dtype=torch.float32)
    tone = torch.cos(2.0 * math.pi * 5.0 * x / w)
    z = tone.view(1, 1, 1, w).expand(2, 4, w, w).contiguous()
    freqs, power = radial_power_spectrum(z)
    peak_radius = int(freqs[power.argmax()].item())
    assert peak_radius == 5
    assert power[5].item() > 0.99  # essentially all power in that annulus


def test_white_noise_slope_near_zero() -> None:
    z = _randn(8, 4, 64, 64, seed=1)
    slope = spectral_slope(z)
    assert abs(slope) < 0.1, slope


def test_smoothed_noise_negative_slope() -> None:
    z = _blur(_randn(8, 4, 64, 64, seed=2), k=5)
    slope = spectral_slope(z)
    assert slope < -0.5, slope


def test_spectral_slope_too_small_raises() -> None:
    with pytest.raises(ValueError):
        spectral_slope(_randn(1, 4, 4, 4, seed=3))


def test_high_freq_fraction_orders_white_vs_blurred() -> None:
    z = _randn(8, 4, 32, 32, seed=4)
    hf_white = high_freq_fraction(z)
    hf_blur = high_freq_fraction(_blur(z, k=5))
    assert 0.0 < hf_blur < hf_white < 1.0
    # cutoff=0 counts everything except DC
    assert high_freq_fraction(z, cutoff=0.0) > 0.9


# ---------------------------------------------------------------- rank


def test_effective_rank_isotropic_noise_near_c() -> None:
    z = _randn(8, 4, 16, 16, seed=5)
    er = effective_rank(z)
    assert 3.8 < er <= 4.0 + 1e-4, er


def test_effective_rank_rank_one_near_one() -> None:
    u = _randn(8, 16, 16, seed=6)
    v = torch.tensor([1.0, 2.0, -1.0, 0.5])
    z = u[:, None, :, :] * v[None, :, None, None]
    er = effective_rank(z)
    assert 1.0 <= er < 1.05, er


def test_effective_rank_constant_latent() -> None:
    assert effective_rank(torch.ones(2, 4, 8, 8)) == 1.0


# ---------------------------------------------------------------- match loss


def test_spectral_match_loss_zero_against_own_spectrum() -> None:
    z = _randn(4, 4, 32, 32, seed=7)
    freqs, power = radial_power_spectrum(z)
    loss = SpectralMatchLoss(freqs, power)(z)
    assert loss.item() < 1e-10


def test_spectral_match_loss_positive_for_mismatched_spectrum() -> None:
    z_white = _randn(4, 4, 32, 32, seed=8)
    z_smooth = _blur(z_white, k=5)
    freqs, power = radial_power_spectrum(z_white)
    loss = SpectralMatchLoss(freqs, power)(z_smooth)
    assert loss.item() > 0.1


def test_spectral_match_loss_differentiable() -> None:
    z = _randn(2, 4, 16, 16, seed=9).requires_grad_(True)
    target = radial_power_spectrum(_blur(_randn(2, 4, 16, 16, seed=10), k=3))
    loss = SpectralMatchLoss(*target)(z)
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum().item() > 0.0


def test_spectral_match_loss_shared_radii_across_sizes() -> None:
    # target measured on 16x16 latents, applied to 32x32 latents: only the
    # intersection of integer radii is compared; loss must be finite.
    target = radial_power_spectrum(_randn(4, 4, 16, 16, seed=11))
    loss = SpectralMatchLoss(*target)(_randn(4, 4, 32, 32, seed=12))
    assert torch.isfinite(loss)


def test_spectral_match_loss_validates_inputs() -> None:
    with pytest.raises(ValueError):
        SpectralMatchLoss(torch.arange(3.0), torch.ones(4))
    with pytest.raises(ValueError):
        SpectralMatchLoss(torch.zeros(0), torch.zeros(0))


# ---------------------------------------------------------------- stats io


def test_spectrum_stats_roundtrip(tmp_path) -> None:
    z = _randn(2, 4, 32, 32, seed=13)
    freqs, power = radial_power_spectrum(z)
    path = save_spectrum_stats(tmp_path / "stats.json", freqs, power)
    freqs2, power2 = load_spectrum_stats(path)
    assert torch.allclose(freqs, freqs2)
    assert torch.allclose(power, power2)
    # loaded stats plug straight into the loss and reproduce (near-)zero
    loss = SpectralMatchLoss(freqs2, power2)(z)
    assert loss.item() < 1e-8
