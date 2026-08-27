"""Latent spectral statistics + the C1-proxy spectral-matching regularizer.

Implements SPEC2 "spectral.py". Science context: docs/research-plan.md §3.5
(collapse monitors: spectral slope, effective rank) and the C1 spectral-matched
control of the mediation battery (plan §5 / RQ5) — condition ``c1proxy`` trains
with :class:`SpectralMatchLoss` INSTEAD of any equivariance branch.

Conventions (binding):

- Latents are ``(B, C, h, w)``.
- The radial spectrum is binned by INTEGER radius on the unshifted FFT grid
  with per-axis integer frequencies (cycles per image):
  ``k_y ∈ fftfreq(h)·h``, ``k_x ∈ fftfreq(w)·w`` and
  ``r = round(sqrt(k_x² + k_y²))``. All radii present in the plane are kept
  (including the incompletely-sampled corner annuli beyond the axis Nyquist),
  so the normalized power vector accounts for the full FFT plane.
- Power at each radius is the MEAN over the annulus bins (radially-AVERAGED
  spectrum — white noise is flat, slope ≈ 0), averaged over batch and
  channels, then normalized so the returned power vector sums to 1.
- Spectrum computation is differentiable w.r.t. ``z`` (out-of-place
  ``index_add`` binning), so :class:`SpectralMatchLoss` is a valid training
  loss; the scalar monitors (:func:`spectral_slope`, :func:`high_freq_fraction`,
  :func:`effective_rank`) run under ``no_grad`` and return floats.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

__all__ = [
    "radial_power_spectrum",
    "spectral_slope",
    "high_freq_fraction",
    "effective_rank",
    "SpectralMatchLoss",
    "save_spectrum_stats",
    "load_spectrum_stats",
]


def _radius_index(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Integer radius ``round(sqrt(k_x² + k_y²))`` per FFT bin, ``(h, w)`` long.

    Frequencies are integer cycles-per-image on each axis (``fftfreq(n, d=1/n)``
    yields exactly ``0, 1, …, n//2 − 1, −n//2, …, −1``), so the radius axis is
    in cycles per image — the "correct frequency axis" of SPEC2.
    """
    fy = torch.fft.fftfreq(h, d=1.0 / h, device=device)
    fx = torch.fft.fftfreq(w, d=1.0 / w, device=device)
    r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    return r.round().long()


def radial_power_spectrum(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Radially-averaged 2D power spectrum of a latent batch (SPEC2, plan §3.5).

    2D FFT per channel, ``|·|²``, binned by integer radius (annulus MEAN),
    averaged over batch + channels, normalized to sum 1.

    Args:
        z: ``(B, C, h, w)`` latent tensor (any float dtype; computed in f32).

    Returns:
        ``(freqs, power)``: ``freqs`` is ``(F,)`` float32 = ``0, 1, …, r_max``
        (cycles per image); ``power`` is ``(F,)``, sums to 1, and is
        differentiable w.r.t. ``z``. Radii with no FFT bins (possible only for
        tiny grids) get power 0.
    """
    if z.ndim != 4:
        raise ValueError(f"expected (B, C, h, w) latents, got shape {tuple(z.shape)}")
    _, _, h, w = z.shape
    spec = torch.fft.fft2(z.to(torch.float32))
    p = (spec.real**2 + spec.imag**2).mean(dim=(0, 1))  # (h, w)
    idx = _radius_index(h, w, z.device)
    n_bins = int(idx.max().item()) + 1
    flat_idx = idx.reshape(-1)
    sums = torch.zeros(n_bins, dtype=p.dtype, device=p.device).index_add(
        0, flat_idx, p.reshape(-1)
    )
    counts = torch.zeros(n_bins, dtype=p.dtype, device=p.device).index_add(
        0, flat_idx, torch.ones(flat_idx.numel(), dtype=p.dtype, device=p.device)
    )
    power = sums / counts.clamp_min(1.0)
    power = power / power.sum().clamp_min(torch.finfo(torch.float32).tiny)
    freqs = torch.arange(n_bins, dtype=torch.float32, device=z.device)
    return freqs, power


def spectral_slope(z: torch.Tensor) -> float:
    """Slope of the linear fit of log(power) vs log(freq) over radii [2, h/2].

    SPEC2: DC (radius 0) and radius 1 are SKIPPED; the fit stops at the axis
    Nyquist ``h // 2`` (``min(h, w) // 2`` for non-square latents), excluding
    the incompletely-sampled corner annuli. White noise → slope ≈ 0 (annulus
    means are flat); smoothed latents → negative slope (plan §3.5 monitor,
    §5 mediation battery).
    """
    n = min(z.shape[-2], z.shape[-1])
    r_max = n // 2
    if r_max < 3:
        raise ValueError(
            f"spatial size {n} too small for a slope fit over radii [2, {r_max}]"
        )
    with torch.no_grad():
        freqs, power = radial_power_spectrum(z)
        mask = (freqs >= 2.0) & (freqs <= float(r_max)) & (power > 0)
        x = freqs[mask].log()
        y = power[mask].log()
        if x.numel() < 2:
            raise ValueError("fewer than 2 nonzero-power radii in [2, h/2]")
        xm = x - x.mean()
        ym = y - y.mean()
        return float(((xm * ym).sum() / (xm * xm).sum()).item())


def high_freq_fraction(z: torch.Tensor, cutoff: float = 0.5) -> float:
    """Fraction of (radially-binned) power at radii above ``cutoff · (h/2)``.

    SPEC2 / plan §5 mediation battery ("high-frequency energy fraction").
    Computed on the annulus-mean power vector of :func:`radial_power_spectrum`
    (the same statistic the slope is fit on), with ``h = min(h, w)``; radii
    strictly greater than ``cutoff · h/2`` count as high frequency, including
    the corner annuli beyond the axis Nyquist.
    """
    n = min(z.shape[-2], z.shape[-1])
    with torch.no_grad():
        freqs, power = radial_power_spectrum(z)
        mask = freqs > cutoff * (n / 2.0)
        total = power.sum().clamp_min(torch.finfo(torch.float32).tiny)
        return float((power[mask].sum() / total).item())


def effective_rank(z: torch.Tensor) -> float:
    """Effective rank (Roy & Vetterli) of the centered ``(B·h·w, C)`` matrix.

    ``exp(H(p))`` where ``p_i = s_i / Σ s_j`` are the L1-normalized singular
    values of the CENTERED matrix of per-site latent vectors (channel means
    subtracted). For ``C = 4`` this lies in ``[1, 4]``. Plan §3.5 collapse
    monitor / §5 mediation battery ("PCA effective rank").
    """
    if z.ndim != 4:
        raise ValueError(f"expected (B, C, h, w) latents, got shape {tuple(z.shape)}")
    c = z.shape[1]
    with torch.no_grad():
        x = z.to(torch.float32).permute(0, 2, 3, 1).reshape(-1, c)
        x = x - x.mean(dim=0, keepdim=True)
        s = torch.linalg.svdvals(x)
        total = s.sum()
        if float(total.item()) <= 0.0:
            return 1.0  # constant latent: a single (degenerate) direction
        p = s / total
        p = p[p > 0]
        entropy = -(p * p.log()).sum()
        return float(entropy.exp().item())


class SpectralMatchLoss(torch.nn.Module):
    """MSE between log target and log measured radial power over SHARED radii.

    The C1-proxy regularizer (SPEC2; plan §5, condition C1 / RQ5 mediation):
    condition ``c1proxy`` uses this INSTEAD of any equivariance branch, to
    match a reference latent spectrum (from :func:`save_spectrum_stats` JSON)
    without imposing equivariance.

    Shared radii are the intersection of the integer radii of the target and
    of the measured spectrum, so a target measured at one latent size can
    regularize another (only the common low radii are compared). Powers are
    clamped to ``eps = 1e-12`` before the log so empty/zero annuli stay
    finite. The loss is differentiable w.r.t. ``z``.
    """

    eps: float = 1e-12

    def __init__(self, freqs: torch.Tensor, power: torch.Tensor) -> None:
        """``freqs, power``: target spectrum as returned by
        :func:`radial_power_spectrum` / :func:`load_spectrum_stats`."""
        super().__init__()
        freqs = torch.as_tensor(freqs, dtype=torch.float32).reshape(-1)
        power = torch.as_tensor(power, dtype=torch.float32).reshape(-1)
        if freqs.shape != power.shape:
            raise ValueError(
                f"freqs {tuple(freqs.shape)} and power {tuple(power.shape)} mismatch"
            )
        if freqs.numel() == 0:
            raise ValueError("empty target spectrum")
        self.register_buffer("target_freqs", freqs)
        self.register_buffer("target_log_power", power.clamp_min(self.eps).log())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """MSE(log target power, log measured power) over shared radii."""
        freqs, power = radial_power_spectrum(z)
        t_map = {int(round(v)): i for i, v in enumerate(self.target_freqs.tolist())}
        m_map = {int(round(v)): i for i, v in enumerate(freqs.tolist())}
        shared = sorted(t_map.keys() & m_map.keys())
        if not shared:
            raise ValueError("no shared radii between target and measured spectra")
        t_pos = torch.tensor(
            [t_map[r] for r in shared], device=self.target_log_power.device
        )
        m_pos = torch.tensor([m_map[r] for r in shared], device=power.device)
        log_m = power[m_pos].clamp_min(self.eps).log()
        log_t = self.target_log_power[t_pos].to(log_m.device)
        return torch.mean((log_m - log_t) ** 2)


def save_spectrum_stats(
    path: str | Path, freqs: torch.Tensor, power: torch.Tensor
) -> Path:
    """Save a radial spectrum to JSON (``{"freqs": [...], "power": [...]}``).

    SPEC2: the c1proxy condition loads these stats at train time.
    """
    path = Path(path)
    freqs = torch.as_tensor(freqs, dtype=torch.float32).reshape(-1)
    power = torch.as_tensor(power, dtype=torch.float32).reshape(-1)
    if freqs.shape != power.shape:
        raise ValueError(
            f"freqs {tuple(freqs.shape)} and power {tuple(power.shape)} mismatch"
        )
    payload = {"freqs": freqs.tolist(), "power": power.tolist()}
    path.write_text(json.dumps(payload))
    return path


def load_spectrum_stats(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load ``(freqs, power)`` float32 tensors from a stats JSON.

    Inverse of :func:`save_spectrum_stats`; the tuple unpacks straight into
    ``SpectralMatchLoss(*load_spectrum_stats(path))``.
    """
    data = json.loads(Path(path).read_text())
    freqs = torch.tensor(data["freqs"], dtype=torch.float32)
    power = torch.tensor(data["power"], dtype=torch.float32)
    if freqs.shape != power.shape:
        raise ValueError(f"corrupt spectrum stats at {path}: freqs/power mismatch")
    return freqs, power
