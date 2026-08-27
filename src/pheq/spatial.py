"""EQ-VAE-style spatial transforms τ_s for B2-lite and composition (SPEC2 spatial.py).

Plan §2 ("spatial predecessors"): every spatial equivariance predecessor
(EQ-VAE, AF-LDM, joint-downsampling) inherits its latent operator from the
grid — the SAME op with the SAME params is applied to the image and to the
latent, whose grid is the image grid divided by the downsampling factor f.
This module implements that inherited operator: isotropic scaling and 90°
rotations (:class:`SpatialParams`, :func:`apply_spatial`).

Photometric–spatial commutation (plan §3, commutation lemma): both rot90 (a
permutation of grid sites) and bilinear interpolation (a convex combination
of grid sites) commute with any pointwise channel-affine map ``z_p ↦ M z_p + m``
— exactly for rot90, and exactly in real arithmetic for interpolation
(convex weights sum to 1, so the bias term passes through). Verified in
tests/test_spatial.py (rot90 bit-exact) and tests/test_commutation.py (warps).

f-alignment convention (binding, SPEC2 spatial.py note): after scaling, the
image target size and the latent size must stay f-aligned — the IMAGE is
scaled to ``round(H*s/f)*f`` (a multiple of f, f = 8 for the SD-VAE) and the
latent to exactly 1/f of that. See :func:`scaled_size` for the arithmetic
that makes computing the latent target from the latent's own size exactly
consistent with computing it from the image size.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

__all__ = ["SpatialParams", "scaled_size", "apply_spatial"]


@dataclass
class SpatialParams:
    """One sampled spatial op τ_s (SPEC2 spatial.py; plan §2 EQ-VAE τ set).

    Fields default to the identity:

    Attributes:
        scale: isotropic scale factor, ∈ [0.25, 1] when sampled.
        rot90: number of 90° rotations k ∈ {0, 1, 2, 3} (``torch.rot90``
            convention: counter-clockwise in the last two dims).
    """

    scale: float = 1.0
    rot90: int = 0

    @classmethod
    def sample(
        cls,
        gen: torch.Generator,
        p_scale: float = 0.5,
        p_rot: float = 0.5,
        scale_range: tuple[float, float] = (0.25, 1.0),
    ) -> "SpatialParams":
        """Sample one spatial op, seeded by ``gen`` (SPEC2 spatial.py).

        With probability ``p_scale`` the scale is uniform in ``scale_range``
        (else 1.0); with probability ``p_rot`` the rotation count is uniform
        over the NON-identity rotations {1, 2, 3} (else 0).

        Spec resolutions (documented):
        - An *active* rotation draws from {1, 2, 3}, not {0, 1, 2, 3}: the
          identity k = 0 is the *inactive* value (mirrors
          ``PhotoParams.sample``, where inactive factors take their identity
          value and active draws cover the non-trivial range).
        - A fixed number of draws (4) is consumed from ``gen`` regardless of
          which branches are active, so downstream sampling stays aligned
          across conditions (same convention as ``PhotoParams.sample``).
        """
        u = torch.rand(4, generator=gen, dtype=torch.float64)
        scale = 1.0
        if float(u[0]) < p_scale:
            lo, hi = scale_range
            scale = float(lo + float(u[1]) * (hi - lo))
        rot = 0
        if float(u[2]) < p_rot:
            rot = 1 + min(int(float(u[3]) * 3.0), 2)
        return cls(scale=scale, rot90=rot)


def scaled_size(size: int, scale: float, f: int = 1) -> int:
    """f-aligned scaled size: ``max(1, round(size * scale / f)) * f``.

    This is the SPEC2 f-alignment arithmetic: the image (size ``H``, a
    multiple of f) is scaled to ``round(H*s/f)*f`` and the latent (size
    ``h = H/f``) to ``round(h*s)`` = exactly 1/f of the image target. The two
    computations agree *bit-exactly* whenever f is a power of two (f = 8 for
    the SD-VAE, f = 2 for the toy AEs): ``H*s = f*(h*s)`` holds exactly in
    binary floating point (multiplication/division by a power of two only
    shifts the exponent), so ``round(H*s/f) == round(h*s)`` including at
    exact .5 ties (both sides see the identical float, and Python's
    banker's rounding is applied to it identically). Callers may therefore
    compute the latent target from the latent's own size with ``f = 1`` and
    the image target from the image size with ``f = 8`` and the results stay
    f-aligned — no cross-communication needed. For non-power-of-two f the
    tie-consistency guarantee is lost; only power-of-two f is used here.

    ``max(1, ·)`` guards degenerate collapse to size 0 (e.g. size 8, s = 0.25,
    f = 8): the guard applies to the *quotient*, so image (→ f) and latent
    (→ 1) remain aligned.

    Args:
        size: current spatial size (H or W).
        scale: isotropic scale factor.
        f: alignment quantum — 1 for latents (the plain ``round(H*scale)``
           rule of SPEC2), the autoencoder downsampling factor for images.

    Returns:
        The target size (positive int, multiple of ``f``).
    """
    return max(1, round(size * scale / f)) * f


def apply_spatial(
    x: torch.Tensor,
    params: SpatialParams,
    antialias: bool = True,
    f: int = 1,
) -> torch.Tensor:
    """Apply τ_s to ANY ``(B, Ch, H, W)`` tensor — image or latent (SPEC2).

    The op is ``torch.rot90`` on the last two dims, THEN bilinear
    ``F.interpolate`` (``align_corners=False``, ``antialias=antialias``) to
    the f-aligned target size computed by :func:`scaled_size` from the
    (post-rotation) size. Identity fast-path: when ``scale == 1.0`` and
    ``rot90 == 0`` the input tensor is returned unchanged (no copy);
    resampling is also skipped when the target size equals the current size.

    The SAME op with the SAME params must be applied to the image and its
    latent (plan §2: the latent operator is inherited from the grid). To keep
    the two f-aligned after scaling (SPEC2 spatial.py note):

    - latent side: call with the default ``f = 1`` → target ``round(h*s)``;
    - image side: call with ``f = <downsampling factor>`` (8 for the SD-VAE,
      2 for the toy AEs) → target ``round(H*s/f)*f``, exactly f × the latent
      target (see :func:`scaled_size` for why the two independent
      computations agree bit-exactly for power-of-two f).

    Spec resolution: SPEC2 prints the signature without ``f`` but its
    f-alignment note requires the image target ``round(H*s/8)*8`` while the
    base rule is ``round(H*scale)`` — no single parameter-free rule produces
    both (e.g. h = 32 is itself a multiple of 8, so divisibility cannot
    distinguish image from latent). ``f`` is therefore a trailing kwarg whose
    default (1) reproduces the printed rule; only the image side passes f.

    Args:
        x: ``(B, Ch, H, W)`` tensor (image or latent).
        params: sampled spatial op.
        antialias: antialiased bilinear resampling (default True; only
            affects downscaling, which is the sampled range s ≤ 1).
        f: alignment quantum for the target size (see above).

    Returns:
        Transformed tensor ``(B, Ch, H', W')``.
    """
    k = params.rot90 % 4
    scale = params.scale
    if k == 0 and scale == 1.0:
        return x
    out = x
    if k:
        out = torch.rot90(out, k, dims=(-2, -1))
    if scale != 1.0:
        h, w = out.shape[-2], out.shape[-1]
        target = (scaled_size(h, scale, f), scaled_size(w, scale, f))
        if target != (h, w):
            out = F.interpolate(
                out,
                size=target,
                mode="bilinear",
                antialias=antialias,
                align_corners=False,
            )
    return out
