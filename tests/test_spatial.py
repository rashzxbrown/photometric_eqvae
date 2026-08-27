"""Tests for pheq.spatial (SPEC2 spatial.py).

Covers: sampling determinism/ranges; the identity fast-path; bit-exact
commutation of rot90 with apply_channel_affine (plan §3 commutation lemma,
permutation case); f-alignment arithmetic of scaled sizes (image a multiple
of f, latent exactly 1/f); and the inherited-operator check on ToyLinearAE —
scale+rot on the latent then decode vs decode then scale+rot on the image
(exact for rot90, interpolation-tolerance for scale).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pheq.analytic import apply_channel_affine
from pheq.spatial import SpatialParams, apply_spatial, scaled_size
from pheq.vae import ToyLinearAE

torch.manual_seed(0)


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


# ---------------------------------------------------------------------------
# SpatialParams.sample
# ---------------------------------------------------------------------------


def test_sample_deterministic() -> None:
    a = [SpatialParams.sample(_gen(7)) for _ in range(1)][0]
    b = SpatialParams.sample(_gen(7))
    assert a == b
    # A stream of samples from one generator is reproducible as a whole.
    g1, g2 = _gen(3), _gen(3)
    s1 = [SpatialParams.sample(g1) for _ in range(20)]
    s2 = [SpatialParams.sample(g2) for _ in range(20)]
    assert s1 == s2


def test_sample_inactive_is_identity() -> None:
    g = _gen(0)
    for _ in range(10):
        p = SpatialParams.sample(g, p_scale=0.0, p_rot=0.0)
        assert p.scale == 1.0 and p.rot90 == 0


def test_sample_active_ranges() -> None:
    g = _gen(1)
    scales, rots = [], []
    for _ in range(200):
        p = SpatialParams.sample(g, p_scale=1.0, p_rot=1.0, scale_range=(0.25, 1.0))
        scales.append(p.scale)
        rots.append(p.rot90)
    assert all(0.25 <= s <= 1.0 for s in scales)
    # Active rotations are the NON-identity set {1, 2, 3} (documented resolution).
    assert set(rots) == {1, 2, 3}
    assert min(scales) < 0.4 and max(scales) > 0.85  # spans the range


def test_sample_fixed_draw_count() -> None:
    """Inactive branches still consume draws: generator state advances identically."""
    g1, g2 = _gen(11), _gen(11)
    SpatialParams.sample(g1, p_scale=0.0, p_rot=0.0)
    SpatialParams.sample(g2, p_scale=1.0, p_rot=1.0)
    assert torch.equal(torch.rand(4, generator=g1), torch.rand(4, generator=g2))


# ---------------------------------------------------------------------------
# apply_spatial basics
# ---------------------------------------------------------------------------


def test_identity_fast_path_returns_input() -> None:
    x = torch.randn(2, 4, 8, 8, generator=_gen(0))
    out = apply_spatial(x, SpatialParams(scale=1.0, rot90=0))
    assert out is x


def test_no_resample_when_target_equals_size() -> None:
    """scale != 1 but rounded target == current size: values pass through bit-exact."""
    x = torch.randn(1, 4, 8, 8, generator=_gen(0))
    out = apply_spatial(x, SpatialParams(scale=0.99, rot90=0))
    assert torch.equal(out, x)


def test_rot90_nonsquare_shape() -> None:
    x = torch.randn(1, 3, 6, 8, generator=_gen(0))
    out = apply_spatial(x, SpatialParams(scale=1.0, rot90=1))
    assert out.shape == (1, 3, 8, 6)
    assert torch.equal(out, torch.rot90(x, 1, dims=(-2, -1)))


def test_antialias_flag() -> None:
    x = torch.randn(1, 3, 16, 16, generator=_gen(0))
    p = SpatialParams(scale=0.5, rot90=0)
    a = apply_spatial(x, p, antialias=True)
    b = apply_spatial(x, p, antialias=False)
    assert a.shape == b.shape == (1, 3, 8, 8)
    assert not torch.equal(a, b)  # the flag reaches F.interpolate


# ---------------------------------------------------------------------------
# rot90 ∘ channel_affine == channel_affine ∘ rot90, bit-exact (care point)
# ---------------------------------------------------------------------------


def test_rot90_commutes_with_channel_affine_bitexact() -> None:
    g = _gen(2)
    z = torch.randn(2, 4, 8, 8, generator=g)
    m_mat = torch.randn(4, 4, generator=g)
    m_vec = torch.randn(4, generator=g)
    for k in range(4):
        p = SpatialParams(scale=1.0, rot90=k)
        lhs = apply_spatial(apply_channel_affine(z, m_mat, m_vec), p)
        rhs = apply_channel_affine(apply_spatial(z, p), m_mat, m_vec)
        assert torch.equal(lhs, rhs), f"rot90 k={k} not bit-exact"


# ---------------------------------------------------------------------------
# f-alignment arithmetic (care point: image multiple of 8, latent exactly 1/8)
# ---------------------------------------------------------------------------


def test_scaled_size_f_alignment_arithmetic() -> None:
    scales = (0.25, 0.15, 0.3, 1.0 / 3.0, 0.5, 0.77, 0.9, 1.0)
    for f in (2, 8):
        for h_lat in (1, 3, 4, 6, 10, 25, 32):
            h_img = f * h_lat
            for s in scales:
                img_t = scaled_size(h_img, s, f)
                lat_t = scaled_size(h_lat, s, 1)
                assert img_t % f == 0
                assert img_t == f * lat_t, (
                    f"misaligned: f={f} H={h_img} s={s}: image {img_t}, latent {lat_t}"
                )


def test_scaled_size_tie_consistency() -> None:
    # H*s/8 lands exactly on .5: both sides must round identically.
    assert scaled_size(48, 0.25, 8) == 8 * scaled_size(6, 0.25, 1)  # 1.5 -> 2
    assert scaled_size(16, 0.75, 8) == 8 * scaled_size(2, 0.75, 1)  # 1.5 -> 2


def test_scaled_size_min_guard() -> None:
    assert scaled_size(8, 0.25, 8) == 8  # would round to 0 without the guard
    assert scaled_size(1, 0.25, 1) == 1
    # the guard preserves alignment on both sides
    assert scaled_size(8, 0.25, 8) == 8 * scaled_size(1, 0.25, 1)


def test_apply_spatial_shapes_f_aligned() -> None:
    """Image (f=8) and latent (f=1) outputs stay exactly f-aligned, incl. rot."""
    img = torch.randn(1, 3, 48, 32, generator=_gen(3))
    lat = torch.randn(1, 4, 6, 4, generator=_gen(4))
    for s in (0.25, 0.5, 0.77, 1.0):
        for k in range(4):
            p = SpatialParams(scale=s, rot90=k)
            img_t = apply_spatial(img, p, f=8)
            lat_t = apply_spatial(lat, p, f=1)
            assert img_t.shape[-2] == 8 * lat_t.shape[-2]
            assert img_t.shape[-1] == 8 * lat_t.shape[-1]
            assert img_t.shape[-2] % 8 == 0 and img_t.shape[-1] % 8 == 0


# ---------------------------------------------------------------------------
# ToyLinearAE: inherited operator — latent-then-decode vs decode-then-image
# ---------------------------------------------------------------------------


def _smooth_flat_latent(size: int, base: int = 2) -> torch.Tensor:
    """Smooth latent with zero detail channel (in ToyLinearAE's rot90-exact range).

    The detail (checkerboard) channel is zeroed: its within-block decode
    pattern is rot90-ODD (P -> -P), so the blockwise decoder only commutes
    with rot90 on the flat color subspace — which is exactly the range of
    block-constant images. Smoothness (low-frequency content) controls the
    interpolation-filter mismatch in the scale comparison.
    """
    z = F.interpolate(
        torch.randn(2, 4, base, base, generator=_gen(5)),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )
    z[:, 3] = 0.0
    return z


def test_toylinear_rot90_decode_commutes_exact() -> None:
    ae = ToyLinearAE()
    z = _smooth_flat_latent(16)
    for k in range(4):
        p = SpatialParams(scale=1.0, rot90=k)
        lhs = ae.decode(apply_spatial(z, p, f=1))
        rhs = apply_spatial(ae.decode(z), p, f=2)
        assert torch.equal(lhs, rhs), f"rot90 k={k} through decoder not exact"


def test_toylinear_scale_rot_decode_commutes_tolerance() -> None:
    """Interpolation-tolerance version for scaling (SPEC2 spatial.py tests).

    Calibrated: for latents that are 2x2 noise upsampled to 64x64, the max
    relative L2 mismatch over s in [0.25, 0.9] x k in {0,1,2} is ~0.04
    (pure interpolation-filter difference between resizing the latent grid
    and resizing the nearest-2x-upsampled image); asserted < 0.08 with 2x
    headroom. The mismatch vanishes in the low-frequency limit — checked by
    comparing against a rougher latent.
    """
    ae = ToyLinearAE()
    z = _smooth_flat_latent(64, base=2)
    worst = 0.0
    for s in (0.25, 0.5, 0.75, 0.9):
        for k in (0, 1, 2):
            p = SpatialParams(scale=s, rot90=k)
            lhs = ae.decode(apply_spatial(z, p, f=1))
            rhs = apply_spatial(ae.decode(z), p, f=2)
            assert lhs.shape == rhs.shape  # f-alignment through the decoder
            rel = float((lhs - rhs).norm() / rhs.norm())
            worst = max(worst, rel)
    assert worst < 0.08, f"scale commutation rel error {worst:.4f}"

    # Rougher latent (same construction, higher base frequency) => larger defect:
    # the residual really is interpolation-filter mismatch, not a bug.
    z_rough = _smooth_flat_latent(64, base=8)
    p = SpatialParams(scale=0.5, rot90=0)
    lhs = ae.decode(apply_spatial(z_rough, p, f=1))
    rhs = apply_spatial(ae.decode(z_rough), p, f=2)
    assert float((lhs - rhs).norm() / rhs.norm()) > worst
