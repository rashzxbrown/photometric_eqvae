"""Numerical verification of the photometric-spatial commutation lemma (plan §3.1).

For a channel-affine map T(z) = M z + m applied pointwise across the latent
grid, and a spatial resampling S built from CONVEX combinations of grid values
(bilinear grid_sample with border padding, horizontal flip, rot90):

    S(M z + m) = M S(z) + m      (exactly, up to float32 roundoff)

because affine maps commute with convex combinations: if each output site is
sum_i w_i z_{p_i} with w_i >= 0 and sum_i w_i = 1 over actual input values,
then sum_i w_i (M z_{p_i} + m) = M (sum_i w_i z_{p_i}) + m.

NEGATIVE control: with padding_mode='zeros' and a warp that samples outside
the grid, out-of-bounds taps contribute literal zeros instead of values of z,
so the interpolation weights over actual input values sum to < 1 and the bias
term breaks: the identity FAILS for m != 0 (while still holding for m = 0,
since the linear part alone commutes with any linear resampling). This is why
the lemma requires convex-combination interpolation.

Only imports torch and pheq.analytic.apply_channel_affine, per SPEC.
"""

import torch
import torch.nn.functional as F

from pheq.analytic import apply_channel_affine

TOL = 1e-5


def _rand_channel_affine(gen: torch.Generator, c: int = 4):
    M = torch.randn(c, c, generator=gen)
    m = torch.randn(c, generator=gen)
    return M, m


def _affine_grid(z_shape, theta: torch.Tensor) -> torch.Tensor:
    return F.affine_grid(theta, list(z_shape), align_corners=False)


def _in_grid_theta(batch: int, gen: torch.Generator) -> torch.Tensor:
    """Random rotation + shrink (scale <= 0.6) + small shift: every sample
    location stays inside [-1, 1]^2 (max radius 0.6*sqrt(2) + 0.1 < 1)."""
    ang = 0.5 * torch.rand(batch, generator=gen) - 0.25
    scale = 0.4 + 0.2 * torch.rand(batch, generator=gen)
    shift = 0.2 * torch.rand(batch, 2, generator=gen) - 0.1
    cos, sin = scale * torch.cos(ang), scale * torch.sin(ang)
    theta = torch.zeros(batch, 2, 3)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, shift[:, 0]
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, shift[:, 1]
    return theta


def _zoom_out_theta(batch: int, factor: float = 1.5) -> torch.Tensor:
    """Zoom-out warp: sample coordinates reach +/-factor, i.e. OUTSIDE the grid."""
    theta = torch.zeros(batch, 2, 3)
    theta[:, 0, 0] = factor
    theta[:, 1, 1] = factor
    return theta


def _defect(spatial, z: torch.Tensor, M: torch.Tensor, m: torch.Tensor) -> float:
    """max |S(M z + m) - (M S(z) + m)| — the commutation defect."""
    lhs = spatial(apply_channel_affine(z, M, m))
    rhs = apply_channel_affine(spatial(z), M, m)
    return float((lhs - rhs).abs().max())


def test_bilinear_border_warp_commutes():
    """Random in-grid bilinear warp, padding_mode='border': exact commutation."""
    gen = torch.Generator().manual_seed(0)
    z = torch.randn(3, 4, 16, 16, generator=gen)
    M, m = _rand_channel_affine(gen)
    grid = _affine_grid(z.shape, _in_grid_theta(z.shape[0], gen))

    def warp(t):
        return F.grid_sample(t, grid, mode="bilinear",
                             padding_mode="border", align_corners=False)

    assert _defect(warp, z, M, m) < TOL


def test_bilinear_border_out_of_grid_still_commutes():
    """Border padding replicates edge VALUES of the input, so even an
    out-of-grid warp is a convex combination of actual values of z and the
    identity still holds — sharpening the contrast with the zeros control."""
    gen = torch.Generator().manual_seed(1)
    z = torch.randn(2, 4, 16, 16, generator=gen)
    M, m = _rand_channel_affine(gen)
    grid = _affine_grid(z.shape, _zoom_out_theta(z.shape[0]))

    def warp(t):
        return F.grid_sample(t, grid, mode="bilinear",
                             padding_mode="border", align_corners=False)

    assert _defect(warp, z, M, m) < TOL


def test_hflip_commutes():
    """Horizontal flip is a permutation of sites: commutation is exact."""
    gen = torch.Generator().manual_seed(2)
    z = torch.randn(2, 4, 16, 16, generator=gen)
    M, m = _rand_channel_affine(gen)

    def hflip(t):
        return torch.flip(t, dims=[-1])

    assert _defect(hflip, z, M, m) < TOL


def test_rot90_commutes():
    """rot90 (all four multiplicities) is a permutation of sites: exact."""
    gen = torch.Generator().manual_seed(3)
    z = torch.randn(2, 4, 16, 16, generator=gen)
    M, m = _rand_channel_affine(gen)
    for k in (1, 2, 3):
        def rot(t, k=k):
            return torch.rot90(t, k=k, dims=(-2, -1))

        assert _defect(rot, z, M, m) < TOL


def test_zeros_padding_out_of_grid_breaks_bias():
    """NEGATIVE control (plan §3.1 lemma hypothesis): padding_mode='zeros' +
    out-of-grid sampling makes weights over actual input values sum to < 1,
    so the identity FAILS for m != 0. At fully-outside sites the defect equals
    |m| exactly: S(Mz + m) = 0 there while M S(z) + m = m. The linear part
    (m = 0) still commutes, isolating the failure to the bias term."""
    gen = torch.Generator().manual_seed(4)
    z = torch.randn(2, 4, 16, 16, generator=gen)
    M, _ = _rand_channel_affine(gen)
    m = torch.tensor([0.5, -1.0, 2.0, 0.25])  # decidedly nonzero bias
    grid = _affine_grid(z.shape, _zoom_out_theta(z.shape[0], factor=1.5))

    def warp(t):
        return F.grid_sample(t, grid, mode="bilinear",
                             padding_mode="zeros", align_corners=False)

    defect = _defect(warp, z, M, m)
    assert defect > 100 * TOL, (
        f"zeros-padding defect {defect} unexpectedly small: negative control failed"
    )
    # Fully-outside sites should realize the defect max|m| = 2.0 (up to roundoff).
    assert abs(defect - float(m.abs().max())) < 1e-4

    # Bias-free control: the identity holds for m = 0 even with zeros padding.
    assert _defect(warp, z, M, torch.zeros(4)) < TOL
