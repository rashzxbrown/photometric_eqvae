"""Tests for pheq.analytic (SPEC.md "pheq/analytic.py", plan §3.2/§3.4).

NOTE on the composition convention: these tests use ``pheq.color.compose`` when
available. At the time of writing, pheq/color.py is being implemented
concurrently, so we fall back to a local homogeneous-coordinate composition
implementing the SAME binding convention from SPEC.md:
``compose(f, g)`` applies f first, then g (result = g∘f), i.e.
``A = A_g @ A_f``, ``b = A_g @ b_f + b_g``.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from pheq.analytic import (
    WFit,
    analytic_operator,
    apply_channel_affine,
    fit_w,
    push_forward_posterior,
)

# ---------------------------------------------------------------------------
# Local photometric algebra (mirrors SPEC.md pheq/color.py; used for test-data
# generation and as a fallback for compose while color.py is written).
# ---------------------------------------------------------------------------

LUMA = torch.tensor([0.299, 0.587, 0.114])


def _brightness_affine(beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    return beta * torch.eye(3), torch.zeros(3)


def _contrast_affine(gamma: float, anchor: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    return gamma * torch.eye(3), (1.0 - gamma) * anchor * torch.ones(3)


def _saturation_affine(s: float) -> tuple[torch.Tensor, torch.Tensor]:
    return s * torch.eye(3) + (1.0 - s) * torch.ones(3, 1) @ LUMA[None, :], torch.zeros(3)


def _hue_affine(theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    t_yiq = torch.tensor(
        [
            [0.299, 0.587, 0.114],
            [0.595716, -0.274453, -0.321263],
            [0.211456, -0.522591, 0.311135],
        ]
    )
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rot = torch.tensor([[1.0, 0.0, 0.0], [0.0, cos_t, -sin_t], [0.0, sin_t, cos_t]])
    return torch.linalg.inv(t_yiq) @ rot @ t_yiq, torch.zeros(3)


def _compose_local(*ops: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """SPEC convention: fold left-to-right; compose(f, g) applies f first (= g∘f)."""
    a_acc, b_acc = torch.eye(3), torch.zeros(3)
    for a_i, b_i in ops:
        b_acc = a_i @ b_acc + b_i
        a_acc = a_i @ a_acc
    return a_acc, b_acc


try:  # pragma: no cover - exercised once color.py lands
    from pheq.color import compose
except ImportError:  # sibling module implemented concurrently
    compose = _compose_local


def _random_wfit(c_lat: int, seed: int) -> WFit:
    gen = torch.Generator().manual_seed(seed)
    w_mat = torch.randn(3, c_lat, generator=gen)
    assert torch.linalg.matrix_rank(w_mat) == 3  # full row rank
    c_vec = torch.randn(3, generator=gen) * 0.3
    return WFit(W=w_mat, c=c_vec, r2=1.0, r2_per_channel=torch.ones(3))


def _decode_linear(z: torch.Tensor, w_mat: torch.Tensor, c_vec: torch.Tensor) -> torch.Tensor:
    """Exact linear pixelwise decoder D(z) = W z + c."""
    return torch.einsum("rc,bchw->brhw", w_mat, z) + c_vec[None, :, None, None]


def _apply_affine(img: torch.Tensor, a_mat: torch.Tensor, b_vec: torch.Tensor) -> torch.Tensor:
    """Pixelwise A @ rgb + b (pre-clip, per SPEC apply_affine default)."""
    return torch.einsum("rs,bshw->brhw", a_mat, img) + b_vec[None, :, None, None]


def _factor_pairs() -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        _brightness_affine(1.3),
        _contrast_affine(0.7),
        _saturation_affine(0.4),
        _hue_affine(0.5),
    ]


# ---------------------------------------------------------------------------
# (1) Homomorphism on the color block
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k_variant", ["I", "0"])
def test_homomorphism_color_block(k_variant: str) -> None:
    """M_{a∘b} ≈ M_a M_b and m_{a∘b} ≈ M_a m_b + m_a for K in {I, 0}.

    a∘b applies b first, then a; with the SPEC/compose convention
    (compose(f, g) = g∘f) the composed element is compose(op_b, op_a).
    """
    torch.manual_seed(0)
    fit = _random_wfit(c_lat=4, seed=1)

    gen = torch.Generator().manual_seed(2)
    random_pair = (
        torch.randn(3, 3, generator=gen) + 2.0 * torch.eye(3),
        torch.randn(3, generator=gen) * 0.2,
    )
    factors = _factor_pairs() + [random_pair]

    for op_a in factors:
        for op_b in factors:
            a_comp, b_comp = compose(op_b, op_a)  # b first, then a  ->  a∘b
            m_a, v_a = analytic_operator(fit, *op_a, K=k_variant)
            m_b, v_b = analytic_operator(fit, *op_b, K=k_variant)
            m_comp, v_comp = analytic_operator(fit, a_comp, b_comp, K=k_variant)
            torch.testing.assert_close(m_comp, m_a @ m_b, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(v_comp, m_a @ v_b + v_a, rtol=1e-4, atol=1e-5)


def test_compose_convention_matches_homogeneous_coordinates() -> None:
    """compose(f, g) must equal g∘f in homogeneous coordinates (SPEC color.py)."""
    op_f = _contrast_affine(0.7)
    op_g = _saturation_affine(0.4)
    a_fg, b_fg = compose(op_f, op_g)
    # Homogeneous 4x4 check: H(g) @ H(f).
    def _homog(a_mat: torch.Tensor, b_vec: torch.Tensor) -> torch.Tensor:
        h_mat = torch.eye(4)
        h_mat[:3, :3] = a_mat
        h_mat[:3, 3] = b_vec
        return h_mat

    h_expected = _homog(*op_g) @ _homog(*op_f)
    torch.testing.assert_close(a_fg, h_expected[:3, :3], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(b_fg, h_expected[:3, 3], rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# (2) Exact-linear-decoder equivariance (load-bearing correctness proof)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("c_lat", [4, 8])
def test_exact_linear_decoder_equivariance(c_lat: int) -> None:
    """D(T_a z) == τ_a(D(z)) exactly for a linear decoder, all K variants.

    The null-space component (I − W⁺W)K must not leak into RGB, so the
    identity holds for K="I", K="0", AND a random full K.
    """
    torch.manual_seed(3)
    fit = _random_wfit(c_lat=c_lat, seed=4)
    gen = torch.Generator().manual_seed(5)
    z = torch.randn(2, c_lat, 6, 5, generator=gen)

    k_random = torch.randn(c_lat, c_lat, generator=gen)
    composed = compose(_saturation_affine(0.4), _hue_affine(-0.6), _brightness_affine(1.3))
    ops = _factor_pairs() + [composed]

    for a_mat, b_vec in ops:
        x_aug = _apply_affine(_decode_linear(z, fit.W, fit.c), a_mat, b_vec)
        for k_variant in ["I", "0", k_random]:
            m_mat, m_vec = analytic_operator(fit, a_mat, b_vec, K=k_variant)
            decoded = _decode_linear(apply_channel_affine(z, m_mat, m_vec), fit.W, fit.c)
            torch.testing.assert_close(decoded, x_aug, rtol=1e-4, atol=1e-4)


def test_analytic_operator_identity_at_identity() -> None:
    """A=I, b=0 with K='I' gives M=I, m=0 (identity element maps to identity)."""
    fit = _random_wfit(c_lat=4, seed=6)
    m_mat, m_vec = analytic_operator(fit, torch.eye(3), torch.zeros(3), K="I")
    torch.testing.assert_close(m_mat, torch.eye(4), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(m_vec, torch.zeros(4), rtol=0.0, atol=1e-6)


def test_analytic_operator_k_validation() -> None:
    fit = _random_wfit(c_lat=4, seed=7)
    with pytest.raises(ValueError):
        analytic_operator(fit, torch.eye(3), torch.zeros(3), K="bogus")
    with pytest.raises(ValueError):
        analytic_operator(fit, torch.eye(3), torch.zeros(3), K=torch.eye(3))  # wrong shape


# ---------------------------------------------------------------------------
# apply_channel_affine: matches an explicit per-site loop
# ---------------------------------------------------------------------------


def test_apply_channel_affine_matches_loop() -> None:
    gen = torch.Generator().manual_seed(8)
    z = torch.randn(2, 4, 3, 2, generator=gen)
    m_mat = torch.randn(4, 4, generator=gen)
    m_vec = torch.randn(4, generator=gen)
    out = apply_channel_affine(z, m_mat, m_vec)
    assert out.shape == z.shape
    for bi in range(z.shape[0]):
        for hi in range(z.shape[2]):
            for wi in range(z.shape[3]):
                expected = m_mat @ z[bi, :, hi, wi] + m_vec
                torch.testing.assert_close(out[bi, :, hi, wi], expected, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# (3) fit_w recovers a planted (W, c)
# ---------------------------------------------------------------------------


def test_fit_w_recovers_planted_map_same_resolution() -> None:
    """Images generated at latent resolution: exact linear relation, r2 ≈ 1."""
    gen = torch.Generator().manual_seed(9)
    n, c_lat, h, w = 8, 4, 12, 12
    w_true = torch.randn(3, c_lat, generator=gen)
    c_true = torch.randn(3, generator=gen) * 0.2
    z = torch.randn(n, c_lat, h, w, generator=gen)
    images = _decode_linear(z, w_true, c_true)

    fit = fit_w(z, images)
    assert fit.W.shape == (3, c_lat)
    assert fit.c.shape == (3,)
    assert fit.r2 > 0.9999
    assert (fit.r2_per_channel > 0.9999).all()
    torch.testing.assert_close(fit.W, w_true, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(fit.c, c_true, rtol=1e-3, atol=1e-3)


def test_fit_w_recovers_planted_map_through_downsampling() -> None:
    """Exercise the area (box-average) downsample path with an exact relation.

    Pointwise-affine maps commute with convex-combination interpolation
    (plan §3.1 commutation lemma; the box average is a convex combination),
    so planting rgb = W z + c at the LATENT resolution — with
    z = W⁺(ds(x) − c) plus null-space noise (which cannot affect RGB since
    W(I − W⁺W) = 0) — keeps the relation exact after fit_w's internal
    downsample.
    """
    gen = torch.Generator().manual_seed(10)
    n, c_lat, h, w, scale = 6, 4, 16, 16, 2
    x = torch.rand(n, 3, h * scale, w * scale, generator=gen)
    x_ds = F.interpolate(x, size=(h, w), mode="area")

    w_true = torch.randn(3, c_lat, generator=gen)
    c_true = torch.randn(3, generator=gen) * 0.1
    w_pinv = torch.linalg.pinv(w_true)
    null_proj = torch.eye(c_lat) - w_pinv @ w_true
    z_color = torch.einsum("cr,brhw->bchw", w_pinv, x_ds - c_true[None, :, None, None])
    z_null = torch.einsum("dc,bchw->bdhw", null_proj, torch.randn(n, c_lat, h, w, generator=gen))
    z = z_color + z_null

    fit = fit_w(z, x)
    assert fit.r2 > 0.999
    assert (fit.r2_per_channel > 0.999).all()
    torch.testing.assert_close(fit.W, w_true, rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(fit.c, c_true, rtol=1e-2, atol=1e-3)


# ---------------------------------------------------------------------------
# (4) push_forward_posterior: marginal std of the pushed Gaussian
# ---------------------------------------------------------------------------


def test_push_forward_posterior_matches_full_covariance_diagonal() -> None:
    """Returned std equals sqrt(diag(M diag(σ²) Mᵀ)) computed explicitly per site."""
    gen = torch.Generator().manual_seed(11)
    mu = torch.randn(2, 4, 3, 3, generator=gen)
    sigma = 0.2 + torch.rand(2, 4, 3, 3, generator=gen)
    m_mat = torch.randn(4, 4, generator=gen)
    m_vec = torch.randn(4, generator=gen)

    mu_p, sigma_p = push_forward_posterior(mu, sigma, m_mat, m_vec)
    torch.testing.assert_close(mu_p, apply_channel_affine(mu, m_mat, m_vec))
    assert (sigma_p > 0).all()
    for bi in range(2):
        for hi in range(3):
            for wi in range(3):
                cov = m_mat @ torch.diag(sigma[bi, :, hi, wi] ** 2) @ m_mat.T
                expected = torch.sqrt(torch.diagonal(cov))
                torch.testing.assert_close(sigma_p[bi, :, hi, wi], expected, rtol=1e-5, atol=1e-6)


def test_push_forward_posterior_diagonal_m_reduces_to_abs() -> None:
    """For diagonal M the marginal std is exactly |M_ii| σ_i."""
    gen = torch.Generator().manual_seed(12)
    mu = torch.randn(1, 4, 2, 2, generator=gen)
    sigma = 0.1 + torch.rand(1, 4, 2, 2, generator=gen)
    diag = torch.tensor([1.5, -0.5, 2.0, -3.0])
    m_mat = torch.diag(diag)
    _, sigma_p = push_forward_posterior(mu, sigma, m_mat, torch.zeros(4))
    torch.testing.assert_close(sigma_p, diag.abs()[None, :, None, None] * sigma, rtol=1e-5, atol=1e-6)


def test_push_forward_posterior_monte_carlo() -> None:
    """Pushed (mean, std) match empirical moments of M(μ + σε) + m over samples."""
    gen = torch.Generator().manual_seed(13)
    b, c_lat, h, w = 2, 4, 3, 3
    n_samples = 200_000
    mu = torch.randn(b, c_lat, h, w, generator=gen)
    sigma = 0.3 + torch.rand(b, c_lat, h, w, generator=gen)
    m_mat = torch.randn(c_lat, c_lat, generator=gen)
    m_vec = torch.randn(c_lat, generator=gen)

    mu_p, sigma_p = push_forward_posterior(mu, sigma, m_mat, m_vec)

    eps = torch.randn(n_samples, b, c_lat, h, w, generator=gen)
    z_samples = mu[None] + sigma[None] * eps
    pushed = torch.einsum("dc,sbchw->sbdhw", m_mat, z_samples) + m_vec[None, None, :, None, None]
    emp_mean = pushed.mean(dim=0)
    emp_std = pushed.std(dim=0)

    torch.testing.assert_close(emp_mean, mu_p, rtol=0.02, atol=0.02)
    torch.testing.assert_close(emp_std, sigma_p, rtol=0.02, atol=0.01)
