"""Tests for pheq.lie_operator (SPEC.md "pheq/lie_operator.py", plan §3.3).

NOTE on the concurrent sibling module: ``LieAffineOperator.init_from_analytic``
lazily imports ``pheq.color``, which is being implemented concurrently. If it is
not importable yet, we register a minimal stand-in module implementing the SAME
binding conventions from SPEC.md (pheq/color.py section) into ``sys.modules`` so
these tests run standalone; once the real pheq/color.py lands, it is used
instead (same pattern as tests/test_analytic.py's local fallback).
"""

import math
import sys
import types

import pytest
import torch

from pheq.analytic import WFit, analytic_operator, apply_channel_affine
from pheq.lie_operator import LieAffineOperator, _compose_phi


def _install_color_stub() -> None:
    """Register a SPEC-conforming pheq.color stand-in if the real one is absent."""
    try:
        import pheq.color  # noqa: F401

        return
    except ImportError:
        pass

    mod = types.ModuleType("pheq.color")
    luma = torch.tensor([0.299, 0.587, 0.114])

    def brightness_affine(beta: float):
        return beta * torch.eye(3), torch.zeros(3)

    def contrast_affine(gamma: float, anchor: float = 0.5):
        return gamma * torch.eye(3), (1.0 - gamma) * anchor * torch.ones(3)

    def saturation_affine(s: float):
        return s * torch.eye(3) + (1.0 - s) * torch.ones(3, 1) @ luma[None, :], torch.zeros(3)

    def hue_affine(theta: float):
        t_yiq = torch.tensor(
            [
                [0.299, 0.587, 0.114],
                [0.595716, -0.274453, -0.321263],
                [0.211456, -0.522591, 0.311135],
            ]
        )
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        # Direction convention pinned in pheq.color.hue_affine / test_color.py:
        # positive theta = increasing HSV hue (torchvision-positive).
        rot = torch.tensor([[1.0, 0.0, 0.0], [0.0, cos_t, sin_t], [0.0, -sin_t, cos_t]])
        return torch.linalg.inv(t_yiq) @ rot @ t_yiq, torch.zeros(3)

    mod.brightness_affine = brightness_affine
    mod.contrast_affine = contrast_affine
    mod.saturation_affine = saturation_affine
    mod.hue_affine = hue_affine
    sys.modules["pheq.color"] = mod
    import pheq

    pheq.color = mod


_install_color_stub()

from pheq import color  # noqa: E402  (real module or SPEC-conforming stub)


def _synthetic_fit(seed: int = 2, channels: int = 4) -> WFit:
    """A WFit for a synthetic exactly-linear decoder D(z) = W z (c = 0)."""
    gen = torch.Generator().manual_seed(seed)
    w_mat = torch.randn(3, channels, generator=gen)
    # Full row rank check so the analytic construction is well posed.
    assert torch.linalg.svdvals(w_mat).min() > 1e-3
    return WFit(W=w_mat, c=torch.zeros(3), r2=1.0, r2_per_channel=torch.ones(3))


def test_exact_identity_at_zero():
    """M(0) = I bit-exact via matrix_exp(0); m(0) = 0 by subtraction parameterization."""
    torch.manual_seed(0)
    op = LieAffineOperator()
    phi0 = torch.zeros(4)
    assert torch.equal(op.M(phi0), torch.eye(4))
    assert torch.equal(op.m(phi0), torch.zeros(4))

    z = torch.randn(2, 4, 8, 8)
    assert torch.equal(op(z, phi0), z)

    # Batched zeros go through the same exactness path.
    zb = torch.randn(3, 4, 5, 5)
    phi0b = torch.zeros(3, 4)
    assert torch.equal(op.M(phi0b), torch.eye(4).expand(3, 4, 4))
    assert torch.equal(op(zb, phi0b), zb)


def test_one_parameter_homomorphism():
    """M(phi1 + phi2) = M(phi1) M(phi2) when only one factor is active."""
    torch.manual_seed(1)
    op = LieAffineOperator()
    with torch.no_grad():
        op.G.copy_(0.3 * torch.randn(4, 4, 4))

    for i in range(4):
        e_i = torch.zeros(4)
        e_i[i] = 1.0
        phi1, phi2 = 0.37 * e_i, -0.61 * e_i
        m_sum = op.M(phi1 + phi2)
        m_prod = op.M(phi2) @ op.M(phi1)
        torch.testing.assert_close(m_sum, m_prod, atol=1e-5, rtol=1e-5)
        # Same generator commutes with itself, so the order is irrelevant too.
        torch.testing.assert_close(m_prod, op.M(phi1) @ op.M(phi2), atol=1e-5, rtol=1e-5)


def test_init_from_analytic_matches_analytic():
    """After init, forward() matches the analytic operator for single-factor transforms.

    Synthetic linear decoder D(z) = W z (c = 0); brightness/saturation/hue have
    b = 0 so the analytic m vanishes and the full forward pass must match.
    Magnitudes deliberately differ from the init reference (1.25 / 0.3): K="I"
    makes each factor a one-parameter subgroup, so exp(phi G_i) reproduces the
    analytic M at any magnitude (plan §3.2/§3.3).
    """
    torch.manual_seed(2)
    fit = _synthetic_fit()
    op = LieAffineOperator()
    op.init_from_analytic(fit)

    # Translation MLP is zero-initialized by init_from_analytic (spec resolution).
    assert torch.equal(op.m(torch.randn(4)), torch.zeros(4))

    z = torch.randn(2, 4, 8, 8)
    cases = [
        (0, color.brightness_affine(1.1), math.log(1.1)),
        (2, color.saturation_affine(0.7), math.log(0.7)),
        (3, color.hue_affine(-0.2), -0.2),
    ]
    for i, (a_mat, b_vec), coord in cases:
        m_an, m_vec_an = analytic_operator(fit, a_mat, b_vec, K="I")
        assert m_vec_an.abs().max() < 1e-6  # b = 0, c = 0 -> analytic m = 0
        phi = torch.zeros(4)
        phi[i] = coord
        out = op(z, phi)
        expected = apply_channel_affine(z, m_an, m_vec_an)
        torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)

    # Contrast has b != 0 (its analytic m is nonzero and left to the MLP to
    # learn); the linear part must still match.
    a_mat, b_vec = color.contrast_affine(1.3)
    m_an, _ = analytic_operator(fit, a_mat, b_vec, K="I")
    phi = torch.zeros(4)
    phi[1] = math.log(1.3)
    torch.testing.assert_close(op.M(phi), m_an, atol=1e-4, rtol=1e-4)


def test_gradient_flows_to_generators_and_mlp():
    torch.manual_seed(3)
    op = LieAffineOperator()
    z = torch.randn(2, 4, 6, 6)
    phi = torch.tensor([0.2, -0.1, 0.3, 0.4])
    loss = op(z, phi).pow(2).sum()
    loss.backward()

    assert op.G.grad is not None
    assert op.G.grad.abs().sum() > 0
    mlp_params = list(op.mlp.parameters())
    assert all(p.grad is not None for p in mlp_params)
    # The final bias cancels exactly in m(phi) = mlp(f(phi)) - mlp(f(0)), so
    # only require the weight gradients to be nonzero.
    weight_grads = sum(
        p.grad.abs().sum() for name, p in op.mlp.named_parameters() if "weight" in name
    )
    assert weight_grads > 0


def test_parameter_count():
    """Plan §3.3: the operator must be tiny (no shadow decoder); spec: < 5000 for C=4."""
    op = LieAffineOperator(channels=4)
    n_params = sum(p.numel() for p in op.parameters())
    assert n_params < 5000


def test_batched_phi():
    """phi (B, 4) goes through matrix_exp batched and matches per-sample forward."""
    torch.manual_seed(4)
    op = LieAffineOperator()
    with torch.no_grad():
        op.G.copy_(0.2 * torch.randn(4, 4, 4))
    z = torch.randn(5, 4, 6, 6)
    phi = 0.5 * torch.randn(5, 4)

    out = op(z, phi)
    assert out.shape == z.shape
    for i in range(5):
        single = op(z[i : i + 1], phi[i])
        torch.testing.assert_close(out[i : i + 1], single, atol=1e-6, rtol=1e-6)


def test_composition_loss():
    torch.manual_seed(5)
    op = LieAffineOperator()
    with torch.no_grad():
        op.G.copy_(0.2 * torch.randn(4, 4, 4))
    z = torch.randn(2, 4, 6, 6)
    phi_a = torch.tensor([0.2, 0.0, -0.3, 0.1])
    phi_b = torch.tensor([-0.15, 0.2, 0.1, 0.0])

    # Composing with the identity is exact (identity exact by parameterization).
    assert op.composition_loss(z, phi_a, torch.zeros(4)).detach().item() <= 1e-12
    assert op.composition_loss(z, torch.zeros(4), phi_b).detach().item() <= 1e-12

    # Generic case: nonnegative scalar; z is detached (no grad to the encoder).
    zz = z.clone().requires_grad_(True)
    loss = op.composition_loss(zz, phi_a, phi_b)
    assert loss.dim() == 0
    assert loss.item() >= 0.0
    loss.backward()
    assert zz.grad is None
    assert op.G.grad is not None

    # PhotoParams duck-typing: anything with .phi() works (color.py sibling).
    class _Params:
        def __init__(self, phi: torch.Tensor) -> None:
            self._phi = phi

        def phi(self) -> torch.Tensor:
            return self._phi

    loss_ducked = op.composition_loss(z, _Params(phi_a), _Params(phi_b))
    torch.testing.assert_close(loss_ducked, op.composition_loss(z, phi_a, phi_b))


def _photo_phi(**kwargs) -> torch.Tensor:
    """Canonical coordinates of a PhotoParams element (real color module)."""
    from pheq.color import PhotoParams

    return PhotoParams(**kwargs).phi()


def test_compose_phi_matches_group_composition() -> None:
    """φ(b∘a) from _compose_phi must reproduce color.compose exactly —
    including the cross-factor brightness/contrast case where
    φ(b∘a) ≠ φ_a + φ_b (the composed contrast bias differs: e.g.
    a = contrast(0.7), b = brightness(1.3) has true bias 0.195, while the
    canonical element at φ_a + φ_b has bias 0.15)."""
    from pheq.color import PhotoParams, compose

    cases = [
        (dict(gamma=0.7), dict(beta=1.3)),  # the reviewer counterexample
        (dict(beta=1.3), dict(gamma=0.7)),
        (dict(beta=0.8, gamma=1.2, sat=0.6, hue=0.2), dict(beta=1.3, gamma=0.7, sat=1.4, hue=-0.5)),
        (dict(gamma=0.6), dict(beta=1.4, gamma=0.6)),  # range-extreme pair
    ]
    for kw_a, kw_b in cases:
        pa, pb = PhotoParams(**kw_a), PhotoParams(**kw_b)
        phi_c = _compose_phi(pa.phi(), pb.phi())
        # Rebuild the element at the composed coordinates and compare (A, b)
        # against the homogeneous-coordinate product b∘a.
        pc = PhotoParams(
            beta=math.exp(float(phi_c[0])),
            gamma=math.exp(float(phi_c[1])),
            sat=math.exp(float(phi_c[2])),
            hue=float(phi_c[3]),
        )
        a_true, b_true = compose(pa.affine(torch.float64), pb.affine(torch.float64))
        a_can, b_can = pc.affine(torch.float64)
        torch.testing.assert_close(a_can, a_true, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(b_can, b_true, atol=1e-5, rtol=1e-5)
        # And φ_a + φ_b is NOT the right target whenever brightness and
        # contrast are split across a and b (regression pin on the defect).
        if kw_a == dict(gamma=0.7):
            p_sum = PhotoParams(beta=1.3, gamma=0.7)
            _, b_sum = p_sum.affine(torch.float64)
            assert (b_sum - b_true).abs().max() > 0.04


def test_composition_loss_zero_for_exact_equivariant_operator() -> None:
    """The loss's minimum must include the exactly-equivariant operator: with
    generators from init_from_analytic and m(φ) overridden to the exact
    analytic translation, the operator is a homomorphism (plan §3.2, K = "I"),
    so L_comp ≈ 0 — including the cross brightness/contrast pair that a
    φ_a + φ_b target gets wrong (loss floor ~1e-3 before the fix)."""
    from pheq import color

    fit = _synthetic_fit(seed=2)
    fit = WFit(W=fit.W, c=torch.tensor([0.1, -0.05, 0.02]), r2=1.0, r2_per_channel=torch.ones(3))

    class _ExactAnalytic(LieAffineOperator):
        def m(self, phi: torch.Tensor) -> torch.Tensor:
            if phi.dim() != 1:
                return torch.stack([self.m(row) for row in phi])
            p = color.PhotoParams(
                beta=math.exp(float(phi[0])),
                gamma=math.exp(float(phi[1])),
                sat=math.exp(float(phi[2])),
                hue=float(phi[3]),
            )
            _, m_vec = analytic_operator(fit, *p.affine(), K="I")
            return m_vec

    torch.manual_seed(11)
    op = _ExactAnalytic()
    op.init_from_analytic(fit)
    z = torch.randn(2, 4, 6, 6)

    pairs = [
        (dict(gamma=0.7), dict(beta=1.3)),  # translation-critical cross pair
        (dict(beta=1.2, gamma=0.8), dict(beta=0.7, gamma=1.3)),
        (dict(gamma=0.7, sat=0.5, hue=0.3), dict(beta=1.3, sat=1.2, hue=-0.1)),
    ]
    for kw_a, kw_b in pairs:
        loss = op.composition_loss(z, color.PhotoParams(**kw_a), color.PhotoParams(**kw_b))
        assert loss.item() < 1e-8, (kw_a, kw_b, loss.item())


def test_absorbing_saturation_terminates_and_is_idempotent() -> None:
    """Plan §3.1/§3.3: s = 0 is tested at eval and g_ψ(·, s = 0) must be
    idempotent. PhotoParams(sat=0).phi() carries log(0) = -inf, on which
    torch.matrix_exp never returns — forward() must terminate (via the
    documented finite floor) and, for the analytic-initialized operator,
    match the analytic operator at the absorbing element and be idempotent."""
    fit = _synthetic_fit()
    op = LieAffineOperator()
    op.init_from_analytic(fit)

    phi0 = _photo_phi(sat=0.0)
    assert torch.isneginf(phi0[2])

    z = torch.randn(2, 4, 8, 8, generator=torch.Generator().manual_seed(12))
    z1 = op(z, phi0)  # must terminate (hangs forever without the guard)
    assert torch.isfinite(z1).all()

    # Matches the analytic operator at the absorbing element A_sat(0).
    from pheq import color

    m_an, m_vec_an = analytic_operator(fit, *color.saturation_affine(0.0), K="I")
    expected = apply_channel_affine(z, m_an, m_vec_an)
    torch.testing.assert_close(z1, expected, atol=1e-4, rtol=1e-4)

    # Idempotency: g(g(z, s=0), s=0) == g(z, s=0).
    z2 = op(z1, phi0)
    torch.testing.assert_close(z2, z1, atol=1e-4, rtol=1e-4)


def test_non_finite_phi_raises() -> None:
    """NaN and +inf coordinates have no group-element interpretation and must
    raise instead of hanging matrix_exp / silently producing NaN."""
    op = LieAffineOperator()
    z = torch.randn(1, 4, 4, 4)
    for bad in (float("nan"), float("inf")):
        phi = torch.zeros(4)
        phi[1] = bad
        with pytest.raises(ValueError):
            op.M(phi)
        with pytest.raises(ValueError):
            op(z, phi)


def test_forward_under_bf16_autocast_cpu() -> None:
    # Regression: torch.matrix_exp is not autocast-safe (CUDA bf16 crash,
    # Oscar gpu2105, p2_lie run). M() now computes in an fp32 island with
    # autocast disabled; forward must succeed and stay finite under autocast.
    op = LieAffineOperator(channels=4)
    z = torch.randn(2, 4, 8, 8)
    phi = torch.tensor([[0.1, -0.05, 0.2, 0.3], [0.0, 0.0, 0.0, 0.0]])
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = op(z, phi)
    assert torch.isfinite(out).all()
    m = op.M(phi)
    assert m.dtype == torch.float32  # fp32 island holds regardless of autocast
