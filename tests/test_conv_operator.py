"""Tests for pheq.conv_operator (SPEC2 conv_operator.py; plan §3.3 L2).

Covers: params_from_phi (inverse of PhotoParams.phi(), incl. the absorbing
sat = 0 element); EXACT identity at phi = 0 for arbitrary (trained) weights
via the h(z, phi) - h(z, 0) subtraction; smoothness (finite, nonzero grad
wrt phi near 0); pure-analytic match when h's weights are zeroed; grad flow
to all parameters and to z; the 50K-150K parameter budget; batched phi.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from pheq.analytic import WFit, analytic_operator, apply_channel_affine
from pheq.color import PhotoParams
from pheq.conv_operator import ConvResidualOperator, params_from_phi
from pheq.vae import ToyLinearAE

torch.manual_seed(0)


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def make_wfit() -> WFit:
    """WFit from ToyLinearAE's planted (W, c) — the exact-linear-decoder setting."""
    w, c = ToyLinearAE().true_w()
    return WFit(W=w, c=c, r2=1.0, r2_per_channel=torch.ones(3))


def randomize(op: ConvResidualOperator, seed: int = 0, std: float = 0.1) -> None:
    """Perturb ALL parameters (simulates a trained operator, h != 0)."""
    g = _gen(seed)
    with torch.no_grad():
        for p in op.parameters():
            p.add_(std * torch.randn(p.shape, generator=g))


def _z(b: int = 2, hw: int = 8, seed: int = 1) -> torch.Tensor:
    return torch.randn(b, 4, hw, hw, generator=_gen(seed))


# ---------------------------------------------------------------------------
# params_from_phi
# ---------------------------------------------------------------------------


def test_params_from_phi_roundtrip() -> None:
    p = PhotoParams(beta=1.2, gamma=0.8, sat=0.5, hue=-0.3)
    q = params_from_phi(p.phi())
    assert math.isclose(q.beta, p.beta, rel_tol=1e-6)
    assert math.isclose(q.gamma, p.gamma, rel_tol=1e-6)
    assert math.isclose(q.sat, p.sat, rel_tol=1e-6)
    assert math.isclose(q.hue, p.hue, rel_tol=1e-6)


def test_params_from_phi_identity_exact() -> None:
    q = params_from_phi(torch.zeros(4))
    assert q == PhotoParams(beta=1.0, gamma=1.0, sat=1.0, hue=0.0)


def test_params_from_phi_absorbing_saturation() -> None:
    # PhotoParams(sat=0).phi() has log(0) = -inf; exp(-inf) = 0.0 exactly.
    q = params_from_phi(PhotoParams(sat=0.0).phi())
    assert q.sat == 0.0


def test_params_from_phi_bad_shape_raises() -> None:
    with pytest.raises(ValueError):
        params_from_phi(torch.zeros(3))


# ---------------------------------------------------------------------------
# Exact identity at phi = 0 (REQUIRED, for ALL parameter values)
# ---------------------------------------------------------------------------


def test_identity_at_phi0_exact_random_weights() -> None:
    op = ConvResidualOperator(make_wfit())
    randomize(op)  # "for all time": not just the zero-init state
    z = _z(3)
    assert torch.equal(op(z, torch.zeros(4)), z)
    assert torch.equal(op(z, torch.zeros(3, 4)), z)


def test_identity_at_phi0_exact_at_init() -> None:
    op = ConvResidualOperator(make_wfit())
    z = _z()
    assert torch.equal(op(z, torch.zeros(4)), z)


def test_identity_mixed_batch_zero_rows() -> None:
    """In a batched phi, exactly-zero rows are bit-exact identity row-wise."""
    op = ConvResidualOperator(make_wfit())
    randomize(op)
    z = _z(3)
    phi = torch.zeros(3, 4)
    phi[1] = PhotoParams(beta=1.2, gamma=0.9, sat=0.7, hue=0.2).phi()
    out = op(z, phi)
    assert torch.equal(out[0], z[0])
    assert torch.equal(out[2], z[2])
    assert not torch.allclose(out[1], z[1])


# ---------------------------------------------------------------------------
# Matches the pure analytic operator when h is zero
# ---------------------------------------------------------------------------


_PARAMS = (
    PhotoParams(beta=1.2, gamma=0.9, sat=0.7, hue=0.2),
    PhotoParams(beta=1.3),
    PhotoParams(gamma=0.7),
    PhotoParams(sat=0.05),
    PhotoParams(hue=-math.pi / 4),
)


def _analytic_expected(fit: WFit, z: torch.Tensor, p: PhotoParams) -> torch.Tensor:
    m_mat, m_vec = analytic_operator(fit, *p.affine(), K="I")
    return apply_channel_affine(z, m_mat, m_vec)


def test_matches_analytic_at_init() -> None:
    """Zero-initialized final conv => h == 0 => g is exactly T^an (plan §3.3)."""
    fit = make_wfit()
    op = ConvResidualOperator(fit)
    z = _z()
    for p in _PARAMS:
        phi = p.phi()
        expected = _analytic_expected(fit, z, params_from_phi(phi))
        assert torch.allclose(op(z, phi), expected, atol=1e-6), p


def test_matches_analytic_when_h_zeroed_after_training() -> None:
    fit = make_wfit()
    op = ConvResidualOperator(fit)
    randomize(op)
    with torch.no_grad():  # zero h's output layer only: h == 0 again
        op.conv_out.weight.zero_()
        op.conv_out.bias.zero_()
    z = _z()
    p = _PARAMS[0]
    phi = p.phi()
    expected = _analytic_expected(fit, z, params_from_phi(phi))
    assert torch.allclose(op(z, phi), expected, atol=1e-6)


def test_neg_inf_sat_finite_and_analytic() -> None:
    """The absorbing sat = 0 element: analytic part exact, h conditioning finite."""
    fit = make_wfit()
    phi = PhotoParams(sat=0.0).phi()  # log sat = -inf
    z = _z()
    op0 = ConvResidualOperator(fit)
    expected = _analytic_expected(fit, z, params_from_phi(phi))
    assert torch.allclose(op0(z, phi), expected, atol=1e-6)
    op = ConvResidualOperator(fit)
    randomize(op)
    assert torch.isfinite(op(z, phi)).all()


# ---------------------------------------------------------------------------
# Gradients: flow to parameters and z; finite + nonzero wrt phi near 0
# ---------------------------------------------------------------------------


def _op_with_live_output(seed: int = 3) -> ConvResidualOperator:
    """Operator whose zero-init output conv is perturbed so h != 0.

    At exact zero-init h == 0 identically, so grads to the earlier conv/FiLM
    layers and to phi vanish (everything upstream is multiplied by the zero
    output weights); the smoothness test therefore perturbs conv_out first.
    """
    op = ConvResidualOperator(make_wfit())
    g = _gen(seed)
    with torch.no_grad():
        op.conv_out.weight.add_(0.05 * torch.randn(op.conv_out.weight.shape, generator=g))
        op.conv_out.bias.add_(0.05 * torch.randn(op.conv_out.bias.shape, generator=g))
    return op


def test_grad_flows_to_all_params_and_z() -> None:
    op = _op_with_live_output()
    z = _z().requires_grad_(True)
    phi = PhotoParams(beta=1.2, gamma=0.9, sat=0.7, hue=0.2).phi()
    (op(z, phi) ** 2).sum().backward()
    for name, p in op.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    for mod in (op.conv_in, op.conv_mid, op.conv_out, op.film1[0], op.film2[0]):
        assert mod.weight.grad.abs().sum() > 0
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0  # joint VAE+operator training needs dL/dz


@pytest.mark.parametrize("where", ["at_zero", "near_zero"])
def test_grad_wrt_phi_finite_nonzero(where: str) -> None:
    """Smoothness of the subtraction parameterization (SPEC2 care point).

    A mask-based identity (h * [phi != 0]) would have zero/undefined grad at
    phi = 0; the subtraction h(z, phi) - h(z, 0) must yield finite, nonzero
    d/dphi both AT 0 and near 0.
    """
    op = _op_with_live_output()
    z = _z()
    base = torch.zeros(4)
    if where == "near_zero":
        base = 1e-3 * torch.randn(4, generator=_gen(4))
    phi = base.clone().requires_grad_(True)
    op(z, phi).sum().backward()
    assert phi.grad is not None
    assert torch.isfinite(phi.grad).all()
    assert phi.grad.abs().sum() > 0


def test_continuity_at_zero() -> None:
    """g(z, eps) stays within O(eps) of z (no jump at the phi = 0 short-circuit)."""
    op = _op_with_live_output()
    z = _z()
    out = op(z, 1e-5 * torch.ones(4))
    assert (out - z).abs().max() < 1e-3


# ---------------------------------------------------------------------------
# Size, batching, validation
# ---------------------------------------------------------------------------


def test_param_count_in_budget() -> None:
    op = ConvResidualOperator(make_wfit())
    n = sum(p.numel() for p in op.parameters())
    assert 50_000 <= n <= 150_000, n  # SPEC2: ~90K, accepted range 50K-150K


def test_buffers_frozen_not_parameters() -> None:
    op = ConvResidualOperator(make_wfit())
    param_names = {name for name, _ in op.named_parameters()}
    for buf in ("W", "c", "freqs", "r2_per_channel"):
        assert buf not in param_names
    assert not op.W.requires_grad and not op.c.requires_grad


def test_batched_phi_matches_per_sample() -> None:
    op = ConvResidualOperator(make_wfit())
    randomize(op)
    z = _z(3)
    phi = torch.stack([p.phi() for p in _PARAMS[:3]])
    out = op(z, phi)
    for i in range(3):
        single = op(z[i : i + 1], phi[i])
        # atol 1e-4: conv kernels reduce in a batch-size-dependent order
        # (B=3 vs B=1 GEMM paths differ by ~5e-6 in float32); the math is
        # per-sample, only the summation order changes.
        assert torch.allclose(out[i], single[0], atol=1e-4), i


def test_shared_phi_matches_expanded_batch() -> None:
    op = ConvResidualOperator(make_wfit())
    randomize(op)
    z = _z(3)
    phi = _PARAMS[0].phi()
    assert torch.allclose(op(z, phi), op(z, phi.expand(3, 4)), atol=1e-6)


def test_accepts_photoparams_directly() -> None:
    op = ConvResidualOperator(make_wfit())
    z = _z()
    p = _PARAMS[0]
    assert torch.allclose(op(z, p), op(z, p.phi()), atol=0.0)


def test_validation_errors() -> None:
    op = ConvResidualOperator(make_wfit())
    z = _z()
    with pytest.raises(ValueError):
        op(z, torch.zeros(3))  # bad phi length
    with pytest.raises(ValueError):
        op(z, torch.zeros(5, 4))  # batch mismatch
    with pytest.raises(ValueError):
        op(torch.randn(1, 3, 8, 8), torch.zeros(4))  # wrong channel count
    with pytest.raises(ValueError):
        ConvResidualOperator(make_wfit(), hidden=60)  # GroupNorm(8) divisibility


def test_hidden_and_nfreq_configurable() -> None:
    op = ConvResidualOperator(make_wfit(), hidden=32, n_freq=4)
    z = _z()
    assert torch.equal(op(z, torch.zeros(4)), z)
    assert op(z, _PARAMS[0].phi()).shape == z.shape
