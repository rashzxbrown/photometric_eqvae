"""Tests for pheq/oracle.py (SPEC.md "pheq/oracle.py" section).

Positive control for RQ0's methodology: on the exactly-linear ToyLinearAE,
decoder inversion of an affine-transformed target must succeed
(final_loss < 1e-4) and the oracle's latent edit must be channel-affine
(oracle_affine_fit R^2 > 0.99), because with a linear decoder the true edit
IS affine (plan section 3.2 closed form).

The color affine (A, b) is built locally rather than via pheq.color so this
file does not depend on the concurrently developed sibling module.
"""

import torch
import torch.nn.functional as F

from pheq.oracle import OracleAffineFit, OracleResult, invert_latent, oracle_affine_fit
from pheq.vae import ToyLinearAE

SEED = 7


def _gen() -> torch.Generator:
    return torch.Generator().manual_seed(SEED)


def _apply_color_affine(img: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pointwise RGB map A x + b (local stand-in for pheq.color.apply_affine)."""
    return torch.einsum("dc,bchw->bdhw", a, img) + b.view(1, 3, 1, 1)


def _block_constant_images(n: int, size: int, gen: torch.Generator) -> torch.Tensor:
    """Random block-constant images: exactly in ToyLinearAE's decoder range,
    and closed under ANY pointwise color-affine map (flat blocks stay flat),
    so the inversion optimum is exactly zero loss."""
    colors = torch.rand((n, 3, size // 2, size // 2), generator=gen) * 0.6 + 0.2
    return F.interpolate(colors, scale_factor=2, mode="nearest")


def test_oracle_positive_control_channel_mixing_affine() -> None:
    # The SPEC's headline test: linear decoder + affine-transformed target
    # => inversion converges (final_loss < 1e-4) and the latent edit is
    # channel-affine (R^2 > 0.99).
    gen = _gen()
    ae = ToyLinearAE()
    a = torch.tensor(
        [
            [0.90, 0.15, 0.05],
            [0.10, 0.80, 0.10],
            [0.05, 0.20, 0.85],
        ]
    )  # generic full matrix: saturation/hue-like channel mixing
    b = torch.tensor([0.02, -0.01, 0.03])

    x = _block_constant_images(6, 16, gen)
    x_aug = _apply_color_affine(x, a, b)
    z_init = ae.encode(x)

    result = invert_latent(ae.decode, x_aug, z_init)
    assert isinstance(result, OracleResult)
    assert result.final_loss < 1e-4
    assert torch.allclose(ae.decode(result.z_opt), x_aug, atol=1e-3)

    fit = oracle_affine_fit([(z_init, result.z_opt)])
    assert isinstance(fit, OracleAffineFit)
    assert fit.r2 > 0.99
    # On block-constant images the color block of the true latent edit equals
    # A itself (z_rgb = 2*(rgb - c) with the planted c, so the conjugation by
    # the encoder/decoder color scaling cancels).
    assert torch.allclose(fit.M[:3, :3], a, atol=0.02)


def test_oracle_positive_control_diagonal_affine_all_channels() -> None:
    # Diagonal A keeps the detail (checkerboard) channel inside the decoder
    # range, so all 4 latent channels are exercised; the true latent edit is
    # M = gamma * I_4 exactly.
    gen = _gen()
    ae = ToyLinearAE()
    gamma = 1.3
    z_src = torch.rand((6, 4, 8, 8), generator=gen) * 0.8
    x = ae.decode(z_src)
    x_aug = gamma * x + 0.05
    z_init = ae.encode(x)

    result = invert_latent(ae.decode, x_aug, z_init)
    assert result.final_loss < 1e-4

    fit = oracle_affine_fit([(z_init, result.z_opt)])
    assert fit.r2 > 0.99
    assert float(fit.r2_per_channel.min()) > 0.99
    assert torch.allclose(fit.M, gamma * torch.eye(4), atol=0.02)


def test_invert_latent_freezes_decoder_and_preserves_inputs() -> None:
    gen = _gen()
    ae = ToyLinearAE()
    for p in ae.decoder.parameters():
        p.requires_grad_(True)  # so we can observe the freeze/restore cycle
    params_before = [p.detach().clone() for p in ae.decoder.parameters()]

    x = _block_constant_images(2, 8, gen)
    x_aug = 1.2 * x + 0.01
    z_init = ae.encode(x)
    z_init_copy = z_init.detach().clone()

    result = invert_latent(ae.decode, x_aug, z_init, steps=50, lr=0.1, log_every=10)

    # Decoder params: values untouched, requires_grad flags restored.
    for p, before in zip(ae.decoder.parameters(), params_before):
        assert torch.equal(p.detach(), before)
        assert p.requires_grad
    # z_init untouched; optimization worked on a leaf clone, grad flowed to z.
    assert torch.equal(z_init, z_init_copy)
    assert not z_init.requires_grad
    assert result.z_opt.shape == z_init.shape
    assert not result.z_opt.requires_grad
    assert not torch.equal(result.z_opt, z_init)
    assert result.final_loss < result.losses[0]


def test_invert_latent_loss_trace_and_bare_callable() -> None:
    gen = _gen()
    ae = ToyLinearAE()
    x = _block_constant_images(2, 8, gen)
    z_init = ae.encode(x)

    # Bare closure decode_fn (no discoverable params) must work too.
    result = invert_latent(lambda z: ae.decode(z), 1.1 * x, z_init, steps=100, lr=0.1, log_every=25)
    # Recorded at steps 0, 25, 50, 75, plus the final loss.
    assert len(result.losses) == 100 // 25 + 1
    assert result.final_loss == result.losses[-1]
    assert result.final_loss < result.losses[0]


def test_invert_latent_bare_closure_never_writes_decoder_grads() -> None:
    """The frozen-decoder contract must hold for bare closures too: when
    decode_fn is a lambda over an UNFROZEN module (no discoverable params, so
    the requires_grad freeze cannot apply), backward(inputs=[z]) must still
    leave every decoder parameter's .grad untouched (None) — a stale .grad
    would silently corrupt the caller's next optimizer.step()."""
    from pheq.vae import ToyConvAE

    gen = _gen()
    ae = ToyConvAE(seed=0)  # parameters require grad by default (unfrozen)
    assert all(p.requires_grad for p in ae.parameters())
    x = torch.rand((2, 3, 8, 8), generator=gen)
    z_init = ae.encode(x).detach()

    result = invert_latent(lambda z: ae.decode(z), 1.1 * x, z_init, steps=5, lr=0.05)

    assert result.z_opt.shape == z_init.shape
    for p in ae.parameters():
        assert p.grad is None, "decoder parameter .grad leaked from invert_latent"
    assert all(p.requires_grad for p in ae.parameters())  # flags untouched


def test_oracle_affine_fit_recovers_planted_map_exactly() -> None:
    gen = _gen()
    z = torch.randn((5, 4, 8, 8), generator=gen)
    m_true = torch.eye(4) + 0.3 * torch.randn((4, 4), generator=gen)
    v_true = 0.1 * torch.randn(4, generator=gen)
    z_opt = torch.einsum("dc,bchw->bdhw", m_true, z) + v_true.view(1, 4, 1, 1)

    # Mixed pair shapes: batched and single (C, h, w).
    fit = oracle_affine_fit([(z[:3], z_opt[:3]), (z[3], z_opt[3]), (z[4], z_opt[4])])
    assert fit.r2 > 0.9999
    assert torch.allclose(fit.M, m_true, atol=1e-4)
    assert torch.allclose(fit.m, v_true, atol=1e-4)
    assert fit.r2_per_channel.shape == (4,)
