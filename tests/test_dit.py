"""Tests for pheq.dit (SPEC2 dit.py): shapes, adaLN-Zero init, class dropout,
param counts, grad flow."""

import torch

import pytest

from pheq.dit import DiT, dit_s, dit_tiny


def _perturb(model: torch.nn.Module, scale: float = 0.02, seed: int = 1) -> None:
    """Add small noise to EVERY parameter so zero-init layers become active."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(scale * torch.randn(p.shape, generator=gen))


def test_forward_shape_unconditional():
    torch.manual_seed(0)
    model = dit_tiny(num_classes=0, input_size=8)
    z = torch.randn(2, 4, 8, 8)
    t = torch.randint(0, 1000, (2,))
    out = model(z, t)
    assert out.shape == (2, 4, 8, 8)


def test_forward_shape_conditional():
    torch.manual_seed(0)
    model = dit_tiny(num_classes=10, input_size=8)
    z = torch.randn(3, 4, 8, 8)
    t = torch.randint(0, 1000, (3,))
    y = torch.tensor([0, 5, 9])
    out = model(z, t, y)
    assert out.shape == (3, 4, 8, 8)


def test_output_exactly_zero_at_init():
    """adaLN-Zero: zero-init modulation AND final linear ⇒ output EXACTLY 0."""
    torch.manual_seed(0)
    for model, y in [
        (dit_tiny(num_classes=0, input_size=8), None),
        (dit_tiny(num_classes=5, input_size=8), torch.tensor([1, 3])),
    ]:
        model.eval()
        z = torch.randn(2, 4, 8, 8)
        t = torch.randint(0, 1000, (2,))
        out = model(z, t, y)
        assert (out == 0).all(), "adaLN-Zero init must give exactly-zero output"


def test_class_dropout_all_dropped_maps_to_null():
    """class_dropout=1.0 in train mode: every label → the learned null class."""
    torch.manual_seed(0)
    model = dit_tiny(num_classes=10, input_size=8, class_dropout=1.0)
    _perturb(model)
    model.train()
    z = torch.randn(2, 4, 8, 8)
    t = torch.full((2,), 100, dtype=torch.long)
    out_a = model(z, t, torch.tensor([0, 1]))
    out_b = model(z, t, torch.tensor([7, 9]))
    assert torch.equal(out_a, out_b)
    # And it matches feeding the null index directly in eval mode.
    model.eval()
    out_null = model(z, t, torch.full((2,), 10, dtype=torch.long))
    assert torch.allclose(out_a, out_null, atol=1e-6)


def test_class_dropout_off_in_eval_mode():
    torch.manual_seed(0)
    model = dit_tiny(num_classes=10, input_size=8, class_dropout=1.0)
    _perturb(model)
    model.eval()
    z = torch.randn(2, 4, 8, 8)
    t = torch.full((2,), 100, dtype=torch.long)
    out_a = model(z, t, torch.tensor([0, 1]))
    out_b = model(z, t, torch.tensor([7, 9]))
    assert not torch.allclose(out_a, out_b), "labels must matter in eval mode"


def test_unconditional_ignores_labels():
    """num_classes=0 → label input ignored entirely (SPEC2)."""
    torch.manual_seed(0)
    model = dit_tiny(num_classes=0, input_size=8)
    _perturb(model)
    model.eval()
    z = torch.randn(2, 4, 8, 8)
    t = torch.full((2,), 42, dtype=torch.long)
    assert torch.equal(model(z, t, None), model(z, t, torch.tensor([3, 9])))


def test_conditional_requires_labels():
    model = dit_tiny(num_classes=10, input_size=8)
    z = torch.randn(2, 4, 8, 8)
    t = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError, match="labels"):
        model(z, t, None)


def test_wrong_spatial_size_raises():
    model = dit_tiny(num_classes=0, input_size=8)
    with pytest.raises(ValueError, match="latents"):
        model(torch.randn(1, 4, 16, 16), torch.zeros(1, dtype=torch.long))


def test_param_count_dit_s():
    """DiT-S/2 ≈ 33M params (SPEC2: assert 30–36M)."""
    n = sum(p.numel() for p in dit_s(num_classes=0).parameters())
    assert 30e6 < n < 36e6, f"dit_s has {n / 1e6:.1f}M params"


def test_param_count_dit_tiny_small():
    n = sum(p.numel() for p in dit_tiny(num_classes=0, input_size=8).parameters())
    assert n < 5e6, f"dit_tiny has {n / 1e6:.1f}M params"


def test_dit_s_architecture():
    """Faithful DiT-S/2 hyperparameters: p=2, hidden 384, depth 12, heads 6."""
    model = dit_s(num_classes=0)
    assert model.patch_size == 2
    assert model.hidden_size == 384
    assert len(model.blocks) == 12
    assert model.blocks[0].attn.num_heads == 6
    assert model.num_patches == 256  # 32/2 squared


def test_timestep_conditioning_matters():
    torch.manual_seed(0)
    model = dit_tiny(num_classes=0, input_size=8)
    _perturb(model)
    model.eval()
    z = torch.randn(2, 4, 8, 8)
    out_a = model(z, torch.full((2,), 10, dtype=torch.long))
    out_b = model(z, torch.full((2,), 900, dtype=torch.long))
    assert not torch.allclose(out_a, out_b)


def test_grad_flow():
    """After perturbing zero-init layers, gradients reach every parameter."""
    torch.manual_seed(0)
    model = dit_tiny(num_classes=0, input_size=8)
    _perturb(model)
    z = torch.randn(2, 4, 8, 8)
    t = torch.randint(0, 1000, (2,))
    loss = model(z, t).pow(2).mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"
    qkv_grad = model.blocks[0].attn.qkv.weight.grad
    assert qkv_grad.abs().sum() > 0, "grad must flow into attention weights"
    assert model.patch_embed.weight.grad.abs().sum() > 0


def test_unpatchify_inverts_patchify_layout():
    """unpatchify places each p×p patch back at its own grid site."""
    model = DiT(input_size=4, patch_size=2, in_channels=1, hidden_size=8,
                depth=1, num_heads=1, num_classes=0)
    # Token k gets constant value k → output must be blockwise-constant.
    tokens = torch.arange(4, dtype=torch.float32)[None, :, None].expand(1, 4, 4)
    out = model.unpatchify(tokens)
    assert out.shape == (1, 1, 4, 4)
    expected = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0],
         [0.0, 0.0, 1.0, 1.0],
         [2.0, 2.0, 3.0, 3.0],
         [2.0, 2.0, 3.0, 3.0]]
    )
    assert torch.equal(out[0, 0], expected)


def test_dit_b_param_count() -> None:
    # DiT-B/2 (hidden 768, depth 12, heads 12): ~130M params, the
    # convergence-pair scale (docs/plan-3month.md hero figure).
    from pheq.dit import dit_b

    model = dit_b(num_classes=0)
    n = sum(p.numel() for p in model.parameters())
    assert 110e6 < n < 150e6, n
    out = model(torch.randn(2, 4, 32, 32), torch.tensor([1, 2]), None)
    assert out.shape == (2, 4, 32, 32)
