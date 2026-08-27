"""Tests for pheq.diffusion (SPEC2 diffusion.py): q_sample statistics, correct
posterior coefficients, the load-bearing overfit smoke, sampler shape/finiteness,
EMA convergence and swap."""

import math

import torch

import pytest

from pheq.diffusion import EMA, GaussianDiffusion
from pheq.dit import dit_tiny


def _perturb(model: torch.nn.Module, scale: float = 0.02, seed: int = 1) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(scale * torch.randn(p.shape, generator=gen))


# ---------------------------------------------------------------- schedule


def test_schedule_buffers():
    diff = GaussianDiffusion(timesteps=1000, beta_start=1e-4, beta_end=2e-2)
    assert diff.betas.shape == (1000,)
    assert math.isclose(float(diff.betas[0]), 1e-4, rel_tol=1e-6)
    assert math.isclose(float(diff.betas[-1]), 2e-2, rel_tol=1e-6)
    # ᾱ_t = Π (1 − β_s), monotone decreasing in (0, 1].
    acp = torch.cumprod(1.0 - diff.betas.double(), dim=0)
    assert torch.allclose(diff.alphas_cumprod, acp.float(), atol=1e-6)
    assert (diff.alphas_cumprod[1:] < diff.alphas_cumprod[:-1]).all()
    assert torch.allclose(
        diff.alphas_cumprod_prev[1:], diff.alphas_cumprod[:-1]
    )
    assert float(diff.alphas_cumprod_prev[0]) == 1.0
    # Buffers are computed in float64 then stored float32, so compare against
    # the float32 recomputation with a small tolerance.
    assert torch.allclose(
        diff.sqrt_alphas_cumprod, diff.alphas_cumprod.sqrt(), atol=1e-6
    )
    assert torch.allclose(
        diff.sqrt_one_minus_alphas_cumprod,
        (1 - diff.alphas_cumprod).sqrt(),
        atol=1e-6,
    )


def test_posterior_coefficients_match_gaussian_algebra():
    """Independent check: q(x_{t-1} | x_t, x0) from the product of the two
    Gaussians q(x_{t-1} | x0) and q(x_t | x_{t-1}) (Ho et al. eq. 6-7)."""
    diff = GaussianDiffusion(timesteps=100)
    betas = diff.betas.double()
    alphas = 1.0 - betas
    acp = torch.cumprod(alphas, dim=0)
    for t in [1, 7, 50, 99]:
        acp_prev = acp[t - 1]
        # Precision of the product of N(sqrt(acp_prev) x0, 1-acp_prev) and
        # the likelihood term from N(sqrt(alpha_t) x_{t-1}, beta_t):
        prec = 1.0 / (1.0 - acp_prev) + alphas[t] / betas[t]
        var = 1.0 / prec
        coef1 = var * acp_prev.sqrt() / (1.0 - acp_prev)  # multiplies x0
        coef2 = var * alphas[t].sqrt() / betas[t]  # multiplies x_t
        assert math.isclose(
            float(diff.posterior_variance[t]), float(var), rel_tol=1e-4
        )
        assert math.isclose(
            float(diff.posterior_mean_coef1[t]), float(coef1), rel_tol=1e-4
        )
        assert math.isclose(
            float(diff.posterior_mean_coef2[t]), float(coef2), rel_tol=1e-4
        )
    # t = 0: posterior collapses to x0 (coef1 = 1, coef2 = 0, var = 0).
    assert math.isclose(float(diff.posterior_mean_coef1[0]), 1.0, rel_tol=1e-5)
    assert float(diff.posterior_mean_coef2[0]) == 0.0
    assert float(diff.posterior_variance[0]) == 0.0


# ---------------------------------------------------------------- q_sample


def test_q_sample_formula():
    diff = GaussianDiffusion()
    gen = torch.Generator().manual_seed(0)
    x0 = torch.randn(3, 4, 8, 8, generator=gen)
    eps = torch.randn(3, 4, 8, 8, generator=gen)
    t = torch.tensor([0, 500, 999])
    out = diff.q_sample(x0, t, eps)
    for i in range(3):
        ti = int(t[i])
        expected = (
            diff.sqrt_alphas_cumprod[ti] * x0[i]
            + diff.sqrt_one_minus_alphas_cumprod[ti] * eps[i]
        )
        assert torch.allclose(out[i], expected, atol=1e-6)


def test_q_sample_statistics():
    """Monte-Carlo moments of q(x_t | x0) for constant x0 (SPEC2)."""
    diff = GaussianDiffusion()
    gen = torch.Generator().manual_seed(0)
    n = 200_000
    c = 0.7
    for ti in [0, 500, 999]:
        x0 = torch.full((n, 1), c)
        eps = torch.randn(n, 1, generator=gen)
        xt = diff.q_sample(x0, torch.full((n,), ti, dtype=torch.long), eps)
        want_mean = float(diff.sqrt_alphas_cumprod[ti]) * c
        want_std = float(diff.sqrt_one_minus_alphas_cumprod[ti])
        # 5-sigma MC tolerance on the mean; 2% on the std.
        assert abs(float(xt.mean()) - want_mean) < 5 * want_std / math.sqrt(n)
        assert abs(float(xt.std()) - want_std) < 0.02 * max(want_std, 1e-3)
    # t → T-1: nearly pure noise (ᾱ ≈ 4e-5).
    assert float(diff.alphas_cumprod[999]) < 1e-3


# ------------------------------------------------------------ training loss


def test_training_loss_at_zero_init_is_eps_power():
    """Zero-init DiT predicts 0 ⇒ loss = E[ε²] ≈ 1 — checks the eps target."""
    torch.manual_seed(0)
    model = dit_tiny(num_classes=0, input_size=8)
    diff = GaussianDiffusion()
    gen = torch.Generator().manual_seed(0)
    x0 = torch.randn(16, 4, 8, 8, generator=gen)
    with torch.no_grad():
        loss = diff.training_loss(model, x0, None, gen)
    assert torch.isfinite(loss)
    assert 0.9 < float(loss) < 1.1


def test_overfit_smoke():
    """LOAD-BEARING (SPEC2): dit_tiny fits a fixed batch — loss decreases
    over 50 optimizer steps. Never skip."""
    torch.manual_seed(0)
    model = dit_tiny(num_classes=0, input_size=8)
    diff = GaussianDiffusion()
    data_gen = torch.Generator().manual_seed(11)
    x0 = torch.randn(8, 4, 8, 8, generator=data_gen)  # FIXED batch

    def eval_loss() -> float:
        gen = torch.Generator().manual_seed(123)
        model.eval()
        with torch.no_grad():
            val = float(diff.training_loss(model, x0, None, gen))
        model.train()
        return val

    before = eval_loss()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    train_gen = torch.Generator().manual_seed(7)
    model.train()
    for _ in range(50):
        opt.zero_grad(set_to_none=True)
        loss = diff.training_loss(model, x0, None, train_gen)
        assert torch.isfinite(loss), "training loss must stay finite"
        loss.backward()
        opt.step()
    after = eval_loss()
    assert after < before, f"loss did not decrease: {before:.4f} -> {after:.4f}"
    assert after < 0.9 * before, (
        f"loss decrease too small: {before:.4f} -> {after:.4f}"
    )


# ---------------------------------------------------------------- sampling


def test_ddpm_sample_shape_finite_deterministic():
    torch.manual_seed(0)
    model = dit_tiny(num_classes=0, input_size=8)
    _perturb(model)
    diff = GaussianDiffusion()
    x = diff.ddpm_sample(
        model, (2, 4, 8, 8), None, "cpu", steps=25,
        gen=torch.Generator().manual_seed(0),
    )
    assert x.shape == (2, 4, 8, 8)
    assert torch.isfinite(x).all()
    x2 = diff.ddpm_sample(
        model, (2, 4, 8, 8), None, "cpu", steps=25,
        gen=torch.Generator().manual_seed(0),
    )
    assert torch.equal(x, x2), "same generator seed must reproduce the sample"


def test_ddpm_sample_restores_train_mode_and_no_grad():
    torch.manual_seed(0)
    model = dit_tiny(num_classes=0, input_size=8)
    model.train()
    diff = GaussianDiffusion()
    x = diff.ddpm_sample(model, (1, 4, 8, 8), None, "cpu", steps=5,
                         gen=torch.Generator().manual_seed(0))
    assert model.training, "sampler must restore the model's train mode"
    assert not x.requires_grad


def test_ddpm_sample_step_bounds():
    diff = GaussianDiffusion(timesteps=10)
    model = dit_tiny(num_classes=0, input_size=8)
    with pytest.raises(ValueError, match="steps"):
        diff.ddpm_sample(model, (1, 4, 8, 8), steps=11)
    with pytest.raises(ValueError, match="steps"):
        diff.ddpm_sample(model, (1, 4, 8, 8), steps=0)


def test_ddpm_sample_steps_one_queries_top_of_chain():
    """steps=1 must be a single x̂₀ jump from t = T-1 (not a t=0 query on
    pure noise that never touches the schedule)."""
    diff = GaussianDiffusion(timesteps=10)
    seen_t = []

    class Probe(torch.nn.Module):
        def forward(self, x, t, y):
            seen_t.extend(t.tolist())
            # Exact eps model for delta data x0 ≡ c.
            ti = int(t[0])
            abar = diff.alphas_cumprod[ti]
            return (x - abar.sqrt() * 0.7) / (1.0 - abar).sqrt()

    x = diff.ddpm_sample(
        Probe(), (4, 2, 4, 4), None, "cpu", steps=1,
        gen=torch.Generator().manual_seed(0),
    )
    assert seen_t == [9] * 4, f"single query must be at t = T-1, got {set(seen_t)}"
    # The x̂₀ plug-in of the exact eps model is identically c = 0.7.
    assert torch.allclose(x, torch.full_like(x, 0.7), atol=1e-5)


def test_ddpm_sample_respaced_marginals_delta_data():
    """Respacing exactness (the classic reuse-per-step-beta bug detector):
    with x0 ≡ c the exact eps model is ε̂ = (x − √ᾱ_t c)/√(1 − ᾱ_t) and the
    plug-in reverse step IS the true reverse chain, so every x the model
    sees at subsequence timestep τ must be ~ N(√ᾱ_τ c, (1 − ᾱ_τ) I). A
    buggy sampler reusing full-chain per-step betas on the subsequence
    drifts by ~2.8 σ-units at these settings; tolerance here is 0.03."""
    diff = GaussianDiffusion()  # T = 1000
    c = 0.7
    seen: dict[int, tuple[float, float]] = {}

    class ExactEps(torch.nn.Module):
        def forward(self, x, t, y):
            ti = int(t[0])
            seen[ti] = (float(x.mean()), float(x.std()))
            abar = diff.alphas_cumprod[ti]
            return (x - abar.sqrt() * c) / (1.0 - abar).sqrt()

    x = diff.ddpm_sample(
        ExactEps(), (100, 4, 16, 16), None, "cpu", steps=50,
        gen=torch.Generator().manual_seed(0),
    )  # 102_400 elements per marginal
    assert len(seen) == 50
    for ti, (mean, std) in seen.items():
        abar = float(diff.alphas_cumprod[ti])
        want_mean = math.sqrt(abar) * c
        want_std = math.sqrt(1.0 - abar)
        assert abs(mean - want_mean) < 0.03, (
            f"t={ti}: marginal mean {mean:.4f} != {want_mean:.4f}"
        )
        assert abs(std - want_std) < 0.03, (
            f"t={ti}: marginal std {std:.4f} != {want_std:.4f}"
        )
    # Final output of the exact model collapses to the delta data.
    assert torch.allclose(x, torch.full_like(x, c), atol=1e-4)


def test_ddpm_sample_conditional_path():
    torch.manual_seed(0)
    model = dit_tiny(num_classes=4, input_size=8)
    _perturb(model)
    diff = GaussianDiffusion()
    y = torch.tensor([0, 3])
    x = diff.ddpm_sample(model, (2, 4, 8, 8), y, "cpu", steps=5,
                         gen=torch.Generator().manual_seed(0))
    assert x.shape == (2, 4, 8, 8)
    assert torch.isfinite(x).all()


# --------------------------------------------------------------------- EMA


def test_ema_default_decay():
    ema = EMA(torch.nn.Linear(2, 2))
    assert ema.decay == 0.9999


def test_ema_converges_toward_params():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4)
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(1.0)
    dists = []
    for _ in range(10):
        ema.update(model)
        gap = (ema.shadow["weight"] - model.weight.detach()).abs().max()
        dists.append(float(gap))
    assert all(b < a for a, b in zip(dists, dists[1:])), "EMA must approach params"
    assert dists[-1] < 1e-2
    # Geometric rate: each update halves the gap at decay 0.5.
    assert abs(dists[1] / dists[0] - 0.5) < 1e-4


def test_ema_copy_to_and_swap():
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 3)
    ema = EMA(model, decay=0.9)
    shadow_w = ema.shadow["weight"].clone()
    with torch.no_grad():
        model.weight.add_(1.0)  # diverge from the shadow
    live_w = model.weight.detach().clone()
    with ema.swap() as m:
        assert m is model
        assert torch.equal(model.weight, shadow_w)
    assert torch.equal(model.weight, live_w), "swap must restore live weights"
    ema.copy_to(model)
    assert torch.equal(model.weight, shadow_w)


def test_ema_swap_restores_on_exception():
    model = torch.nn.Linear(3, 3)
    ema = EMA(model, decay=0.9)
    with torch.no_grad():
        model.weight.add_(1.0)
    live_w = model.weight.detach().clone()
    with pytest.raises(RuntimeError, match="boom"):
        with ema.swap():
            raise RuntimeError("boom")
    assert torch.equal(model.weight, live_w)


def test_ema_state_dict_roundtrip():
    model = torch.nn.Linear(3, 3)
    ema = EMA(model, decay=0.9)
    ema.update(model)
    state = ema.state_dict()
    ema2 = EMA(torch.nn.Linear(3, 3), decay=0.5)
    ema2.load_state_dict(state)
    assert ema2.decay == 0.9
    assert torch.equal(ema2.shadow["weight"], ema.shadow["weight"])
