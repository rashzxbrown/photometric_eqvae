"""DDPM diffusion process + EMA for the DiT proxy (SPEC2 diffusion.py).

Standard DDPM (Ho et al. 2020) with a linear beta schedule 1e-4 → 2e-2,
T = 1000, eps-prediction training loss, and an ancestral sampler over an
evenly-spaced timestep subsequence (250 steps by default). Sampling is
cfg-FREE — no guidance — matching the EQ-VAE anchor protocol (plan §5 Tier 2:
"CFG-free sampling, 250 DDPM steps"; SPEC2 diffusion.py).

Schedule buffers are registered on the module (float64 math, float32 storage)
so ``.to(device)`` moves them with the object.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn.functional as F
from torch import nn


def _extract(buf: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
    """Gather per-timestep coefficients and reshape for (B, ...) broadcast."""
    return buf.gather(0, t).view(-1, *([1] * (ndim - 1)))


class GaussianDiffusion(nn.Module):
    """DDPM forward/reverse process with linear betas (SPEC2 diffusion.py).

    Buffers (all (T,), float32, computed in float64):
        betas, alphas_cumprod, alphas_cumprod_prev,
        sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
        posterior_variance      = β_t (1 − ᾱ_{t−1}) / (1 − ᾱ_t),
        posterior_mean_coef1    = β_t √ᾱ_{t−1} / (1 − ᾱ_t)      (× x̂₀),
        posterior_mean_coef2    = √α_t (1 − ᾱ_{t−1}) / (1 − ᾱ_t) (× x_t),
    the exact q(x_{t−1} | x_t, x₀) posterior coefficients (Ho et al. eq. 7).

    Args:
        timesteps: T (default 1000).
        beta_start / beta_end: linear schedule endpoints (1e-4 → 2e-2).
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        super().__init__()
        if timesteps < 1:
            raise ValueError(f"timesteps must be >= 1, got {timesteps}")
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        acp_prev = torch.cat([torch.ones(1, dtype=torch.float64), acp[:-1]])

        def reg(name: str, x: torch.Tensor) -> None:
            self.register_buffer(name, x.to(torch.float32))

        reg("betas", betas)
        reg("alphas_cumprod", acp)
        reg("alphas_cumprod_prev", acp_prev)
        reg("sqrt_alphas_cumprod", acp.sqrt())
        reg("sqrt_one_minus_alphas_cumprod", (1.0 - acp).sqrt())
        reg("posterior_variance", betas * (1.0 - acp_prev) / (1.0 - acp))
        reg("posterior_mean_coef1", betas * acp_prev.sqrt() / (1.0 - acp))
        reg("posterior_mean_coef2", alphas.sqrt() * (1.0 - acp_prev) / (1.0 - acp))

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        """Forward-noise x₀ at timesteps t: √ᾱ_t x₀ + √(1 − ᾱ_t) ε.

        Args:
            x0: (B, ...) clean data.
            t: (B,) int64 timesteps in [0, T).
            eps: (B, ...) standard normal noise, same shape as x0.

        Returns:
            (B, ...) noised sample x_t.
        """
        nd = x0.dim()
        return (
            _extract(self.sqrt_alphas_cumprod, t, nd) * x0
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, nd) * eps
        )

    def training_loss(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        y: torch.Tensor | None = None,
        gen: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Simple DDPM loss: MSE(ε̂, ε) at uniformly sampled t (Ho et al. eq. 14).

        t and ε are drawn on ``gen``'s device (CPU generator ⇒ the SAME t/ε
        across accelerators) and moved to ``x0.device``. Caveat: this pins
        only the noise drawn HERE — a conditional model in train mode draws
        its class-dropout mask (``dit.LabelEmbedder``) from the GLOBAL RNG on
        ``y.device``, which this generator does not control; for fully
        deterministic conditional losses seed ``torch.manual_seed`` too (or
        run the model in eval mode).

        Args:
            model: eps-predictor with signature model(x_t, t, y).
            x0: (B, C, h, w) clean latents.
            y: labels forwarded to the model (None for unconditional).
            gen: optional torch.Generator for deterministic t/ε.

        Returns:
            Scalar MSE loss.
        """
        b = x0.shape[0]
        sample_device = gen.device if gen is not None else x0.device
        t = torch.randint(
            0, self.timesteps, (b,), generator=gen, device=sample_device
        ).to(x0.device)
        eps = torch.randn(
            x0.shape, generator=gen, device=sample_device, dtype=x0.dtype
        ).to(x0.device)
        x_t = self.q_sample(x0, t, eps)
        eps_hat = model(x_t, t, y)
        return F.mse_loss(eps_hat, eps)

    @torch.no_grad()
    def ddpm_sample(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        y: torch.Tensor | None = None,
        device: torch.device | str | None = None,
        steps: int = 250,
        gen: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Ancestral DDPM sampling over an evenly-spaced subsequence (SPEC2).

        The subsequence τ = round(linspace(0, T−1, steps)) (includes 0 and
        T−1; the degenerate ``steps == 1`` case uses τ = [T−1] — a single
        x̂₀ jump from pure noise at the top of the chain, the standard
        1-step convention) defines a respaced chain with
        β'_i = 1 − ᾱ_{τ_i}/ᾱ_{τ_{i−1}};
        each reverse step uses the exact posterior mean/variance of that chain
        with x̂₀ = (x − √(1 − ᾱ) ε̂)/√ᾱ (latents are NOT clamped — they are
        unbounded, unlike pixel data). cfg-FREE: no guidance, matching the
        EQ-VAE anchor protocol (plan §5 Tier 2). The model is put in eval mode
        for the duration (class dropout off) and restored afterwards.

        Args:
            model: eps-predictor model(x_t, t, y); t values are the ORIGINAL
                chain indices τ_i, so timestep embeddings match training.
            shape: output shape (B, C, h, w).
            y: labels forwarded to the model (None for unconditional).
            device: device for the sample (default: schedule buffers' device).
            steps: subsequence length (1 ≤ steps ≤ T; default 250).
            gen: optional torch.Generator for deterministic sampling; noise is
                drawn on its device and moved to ``device``.

        Returns:
            (B, C, h, w) sample x₀.
        """
        if not 1 <= steps <= self.timesteps:
            raise ValueError(
                f"steps must be in [1, {self.timesteps}], got {steps}"
            )
        if device is None:
            device = self.betas.device
        sample_device = gen.device if gen is not None else device
        if steps == 1:
            # linspace(0, T-1, 1) = [0] would query the model at t = 0 on
            # pure N(0, 1) noise, never touching the schedule. The standard
            # 1-step convention is a single x̂₀ jump from t = T-1.
            taus = torch.tensor([self.timesteps - 1], dtype=torch.long)
        else:
            taus = torch.linspace(0, self.timesteps - 1, steps).round().to(torch.long)
            taus = torch.unique_consecutive(taus)  # dedupe only when steps ≈ T
        acp = self.alphas_cumprod.to(device)
        x = torch.randn(shape, generator=gen, device=sample_device).to(device)
        was_training = model.training
        model.eval()
        try:
            for i in range(len(taus) - 1, -1, -1):
                t_val = int(taus[i])
                abar = acp[t_val]
                abar_prev = (
                    acp[int(taus[i - 1])]
                    if i > 0
                    else torch.ones((), device=device)
                )
                beta = 1.0 - abar / abar_prev
                alpha = 1.0 - beta
                t_batch = torch.full(
                    (shape[0],), t_val, device=device, dtype=torch.long
                )
                eps_hat = model(x, t_batch, y)
                x0_hat = (x - (1.0 - abar).sqrt() * eps_hat) / abar.sqrt()
                mean = (beta * abar_prev.sqrt() / (1.0 - abar)) * x0_hat + (
                    alpha.sqrt() * (1.0 - abar_prev) / (1.0 - abar)
                ) * x
                if i > 0:
                    var = beta * (1.0 - abar_prev) / (1.0 - abar)
                    noise = torch.randn(
                        shape, generator=gen, device=sample_device
                    ).to(device)
                    x = mean + var.sqrt() * noise
                else:
                    x = mean  # final step: posterior collapses to x̂₀
        finally:
            model.train(was_training)
        return x


class EMA:
    """Exponential moving average of model parameters (SPEC2 diffusion.py).

    Tracks PARAMETERS only (DiT's sole buffer is the fixed sin-cos pos-embed,
    which never changes). ``swap`` temporarily loads the EMA weights into the
    model (for sampling/eval) and restores the training weights on exit, even
    on exception.

    Spec resolution: SPEC2 writes ``swap()`` with no argument — the model is
    captured at construction; ``update``/``copy_to``/``swap`` accept an
    explicit model override for flexibility.

    Args:
        model: source of the shadowed parameters.
        decay: EMA decay (default 0.9999, the DiT recipe).
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self.decay = decay
        self._model = model
        self.shadow: dict[str, torch.Tensor] = {
            name: p.detach().clone() for name, p in model.named_parameters()
        }

    @torch.no_grad()
    def update(self, model: nn.Module | None = None) -> None:
        """shadow ← decay · shadow + (1 − decay) · params."""
        model = self._model if model is None else model
        for name, p in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(
                p.detach(), alpha=1.0 - self.decay
            )

    @torch.no_grad()
    def copy_to(self, model: nn.Module | None = None) -> None:
        """Copy shadow parameters into the model (in place)."""
        model = self._model if model is None else model
        for name, p in model.named_parameters():
            p.copy_(self.shadow[name].to(p.device))

    @contextmanager
    def swap(self, model: nn.Module | None = None) -> Iterator[nn.Module]:
        """Context manager: EMA weights inside, training weights restored after."""
        model = self._model if model is None else model
        backup = {
            name: p.detach().clone() for name, p in model.named_parameters()
        }
        self.copy_to(model)
        try:
            yield model
        finally:
            with torch.no_grad():
                for name, p in model.named_parameters():
                    p.copy_(backup[name])

    def state_dict(self) -> dict:
        """Checkpointable state (shadow tensors + decay) for train_dit resume."""
        return {
            "decay": self.decay,
            "shadow": {k: v.clone() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from :meth:`state_dict` output."""
        self.decay = float(state["decay"])
        for k, v in state["shadow"].items():
            self.shadow[k].copy_(v.to(self.shadow[k].device))
