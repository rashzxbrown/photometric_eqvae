"""Decoder-inversion oracle (RQ0): expressivity ceiling and affine-fit test.

Implements the ``pheq/oracle.py`` section of SPEC.md, operationalizing RQ0 of
docs/research-plan.md ("Decoder expressivity — the premise"): optimize a
latent ``z'`` on the FROZEN autoencoder to minimize ``L(τ_a(x), D(z'))``.
The resulting oracle equivariance error is the decoder's expressivity
ceiling; the R² of a channel-affine fit from the original latents to the
oracle latents tests the analytic-form hypothesis of plan §3.2
(``T_a(z_p) = M_a z_p + m_a``) before any training.

The least-squares fit here mirrors the ``fit_w`` lstsq pattern of
pheq/analytic.py (augmented ``[z; 1]`` design, per-channel R²); it is
implemented locally because the sibling module is developed concurrently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from pheq.analytic import lstsq_minnorm


@dataclass
class OracleResult:
    """Result of :func:`invert_latent`.

    Attributes:
        z_opt: optimized latent, detached, same shape as ``z_init``.
        losses: loss recorded at step 0, every ``log_every`` steps thereafter,
            plus the final post-optimization loss as the last entry.
        final_loss: loss of ``decode_fn(z_opt)`` vs the target after the last
            step (equals ``losses[-1]``).
    """

    z_opt: torch.Tensor
    losses: list[float]
    final_loss: float


@dataclass
class OracleAffineFit:
    """Channel-affine fit ``z_opt ≈ M z_orig + m`` from :func:`oracle_affine_fit`.

    Attributes:
        M: ``(C, C)`` full channel-mixing matrix (never diagonalized; plan §3.2).
        m: ``(C,)`` translation.
        r2: variance-weighted total R² pooled over channels
            (``1 - Σ_c SS_res,c / Σ_c SS_tot,c``) — robust to latent channels
            with (near-)constant targets, e.g. an untouched null-space channel.
        r2_per_channel: ``(C,)`` per-channel R² (1.0 where the target channel
            is constant and exactly predicted, 0.0 where constant but missed).
    """

    M: torch.Tensor
    m: torch.Tensor
    r2: float
    r2_per_channel: torch.Tensor


def _decoder_parameters(decode_fn: Callable[..., torch.Tensor]) -> list[torch.nn.Parameter]:
    """Best-effort discovery of the parameters behind ``decode_fn``.

    Checks, in order: ``decode_fn`` itself being an ``nn.Module``; a bound
    method's ``__self__``; a ``module`` attribute (attached e.g. by
    :func:`pheq.vae.load_sd_vae` to its ``decode_latents`` closure). Returns
    ``[]`` for a bare closure — still safe, because :func:`invert_latent`
    computes gradients with ``backward(inputs=[z])``, which never writes
    ``.grad`` on decoder parameters even when they are undiscoverable here;
    the ``requires_grad_(False)`` freeze is an optimization, not the safety
    mechanism.
    """
    for candidate in (decode_fn, getattr(decode_fn, "__self__", None), getattr(decode_fn, "module", None)):
        if isinstance(candidate, torch.nn.Module):
            return list(candidate.parameters())
    return []


def _resolve_loss(
    loss: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Resolve the ``loss`` argument: ``'l2'`` (MSE), ``'l1'``, or a callable."""
    if callable(loss):
        return loss
    if loss == "l2":
        return lambda pred, target: torch.mean((pred - target) ** 2)
    if loss == "l1":
        return lambda pred, target: torch.mean(torch.abs(pred - target))
    raise ValueError(f"unknown loss {loss!r}; expected 'l2', 'l1', or a callable")


def invert_latent(
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    x_target: torch.Tensor,
    z_init: torch.Tensor,
    steps: int = 300,
    lr: float = 0.1,
    loss: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "l2",
    log_every: int = 50,
) -> OracleResult:
    """Per-image latent inversion on the frozen decoder (RQ0, plan §2/§5).

    Runs Adam on a leaf clone of ``z_init`` to minimize
    ``loss(decode_fn(z), x_target)``. The decoder is frozen: any discoverable
    decoder parameters get ``requires_grad_(False)`` for the duration (flags
    restored afterwards) and are never registered with the optimizer —
    but gradients still flow THROUGH the decoder graph to ``z``. The backward
    pass uses ``backward(inputs=[z])``, so ``.grad`` is accumulated on ``z``
    ONLY: even when ``decode_fn`` is a bare closure over an unfrozen module
    (whose parameters :func:`_decoder_parameters` cannot discover), no stale
    ``.grad`` is ever written into decoder parameters — a leak that would
    silently corrupt the caller's next ``optimizer.step()`` in a surrounding
    training loop.

    Args:
        decode_fn: differentiable ``z -> image`` map (module, bound method,
            or the ``decode_latents`` helper from :func:`pheq.vae.load_sd_vae`).
        x_target: target image batch ``(B, 3, H, W)`` (e.g. ``τ_a(x)`` pre-clip).
        z_init: initialization ``(B, C, h, w)``, typically ``E(x)`` moments;
            not modified.
        steps: number of Adam steps.
        lr: Adam learning rate.
        loss: ``'l2'`` (mean squared error), ``'l1'``, or a callable
            ``(pred, target) -> scalar``.
        log_every: record the running loss every this many steps (no printing;
            library code is silent per SPEC.md style rules).

    Returns:
        :class:`OracleResult` with the detached optimized latent and loss trace.
    """
    loss_fn = _resolve_loss(loss)
    z = z_init.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)

    params = _decoder_parameters(decode_fn)
    saved_flags = [p.requires_grad for p in params]
    losses: list[float] = []
    try:
        for p in params:
            p.requires_grad_(False)
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            value = loss_fn(decode_fn(z), x_target)
            # inputs=[z]: accumulate .grad on z only — never on decoder
            # parameters, even for bare-closure decode_fns (SPEC: "no grad
            # to decoder params").
            value.backward(inputs=[z])
            optimizer.step()
            if step % log_every == 0:
                losses.append(float(value.detach()))
        with torch.no_grad():
            final = float(loss_fn(decode_fn(z), x_target))
        losses.append(final)
    finally:
        for p, flag in zip(params, saved_flags):
            p.requires_grad_(flag)
    return OracleResult(z_opt=z.detach(), losses=losses, final_loss=final)


def oracle_affine_fit(
    z_pairs: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> OracleAffineFit:
    """Fit a channel-affine map ``z_opt ≈ M z_orig + m`` over all pixel sites.

    Answers RQ0's second question: "is the oracle's latent edit
    channel-affine?" (plan §3.2 hypothesis ``T_a(z_p) = M_a z_p + m_a``).
    All latent sites of all pairs are pooled into one least-squares problem
    on the augmented design ``[z; 1]``, solved with
    :func:`pheq.analytic.lstsq_minnorm` (platform-deterministic on
    rank-deficient designs — see that docstring).

    Args:
        z_pairs: sequence of ``(z_orig, z_opt)`` tensors, each ``(B, C, h, w)``
            or ``(C, h, w)``; shapes may differ across pairs but C must match.

    Returns:
        :class:`OracleAffineFit` with the full ``(C, C)`` matrix ``M``,
        translation ``m``, pooled (variance-weighted) R² and per-channel R².
    """
    if len(z_pairs) == 0:
        raise ValueError("z_pairs must contain at least one (z_orig, z_opt) pair")

    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for z_orig, z_opt in z_pairs:
        if z_orig.dim() == 3:
            z_orig = z_orig.unsqueeze(0)
        if z_opt.dim() == 3:
            z_opt = z_opt.unsqueeze(0)
        if z_orig.shape != z_opt.shape:
            raise ValueError(f"pair shape mismatch: {tuple(z_orig.shape)} vs {tuple(z_opt.shape)}")
        c = z_orig.shape[1]
        xs.append(z_orig.detach().permute(0, 2, 3, 1).reshape(-1, c))
        ys.append(z_opt.detach().permute(0, 2, 3, 1).reshape(-1, c))
    x = torch.cat(xs, dim=0).to(torch.float32)
    y = torch.cat(ys, dim=0).to(torch.float32)
    n, c = x.shape

    design = torch.cat([x, torch.ones(n, 1, dtype=x.dtype, device=x.device)], dim=1)
    solution = lstsq_minnorm(design, y)  # (C+1, C)
    m_matrix = solution[:c].T.contiguous()  # (C, C): z_opt ≈ M z_orig + m
    m_vec = solution[c].contiguous()  # (C,)

    pred = design @ solution
    ss_res = ((y - pred) ** 2).sum(dim=0)  # (C,)
    ss_tot = ((y - y.mean(dim=0)) ** 2).sum(dim=0)  # (C,)

    # Pooled, variance-weighted R² (headline number): channels whose oracle
    # target is (near-)constant — e.g. a null-space channel the oracle never
    # touches — contribute ~0 to both sums instead of poisoning the average.
    r2_total = float(1.0 - ss_res.sum() / ss_tot.sum()) if float(ss_tot.sum()) > 0.0 else 1.0

    eps = 1e-12
    degenerate = ss_tot <= eps
    r2_per_channel = torch.where(
        degenerate,
        torch.where(ss_res <= eps, torch.ones_like(ss_tot), torch.zeros_like(ss_tot)),
        1.0 - ss_res / torch.clamp(ss_tot, min=eps),
    )
    return OracleAffineFit(M=m_matrix, m=m_vec, r2=r2_total, r2_per_channel=r2_per_channel)
