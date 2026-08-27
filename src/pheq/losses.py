"""Reconstruction loss suite for the GAN-free fine-tunes (SPEC2 "losses.py").

Sprint-scale objective (docs/plan-3month.md M1, plan §3.5 adapted GAN-free):
``loss = L1 + lambda_lpips * LPIPS + lambda_kl * KL``. This module provides
the L1+LPIPS reconstruction term (:class:`ReconLoss`) and the diagonal
Gaussian KL (:func:`kl_loss`); train_ae assembles the total.

Conventions (SPEC.md): images ``(B, 3, H, W)`` float32 in ``[0, 1]``
(pre-clip — decoder outputs are NOT clamped, plan §3.1).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ReconLoss(nn.Module):
    """``forward(pred, target) -> l1 + lambda_lpips * lpips`` (SPEC2 "losses.py").

    LPIPS handling:

    - The ``lpips`` package (``net='vgg'``) is imported LAZILY inside
      ``__init__`` so that ``pheq.losses`` imports (and the L1-only path
      runs) without the package or its VGG16 backbone weights present.
    - The LPIPS network is frozen (``requires_grad_(False)``) and kept in
      eval mode permanently — :meth:`train` is overridden so a
      ``recon_loss.train()`` call from a training loop cannot flip it.
    - LPIPS inputs are mapped ``[0, 1] -> [-1, 1]`` (the package's native
      range) immediately before the call.
    - Pre-clip clamp (SPEC2 care point): both ``pred`` and ``target`` are
      clamped to ``[-0.1, 1.1]`` BEFORE the range mapping, and this clamp is
      applied ONLY to the LPIPS inputs. Rationale: equivariance targets are
      pre-clip (plan §3.1) and can stray outside ``[0, 1]``; VGG features
      explode far out of range, so LPIPS sees lightly clamped inputs — while
      the L1 term stays fully unclamped so the gradient still pulls
      out-of-range predictions toward the (possibly out-of-range) target.

    If constructing the LPIPS network fails (package missing or weight
    download unavailable):

    - ``require_lpips=True`` (default): raise ``RuntimeError``.
    - ``require_lpips=False``: fall back to L1-only and set
      ``self.lpips_active = False`` (CLIs are expected to log this loudly).

    Args:
        lambda_lpips: weight on the LPIPS term (default 1.0, SPEC2).
        require_lpips: whether LPIPS is mandatory (see above).
    """

    def __init__(self, lambda_lpips: float = 1.0, require_lpips: bool = True) -> None:
        super().__init__()
        self.lambda_lpips = float(lambda_lpips)
        self.lpips_active: bool = False
        self.lpips_net: nn.Module | None = None
        try:
            import lpips  # lazy: package must import without it (SPEC2)

            net = lpips.LPIPS(net="vgg")
        except Exception as exc:  # import error OR weight-download failure
            if require_lpips:
                raise RuntimeError(
                    "ReconLoss: LPIPS (net='vgg') unavailable and "
                    "require_lpips=True; install `lpips` / pre-download the "
                    "VGG16 weights, or pass require_lpips=False for L1-only."
                ) from exc
            return
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.lpips_net = net  # registered submodule: follows .to(device)
        self.lpips_active = True

    def train(self, mode: bool = True) -> "ReconLoss":
        """Standard ``train()``, but the frozen LPIPS net stays in eval mode."""
        super().train(mode)
        if self.lpips_net is not None:
            self.lpips_net.eval()
        return self

    def lpips_term(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Unweighted mean LPIPS distance (clamped + range-mapped inputs).

        Convenience for monitoring (train_ae logs L1 and LPIPS separately);
        returns 0 when LPIPS is inactive. The clamp to ``[-0.1, 1.1]`` and
        the ``[0, 1] -> [-1, 1]`` mapping documented on the class happen
        here — and ONLY here, never on the L1 inputs.
        """
        if not self.lpips_active:
            return pred.new_zeros(())
        assert self.lpips_net is not None
        if min(pred.shape[-2:]) < 16:
            # VGG16's pooling stack needs >= 16 px per side (an 8-px input
            # collapses to 0x0 mid-trunk and raises). Only reachable at
            # tiny test/smoke sizes — e.g. b2lite's spatial branch scaling
            # a 32-px image by 0.25; production sizes (256 -> >= 64 after
            # the [0.25, 1] scale range) are unaffected. Skip the term.
            return pred.new_zeros(())
        p = 2.0 * pred.clamp(-0.1, 1.1) - 1.0
        t = 2.0 * target.clamp(-0.1, 1.1) - 1.0
        return self.lpips_net(p, t).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """``l1 + lambda_lpips * lpips`` (L1-only when ``lpips_active`` is False).

        Args:
            pred: decoded images ``(B, 3, H, W)``, pre-clip (unclamped).
            target: reconstruction targets, same shape, pre-clip.

        Returns:
            Scalar loss tensor.
        """
        l1 = F.l1_loss(pred, target)  # UNCLAMPED (see class docstring)
        if not self.lpips_active:
            return l1
        return l1 + self.lambda_lpips * self.lpips_term(pred, target)


def kl_loss(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Diagonal-Gaussian KL to N(0, I), taking SIGMA — not logvar.

    SPEC2 "losses.py": ``kl = 0.5 * (mu^2 + sigma^2 - 1 - log sigma^2)``,
    SUMMED over each batch element's latent dims and then averaged over the
    batch ("mean per batch element"). This is the LDM/SD-VAE reduction
    (CompVis sums the KL over dims [1, 2, 3] and means over the batch) — the
    convention the default ``lambda_kl = 1e-6`` was calibrated for, and the
    resolution-safe choice (a global mean would silently rescale the
    effective KL weight by ``1 / (C * h * w)``). NOTE: sum-per-element and
    global mean are NOT equivalent — for a ``(B, 4, 32, 32)`` latent they
    differ by exactly 4096x. The second argument is the posterior STANDARD
    DEVIATION, matching ``vae.encode_moments -> (mu, sigma)`` (SPEC
    pheq/vae.py, plan §3.4 — the push-forward also operates on (mu, sigma)).

    The KL in the training objective is computed on the UNTRANSFORMED
    posterior (plan §3.4, as in EQ-VAE); callers pass the raw encoder
    moments, never pushed-forward ones.

    Args:
        mu: posterior mean, any shape (typically ``(B, C, h, w)``); the first
            dim is the batch dim (a 0-d/1-d input is treated as one batch of
            scalars per element, i.e. dims after the first are summed).
        sigma: posterior std, same shape, ``sigma > 0`` (``sigma = 0`` gives
            an infinite KL, mathematically correct for a degenerate
            posterior — no epsilon is added).

    Returns:
        Scalar KL tensor (sum over non-batch dims, mean over the batch).
    """
    density = 0.5 * (mu.pow(2) + sigma.pow(2) - 1.0 - torch.log(sigma.pow(2)))
    if density.dim() <= 1:
        return density.mean()
    return density.sum(dim=tuple(range(1, density.dim()))).mean()
