"""P3 diagnostic conv-residual latent operator g_ψ (plan §3.3 L2; SPEC2).

Implements the spatially-adaptive residual operator

    g_ψ(z, φ) = T^an(z, φ) + [ h_ψ(z, φ) − h_ψ(z, 0) ],

where T^an is the frozen closed-form analytic operator (plan §3.2) built from
the checkpoint's fitted (W, c), and h_ψ is a small FiLM-conditioned conv net
(three 3×3 convs C→64→64→C, GroupNorm(8), SiLU, ~90K params). L2 deliberately
breaks the pointwise/channel-affine hypothesis: if it beats L1 by a large EE
margin, the true latent color action is spatially heterogeneous or nonlinear —
a finding about tokenizer structure, not a method we advocate (plan §3.3).

Identity at φ = 0 is exact for ALL parameter values by construction
(SPEC2 conv_operator.py): the analytic part is the identity at φ = 0 (enforced
exactly, see :meth:`ConvResidualOperator._analytic_affine`) and the residual
uses the subtraction parameterization h(z, φ) − h(z, 0) — the same trick as
the Lie operator's translation MLP — which is bit-exact zero at φ = 0 and
smooth in φ (unlike a ``phi != 0`` mask, which would break differentiability).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from pheq.analytic import WFit, analytic_operator, apply_channel_affine
from pheq.color import PhotoParams
from pheq.lie_operator import LieAffineOperator, _as_phi, _sanitize_phi

__all__ = ["params_from_phi", "ConvResidualOperator"]


def params_from_phi(phi: torch.Tensor) -> PhotoParams:
    """Inverse of ``PhotoParams.phi()`` (SPEC2 conv_operator.py, free function).

    Canonical coordinates φ = (log β, log γ, log s, θ) (plan §3.3) map back to
    the group element via ``beta = exp(φ0)``, ``gamma = exp(φ1)``,
    ``sat = exp(φ2)``, ``hue = φ3``. The absorbing grayscale element is
    handled exactly: ``φ2 = -inf ↦ sat = 0.0`` (plan §3.1). This is a FREE
    function by spec ("do NOT edit color.py").

    Note: extraction goes through python floats, so ``params_from_phi`` is
    NOT differentiable in φ — the analytic branch of the conv operator is
    frozen closed-form anyway; φ-gradients flow through the residual h only.

    Args:
        phi: ``(4,)`` canonical-coordinate tensor (any real dtype; -inf
            allowed in the log-coordinates).

    Returns:
        The corresponding :class:`pheq.color.PhotoParams`.

    Raises:
        ValueError: if ``phi`` does not have exactly 4 elements.
    """
    phi_t = torch.as_tensor(phi, dtype=torch.float32).reshape(-1)
    if phi_t.numel() != 4:
        raise ValueError(f"phi must have 4 elements, got {phi_t.numel()}")
    return PhotoParams(
        beta=math.exp(float(phi_t[0])),
        gamma=math.exp(float(phi_t[1])),
        sat=math.exp(float(phi_t[2])),
        hue=float(phi_t[3]),
    )


def _film_mlp(feat_dim: int, hidden: int, width: int = 128) -> nn.Sequential:
    """FiLM conditioner: fourier(φ) → width → (scale, shift) for ``hidden`` channels.

    The small hidden layer (width 128) puts the total operator at ~90K params
    (plan §3.3 L2's stated budget); a single linear layer would land at ~58K.
    """
    return nn.Sequential(
        nn.Linear(feat_dim, width),
        nn.SiLU(),
        nn.Linear(width, 2 * hidden),
    )


class ConvResidualOperator(nn.Module):
    """Conv-residual operator g_ψ(z, φ) = T^an(z, φ) + h_ψ(z, φ) − h_ψ(z, 0).

    (plan §3.3 L2 — diagnostic upper bound only; SPEC2 conv_operator.py.)

    The analytic part T^an is computed internally per sample from the frozen
    ``(W, c)`` buffers: φ → :func:`params_from_phi` → ``PhotoParams.affine()``
    → :func:`pheq.analytic.analytic_operator` (K = "I") →
    :func:`pheq.analytic.apply_channel_affine`. The residual h_ψ is
    C→hidden→hidden→C with 3×3 convs, GroupNorm(8), FiLM after each hidden
    layer on Fourier features of φ (helper REUSED from
    :class:`pheq.lie_operator.LieAffineOperator` by import, not duplicated),
    SiLU, and a ZERO-INITIALIZED final conv — so g equals the pure analytic
    operator at init, and equals the identity at φ = 0 for all time via the
    subtraction parameterization (bit-exact: both h passes see identical
    inputs at φ = 0, and GroupNorm/conv act per-sample, so the difference is
    exactly zero row-wise even in mixed batches).

    ~91.5K parameters for C = 4, hidden = 64, n_freq = 8 (SPEC2: 50K–150K).
    """

    def __init__(self, wfit: WFit, hidden: int = 64, n_freq: int = 8) -> None:
        """Build the operator around a frozen W-fit (SPEC2 conv_operator.py).

        Args:
            wfit: fitted linear preview map (plan §3.2); ``W (3, C)`` and
                ``c (3,)`` are registered as frozen buffers (they move with
                ``.to(device)`` but receive no gradient).
            hidden: conv width (must be divisible by 8 for GroupNorm(8)).
            n_freq: Fourier-feature octaves for the FiLM conditioning
                (matches the Lie operator's default).
        """
        super().__init__()
        if hidden % 8 != 0:
            raise ValueError(f"hidden must be divisible by 8 (GroupNorm(8)), got {hidden}")
        self.channels = int(wfit.W.shape[1])
        self.n_freq = n_freq
        self.register_buffer("W", wfit.W.detach().clone().to(torch.float32))
        self.register_buffer("c", wfit.c.detach().clone().to(torch.float32))
        self.register_buffer(
            "r2_per_channel",
            torch.as_tensor(wfit.r2_per_channel, dtype=torch.float32).detach().clone(),
        )
        self.r2 = float(wfit.r2)
        # Same buffer name/semantics as LieAffineOperator so its fourier()
        # method can be reused unbound (SPEC2: import, don't duplicate).
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freq, dtype=torch.float32))

        feat_dim = 4 * 2 * n_freq
        c_lat = self.channels
        self.conv_in = nn.Conv2d(c_lat, hidden, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, hidden)
        self.film1 = _film_mlp(feat_dim, hidden)
        self.conv_mid = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, hidden)
        self.film2 = _film_mlp(feat_dim, hidden)
        self.conv_out = nn.Conv2d(hidden, c_lat, kernel_size=3, padding=1)
        # Zero-initialized output conv (plan §3.3 L2): h ≡ 0 at init, so the
        # freshly built operator matches the pure analytic operator exactly.
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def fourier(self, phi: torch.Tensor) -> torch.Tensor:
        """Fourier features of φ — REUSED from the Lie operator by import.

        Calls :meth:`pheq.lie_operator.LieAffineOperator.fourier` unbound
        (it only reads ``self.freqs``, which this module registers with
        identical semantics), per SPEC2's "reuse the fourier helper from
        lie_operator; import, don't duplicate".
        """
        return LieAffineOperator.fourier(self, phi)

    def _fit(self) -> WFit:
        """WFit view over the registered buffers (device/dtype follow the module)."""
        return WFit(W=self.W, c=self.c, r2=self.r2, r2_per_channel=self.r2_per_channel)

    def _analytic_affine(self, phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-row analytic (M, m) from canonical coordinates (plan §3.2).

        Builds the ``(A, b)`` pixel affine per batch element via
        :func:`params_from_phi` + ``PhotoParams.affine()`` and conjugates it
        through the frozen W-fit with K = "I". Cost (documented per SPEC2): a
        python loop over B, each iteration building a 4×4 matrix including a
        ``pinv`` of the (3, C) W — negligible next to the conv stack for
        sprint batch sizes (B ≤ ~256), and the train loop samples ONE params
        per batch anyway, which hits the single-row path.

        Exact identity at φ = 0 (REQUIRED by SPEC2): rows that are exactly
        zero short-circuit to ``(I_C, 0)``. The generic path lands within
        ~1 ulp of that (e.g. ``hue_affine(0)`` is the float64 product
        ``T_yiq⁻¹ T_yiq`` ≈ I, and W⁺AW + (I − W⁺W) re-adds to I only up to
        rounding), so the short-circuit is the limit value up to float noise
        — continuity in φ is preserved to ~1e-7, below every tolerance used
        downstream. No gradient flows through this branch in either case
        (see :func:`params_from_phi`).

        Args:
            phi: ``(B, 4)`` canonical coordinates (rows may contain -inf in
                the log-saturation slot: the absorbing s = 0 element).

        Returns:
            ``(M, m)``: ``(B, C, C)`` and ``(B, C)`` on the module's device.
        """
        fit = self._fit()
        eye = torch.eye(self.channels, dtype=self.W.dtype, device=self.W.device)
        zero = torch.zeros(self.channels, dtype=self.W.dtype, device=self.W.device)
        mats: list[torch.Tensor] = []
        vecs: list[torch.Tensor] = []
        for row in phi.detach():
            if bool((row == 0).all()):
                mats.append(eye)
                vecs.append(zero)
                continue
            params = params_from_phi(row)
            a_mat, b_vec = params.affine()
            m_mat, m_vec = analytic_operator(fit, a_mat, b_vec, K="I")
            mats.append(m_mat)
            vecs.append(m_vec)
        return torch.stack(mats), torch.stack(vecs)

    def h(self, z: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Residual conv net h_ψ(z, φ) (plan §3.3 L2 architecture).

        conv3×3 → GroupNorm(8) → FiLM(fourier(φ)) → SiLU, twice, then the
        zero-initialized output conv. FiLM uses the ``(1 + scale)`` convention
        so a zero conditioner is a no-op. All ops act per sample (convs,
        GroupNorm, row-wise FiLM), which makes the identity-by-subtraction in
        :meth:`forward` bit-exact row-wise.

        Args:
            z: ``(B, C, h, w)`` latents.
            phi: ``(B, 4)`` FINITE canonical coordinates (callers sanitize).

        Returns:
            ``(B, C, h, w)`` residual.
        """
        feats = self.fourier(phi)  # (B, 4*2*n_freq)
        x = self.norm1(self.conv_in(z))
        s1, t1 = self.film1(feats).chunk(2, dim=-1)
        x = F.silu(x * (1.0 + s1[:, :, None, None]) + t1[:, :, None, None])
        x = self.norm2(self.conv_mid(x))
        s2, t2 = self.film2(feats).chunk(2, dim=-1)
        x = F.silu(x * (1.0 + s2[:, :, None, None]) + t2[:, :, None, None])
        return self.conv_out(x)

    def forward(self, z: torch.Tensor, phi: "torch.Tensor | PhotoParams") -> torch.Tensor:
        """g(z, φ) = T^an(z, φ) + h(z, φ) − h(z, 0) (plan §3.3 L2; SPEC2).

        The subtraction parameterization enforces identity at φ = 0 for ALL
        time: at φ = 0 the two h passes compute identical values (difference
        exactly 0) and the analytic part is exactly the identity, so
        ``g(z, 0) == z`` bit-exactly — while staying smooth in φ (finite,
        generally nonzero ∂g/∂φ near 0 through the FiLM conditioning; a
        ``phi != 0`` output mask would break that, SPEC2).

        h's conditioning is sanitized via the Lie operator's guard
        (imported): ``-inf`` log-saturation (the absorbing s = 0 element) is
        replaced by a finite floor for the Fourier features only — the
        analytic branch receives the exact ``sat = 0`` absorbing affine.

        Args:
            z: ``(B, C, h, w)`` latents.
            phi: ``(4,)`` canonical coordinates shared across the batch (the
                train-loop convention: one params per batch), ``(B, 4)``
                per-sample coordinates, or a ``PhotoParams`` (via ``.phi()``).

        Returns:
            ``(B, C, h, w)`` transformed latents.
        """
        phi_t = _as_phi(phi)
        if z.dim() != 4 or z.shape[1] != self.channels:
            raise ValueError(
                f"z must be (B, {self.channels}, h, w), got shape {tuple(z.shape)}"
            )
        if phi_t.dim() == 1:
            if phi_t.shape != (4,):
                raise ValueError(f"phi must be (4,) or (B, 4), got {tuple(phi_t.shape)}")
            m_mat, m_vec = self._analytic_affine(phi_t.unsqueeze(0))
            analytic = apply_channel_affine(z, m_mat[0], m_vec[0])
            phi_b = phi_t.unsqueeze(0).expand(z.shape[0], 4)
        elif phi_t.dim() == 2:
            if phi_t.shape != (z.shape[0], 4):
                raise ValueError(
                    f"batched phi must be ({z.shape[0]}, 4), got {tuple(phi_t.shape)}"
                )
            m_mat, m_vec = self._analytic_affine(phi_t)
            analytic = (
                torch.einsum("bdc,bchw->bdhw", m_mat, z) + m_vec[:, :, None, None]
            )
            phi_b = phi_t
        else:
            raise ValueError(f"phi must be (4,) or (B, 4), got shape {tuple(phi_t.shape)}")
        phi_b = _sanitize_phi(phi_b)
        residual = self.h(z, phi_b) - self.h(z, torch.zeros_like(phi_b))
        return analytic + residual
