"""Learned Lie-affine latent operator g_ψ (plan §3.3, L1 — primary).

Implements the channel-affine operator

    g_ψ(z, a)_p = M_ψ(a) z_p + m_ψ(a),   M_ψ(a) = exp( Σᵢ φᵢ(a) Gᵢ ),

with learned generators Gᵢ ∈ ℝ^{C×C} (one per photometric factor) and canonical
coordinates φ = (log β, log γ, log s, θ), so each one-parameter subgroup
satisfies the homomorphism exactly by construction (plan §3.3). The translation
m_ψ is an MLP over Fourier features of φ with m_ψ(0) = 0 enforced by the
subtraction parameterization, making the identity exact by parameterization.

Generators are initialized from matrix logarithms of the analytic operator
(plan §3.2) at reference magnitudes via :meth:`LieAffineOperator.init_from_analytic`.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import torch
from torch import nn

from pheq.analytic import WFit, analytic_operator, apply_channel_affine

if TYPE_CHECKING:  # pragma: no cover - typing only; color.py is a sibling module
    from pheq.color import PhotoParams

# Reference magnitudes for generator initialization (SPEC.md pheq/lie_operator.py):
# beta = gamma = sat = 1.25, hue = 0.3 rad. Canonical coordinates at those
# magnitudes are (log 1.25, log 1.25, log 1.25, 0.3) — plan §3.1/§3.3.
_REF_MAGNITUDES: tuple[float, float, float, float] = (1.25, 1.25, 1.25, 0.3)
_REF_COORDS: tuple[float, float, float, float] = (
    math.log(1.25),
    math.log(1.25),
    math.log(1.25),
    0.3,
)

#: Finite substitute for phi = -inf coordinates (the absorbing s = 0 grayscale
#: element maps to log(sat) = -inf, plan §3.1 / color.PhotoParams.phi()).
#: torch.matrix_exp never terminates on non-finite entries (its C-level
#: scaling-and-squaring loop spins forever on ±inf/NaN), so -inf is replaced by
#: log(1e-7): exp(log(1e-7) · G) matches the t → -inf limit of exp(t · G) to
#: ~1e-7 (float32 resolution) for generators whose non-null eigenvalues have
#: positive real part (the analytic saturation generator has eigenvalue 1 on
#: the chroma directions).
_PHI_NEG_INF_FLOOR: float = math.log(1e-7)


def _as_phi(params: "PhotoParams | torch.Tensor") -> torch.Tensor:
    """Canonical coordinates of a group element: `.phi()` if available, else tensor."""
    if hasattr(params, "phi"):
        return params.phi()
    return torch.as_tensor(params, dtype=torch.float32)


def _sanitize_phi(phi: torch.Tensor) -> torch.Tensor:
    """Guard canonical coordinates against non-finite entries (plan §3.1 s = 0).

    ``PhotoParams.phi()`` maps the absorbing grayscale element ``s = 0`` to
    ``log s = -inf`` by design, but ``torch.matrix_exp`` (used by
    :meth:`LieAffineOperator.M`) hangs — a non-interruptible C-level loop —
    on any non-finite entry. -inf coordinates are therefore substituted with
    :data:`_PHI_NEG_INF_FLOOR` (see its docstring for the limit argument);
    NaN and +inf have no group-element interpretation and raise.
    """
    if torch.isfinite(phi).all():
        return phi
    if torch.isnan(phi).any() or torch.isposinf(phi).any():
        raise ValueError(
            "phi must be finite, or -inf in coordinates whose factor is at its "
            f"absorbing zero (e.g. sat = 0); got {phi.tolist()}"
        )
    return torch.where(
        torch.isneginf(phi), torch.full_like(phi, _PHI_NEG_INF_FLOOR), phi
    )


def _compose_phi(phi_a: torch.Tensor, phi_b: torch.Tensor) -> torch.Tensor:
    """Canonical coordinates φ(b∘a) of the composed element (a first, then b).

    Every canonical element (SPEC pheq/color.py fixed factor order
    brightness → contrast → saturation → hue, anchor g = 0.5) has the form

        A = β γ · A_sat(s) A_hue(θ),   b = (1 − γ) · g · 𝟙,

    and since A_sat / A_hue fix 𝟙 and all linear parts commute, the
    homogeneous-coordinate product (A_b, b_b)∘(A_a, b_a) is again canonical
    with

        s' = s_a s_b,   θ' = θ_a + θ_b,
        γ' = γ_b (1 − β_b (1 − γ_a)),   β' = β_a β_b γ_a γ_b / γ'.

    NOTE (plan §3.3 / SPEC composition_loss): φ(b∘a) ≠ φ_a + φ_b in general —
    the log-linear parts add, but the contrast bias does not (e.g.
    a = contrast(0.7), b = brightness(1.3): true composed bias
    1.3·0.3·0.5 = 0.195, while the canonical element at φ_a + φ_b has bias
    0.3·0.5 = 0.15). Verified against ``color.compose`` in the tests.

    Raises:
        ValueError: if the composed element leaves the canonical coordinate
            chart (γ' ≤ 0, impossible within the plan §3.1 sampling ranges).
    """
    sat_hue = phi_a[2:] + phi_b[2:]
    # Bit-exact shortcuts (identity-composition exactness, plan §3.3): when
    # either side has trivial brightness AND contrast, the composed (β, γ)
    # coordinates are exactly the other side's — skip the exp/log round trip.
    if not bool((phi_b[:2] != 0).any()):
        return torch.cat([phi_a[:2], sat_hue])
    if not bool((phi_a[:2] != 0).any()):
        return torch.cat([phi_b[:2], sat_hue])
    beta_a, gamma_a = torch.exp(phi_a[0]), torch.exp(phi_a[1])
    beta_b, gamma_b = torch.exp(phi_b[0]), torch.exp(phi_b[1])
    gamma_c = gamma_b * (1.0 - beta_b * (1.0 - gamma_a))
    if float(gamma_c) <= 0.0:
        raise ValueError(
            "composed element has non-positive contrast coordinate "
            f"gamma' = {float(gamma_c):.6f}; it lies outside the canonical "
            "chart (cannot occur within the plan §3.1 sampling ranges)"
        )
    beta_c = beta_a * beta_b * gamma_a * gamma_b / gamma_c
    return torch.cat(
        [torch.stack([torch.log(beta_c), torch.log(gamma_c)]), sat_hue]
    )


class LieAffineOperator(nn.Module):
    """Lie-affine channel operator M_ψ(φ) z + m_ψ(φ) (plan §3.3, operator L1).

    Parameters:
        G: (4, C, C) learned generators, one per factor
           (brightness, contrast, saturation, hue).
        mlp: translation map over Fourier features of φ,
           phi (4,) → fourier (4*2*n_freq,) → hidden → C,
           with m(0) = 0 enforced by m(φ) = mlp(fourier(φ)) − mlp(fourier(0)).

    ~4.5K parameters for C = 4 — deliberately too small to become a shadow
    decoder (plan §3.3 / §3.5 collapse defenses).
    """

    def __init__(self, channels: int = 4, hidden: int = 64, n_freq: int = 8) -> None:
        super().__init__()
        self.channels = channels
        self.n_freq = n_freq
        # Small random init; real initialization comes from init_from_analytic.
        # phi = 0 still gives sum_i 0 * G_i = exact zero matrix, so identity at
        # phi = 0 is exact regardless of G.
        self.G = nn.Parameter(0.01 * torch.randn(4, channels, channels))
        self.register_buffer(
            "freqs", 2.0 ** torch.arange(n_freq, dtype=torch.float32)
        )
        self.mlp = nn.Sequential(
            nn.Linear(4 * 2 * n_freq, hidden),
            nn.SiLU(),
            nn.Linear(hidden, channels),
        )

    def fourier(self, phi: torch.Tensor) -> torch.Tensor:
        """Fourier features of φ: [sin(2^k φᵢ), cos(2^k φᵢ)] (plan §3.3 conditioning).

        Args:
            phi: (..., 4) canonical coordinates.

        Returns:
            (..., 4 * 2 * n_freq) feature tensor.
        """
        ang = phi.unsqueeze(-1) * self.freqs  # (..., 4, n_freq)
        feats = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (..., 4, 2*n_freq)
        return feats.flatten(-2)

    def M(self, phi: torch.Tensor) -> torch.Tensor:
        """Channel-mixing matrix M_ψ(φ) = exp(Σᵢ φᵢ Gᵢ) (plan §3.3).

        `torch.matrix_exp` of the exact zero matrix is exactly the identity,
        so M(0) = I bit-exactly.

        Non-finite coordinates are guarded by :func:`_sanitize_phi`
        (``matrix_exp`` hangs on ±inf/NaN entries): -inf — the absorbing
        ``s = 0`` element, plan §3.1 — is evaluated at the finite floor
        :data:`_PHI_NEG_INF_FLOOR`; NaN/+inf raise ``ValueError``.

        Args:
            phi: (4,) or (B, 4) canonical coordinates.

        Returns:
            (C, C) or (B, C, C) matrix; matrix_exp batches over leading dims.
        """
        # fp32 island: torch.matrix_exp is not autocast-safe — under CUDA bf16
        # autocast its internal scaling-and-squaring hits a mixed-dtype
        # index_put ("BFloat16 destination, Float source"). Compute M in
        # float32 with autocast disabled; callers' einsum re-promotes as
        # needed. (Observed on Oscar gpu2105, torch 2.13+cu129, p2_lie run.)
        phi_s = _sanitize_phi(phi)
        with torch.autocast(device_type=phi_s.device.type, enabled=False):
            gen = torch.einsum(
                "...i,icd->...cd", phi_s.to(torch.float32), self.G.to(torch.float32)
            )
            return torch.matrix_exp(gen)

    def m(self, phi: torch.Tensor) -> torch.Tensor:
        """Translation m_ψ(φ) with m(0) = 0 by subtraction parameterization (plan §3.3).

        m(φ) = mlp(fourier(φ)) − mlp(fourier(0)). The baseline is evaluated on
        `zeros_like(phi)` (same shape, same kernel), so zero rows of a batched φ
        map to bit-exact zero translations. Non-finite coordinates are guarded
        by :func:`_sanitize_phi` (sin/cos of -inf is NaN): -inf is evaluated at
        the finite floor :data:`_PHI_NEG_INF_FLOOR`; NaN/+inf raise.

        Args:
            phi: (4,) or (B, 4) canonical coordinates.

        Returns:
            (C,) or (B, C) translation.
        """
        phi = _sanitize_phi(phi)
        return self.mlp(self.fourier(phi)) - self.mlp(self.fourier(torch.zeros_like(phi)))

    def forward(self, z: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Apply the operator: z_p ↦ M_ψ(φ) z_p + m_ψ(φ) (plan §3.3).

        Args:
            z: (B, C, h, w) latents.
            phi: (4,) canonical coordinates shared across the batch, or (B, 4)
                per-sample coordinates.

        Returns:
            (B, C, h, w) transformed latents.
        """
        if phi.dim() == 1:
            return apply_channel_affine(z, self.M(phi), self.m(phi))
        if phi.dim() == 2:
            if phi.shape[0] != z.shape[0]:
                raise ValueError(
                    f"batched phi has B={phi.shape[0]} but z has B={z.shape[0]}"
                )
            m_mat = self.M(phi)  # (B, C, C)
            m_vec = self.m(phi)  # (B, C)
            return torch.einsum("bdc,bchw->bdhw", m_mat, z) + m_vec[:, :, None, None]
        raise ValueError(f"phi must be (4,) or (B, 4), got shape {tuple(phi.shape)}")

    def init_from_analytic(self, fit: WFit) -> None:
        """Initialize generators from matrix logs of the analytic operator (plan §3.3).

        For each factor i, computes the analytic M (plan §3.2, K = "I") at the
        reference magnitude (β = γ = s = 1.25, θ = 0.3), takes
        ``scipy.linalg.logm`` (real part; warns if the imaginary norm exceeds
        1e-6), divides by the canonical coordinate at that magnitude, and copies
        the result into Gᵢ. Because the analytic family is a one-parameter
        subgroup per factor (K = "I" preserves the homomorphism on the color
        block, plan §3.2), exp(φ Gᵢ) then reproduces the analytic M at *any*
        single-factor magnitude.

        Spec resolution: the final MLP layer is zero-initialized here so that
        m_ψ(φ) ≡ 0 at init — the initialized operator matches the analytic
        operator's linear part exactly, and the translation is learned during
        training (mirrors the zero-initialized-output convention of plan §3.3 L2).

        Args:
            fit: WFit with the fitted (W, c) linear preview map.
        """
        import numpy as np
        import scipy.linalg

        from pheq import color

        factor_affines = (
            color.brightness_affine(_REF_MAGNITUDES[0]),
            color.contrast_affine(_REF_MAGNITUDES[1]),
            color.saturation_affine(_REF_MAGNITUDES[2]),
            color.hue_affine(_REF_MAGNITUDES[3]),
        )
        with torch.no_grad():
            for i, ((a_mat, b_vec), coord) in enumerate(zip(factor_affines, _REF_COORDS)):
                m_mat, _ = analytic_operator(fit, a_mat, b_vec, K="I")
                log_m = scipy.linalg.logm(
                    np.asarray(m_mat.detach().cpu(), dtype=np.float64)
                )
                imag_norm = float(np.linalg.norm(np.imag(log_m)))
                if imag_norm > 1e-6:
                    warnings.warn(
                        f"logm of analytic M for factor {i} has imaginary norm "
                        f"{imag_norm:.3e} > 1e-6; taking the real part.",
                        stacklevel=2,
                    )
                g_i = torch.as_tensor(
                    np.real(log_m), dtype=self.G.dtype, device=self.G.device
                )
                self.G[i].copy_(g_i / coord)
            last = self.mlp[-1]
            last.weight.zero_()
            last.bias.zero_()

    def composition_loss(
        self,
        z: torch.Tensor,
        params_a: "PhotoParams | torch.Tensor",
        params_b: "PhotoParams | torch.Tensor",
    ) -> torch.Tensor:
        """Composition loss L_comp = ‖g(g(z, φa), φb) − g(z, φ(b∘a))‖² / ‖z‖² (plan §3.3).

        Nudges the generators toward the correct bracket structure (cross-factor
        composition holds only up to Baker–Campbell–Hausdorff terms). ``z`` is
        detached — the loss trains ψ only, never the encoder (plan §3.5,
        collapse defense 1).

        The target coordinates are the TRUE canonical coordinates φ(b∘a) of the
        composed group element (SPEC pheq/lie_operator.py, plan §3.3: the
        operator conditions on the composed element), computed by
        :func:`_compose_phi`. Note φ(b∘a) ≠ φ_a + φ_b whenever brightness and
        contrast are split across a and b: the coordinate sum matches the
        composed linear part (all factor A's commute) but NOT the contrast
        bias, so a φ_a + φ_b target would exclude the exactly-equivariant
        operator from the loss minimum (no translation map can absorb the
        mismatch: exact zero would need the cocycle m(φ1 + φ2) =
        M(φ2) m(φ1) + m(φ2), whose only solutions are coboundaries, and the
        true photometric translation is not one).

        Args:
            z: (B, C, h, w) latents (detached internally).
            params_a: first-applied group element — PhotoParams (uses `.phi()`)
                or a (4,) canonical-coordinate tensor.
            params_b: second-applied group element, same conventions.

        Returns:
            Scalar loss tensor.
        """
        phi_a = _as_phi(params_a)
        phi_b = _as_phi(params_b)
        z = z.detach()
        lhs = self.forward(self.forward(z, phi_a), phi_b)
        rhs = self.forward(z, _compose_phi(phi_a, phi_b))
        return (lhs - rhs).pow(2).sum() / z.pow(2).sum()
