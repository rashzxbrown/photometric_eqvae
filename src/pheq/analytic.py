"""Closed-form analytic channel-affine latent operator and W-fit.

Implements plan §3.2 (analytic operator) and §3.4 (posterior push-forward):

    M_a = W⁺ A_a W + (I_C − W⁺W) K,   m_a = W⁺ (A_a c + b_a − c),

where ``rgb ≈ W z + c`` is the fitted linear latent→RGB "preview" map,
``W⁺`` is the Moore–Penrose pseudo-inverse, and ``K`` acts on the
(C−3)-dimensional null space of ``W``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class WFit:
    """Least-squares fit of the linear latent→RGB map ``rgb ≈ W z + c``.

    Attributes:
        W: (3, C) linear color-decoding matrix (plan §3.2).
        c: (3,) offset.
        r2: variance-weighted total R² (pooled over the three RGB channels,
            i.e. channels weighted by their variance).
        r2_per_channel: (3,) R² per RGB channel.
    """

    W: torch.Tensor
    c: torch.Tensor
    r2: float
    r2_per_channel: torch.Tensor


def lstsq_minnorm(design: torch.Tensor, y: torch.Tensor, rtol: float = 1e-10) -> torch.Tensor:
    """Minimum-norm least-squares solve that is deterministic across platforms.

    ``torch.linalg.lstsq`` delegates rank handling to the platform LAPACK; on
    exactly rank-deficient designs (e.g. ToyLinearAE's constant 4th latent
    channel, collinear with the intercept column) different backends
    (Accelerate on macOS vs MKL/OpenBLAS on Linux) pivot differently and can
    return numerically poor float32 solutions — observed as R² 0.61 vs 0.999
    for the identical fit. An explicit float64 pseudo-inverse with a fixed
    singular-value cutoff yields the same minimum-norm solution everywhere:
    rtol=1e-10 (relative to σ_max, in float64) discards exact null directions
    while keeping every informative one.
    """
    solution = torch.linalg.pinv(design.double(), rtol=rtol) @ y.double()
    return solution.to(y.dtype)


def fit_w(latents: torch.Tensor, images: torch.Tensor) -> WFit:
    """Fit ``rgb ≈ W z + c`` by least squares over all pixel sites (plan §3.2).

    Images are box-downsampled (``mode='area'``, i.e. the mean over each
    latent site's pixel block for f-aligned autoencoders) to the latent
    spatial resolution, spatial dims are flattened, and the augmented design
    ``[z; 1]`` is solved over all N*h*w sites via :func:`lstsq_minnorm`.

    The box average is used INSTEAD of an antialiased bilinear downsample
    because each area-downsampled pixel depends only on its own latent site's
    block: an antialias kernel (e.g. 1/8, 3/8, 3/8, 1/8 for f = 2) mixes
    adjacent blocks and is a biased estimator of the pointwise latent→RGB
    map — on images with cross-block high-frequency content it attenuates W
    by the kernel's own-block mass (~44% shrinkage at f = 2), which cancels
    in M = W⁺AW but inflates m = W⁺(Ac + b − c). Regression-tested against
    ToyLinearAE with white-noise latents in tests/test_vae.py.

    Args:
        latents: (N, C, h, w) latent tensor.
        images: (N, 3, H, W) images in [0, 1].

    Returns:
        WFit with W (3, C), c (3,), per-channel R² and the
        variance-weighted total R².
    """
    n, c_lat, h, w = latents.shape
    rgb = F.interpolate(images, size=(h, w), mode="area")
    # (N*h*w, C) and (N*h*w, 3)
    z = latents.permute(0, 2, 3, 1).reshape(-1, c_lat)
    y = rgb.permute(0, 2, 3, 1).reshape(-1, 3)

    design = torch.cat([z, torch.ones(z.shape[0], 1, dtype=z.dtype, device=z.device)], dim=1)
    solution = lstsq_minnorm(design, y)  # (C+1, 3)
    w_mat = solution[:c_lat].T.contiguous()  # (3, C)
    c_vec = solution[c_lat].contiguous()  # (3,)

    residual = y - design @ solution
    ss_res = (residual**2).sum(dim=0)  # (3,)
    centered = y - y.mean(dim=0, keepdim=True)
    ss_tot = (centered**2).sum(dim=0)  # (3,)
    r2_per_channel = 1.0 - ss_res / ss_tot
    # Variance-weighted total: pool SS over channels (weights channels by variance).
    r2_total = float(1.0 - ss_res.sum() / ss_tot.sum())
    return WFit(W=w_mat, c=c_vec, r2=r2_total, r2_per_channel=r2_per_channel)


def analytic_operator(
    fit: WFit,
    A: torch.Tensor,
    b: torch.Tensor,
    K: str | torch.Tensor = "I",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form channel-affine latent operator (plan §3.2).

    Computes the full (never diagonalized) C×C operator

        M = W⁺ A W + (I_C − W⁺W) K,   m = W⁺ (A c + b − c),

    with ``W⁺ = torch.linalg.pinv(W)``. ``K`` acts on the null space of W
    (latent directions carrying non-color information).

    Args:
        fit: WFit with W (3, C) and c (3,).
        A: (3, 3) pixel-space color matrix.
        b: (3,) pixel-space color offset.
        K: null-space action — "I" (default, identity on the residual),
           "0" (annihilate the residual), or an explicit (C, C) tensor.

    Returns:
        (M, m): M (C, C), m (C,).
    """
    w_mat = fit.W
    c_lat = w_mat.shape[1]
    dtype, device = w_mat.dtype, w_mat.device
    A = A.to(dtype=dtype, device=device)
    b = b.to(dtype=dtype, device=device)

    if isinstance(K, str):
        if K == "I":
            k_mat = torch.eye(c_lat, dtype=dtype, device=device)
        elif K == "0":
            k_mat = torch.zeros(c_lat, c_lat, dtype=dtype, device=device)
        else:
            raise ValueError(f"K must be 'I', '0', or a (C, C) tensor, got {K!r}")
    else:
        if K.shape != (c_lat, c_lat):
            raise ValueError(f"explicit K must have shape ({c_lat}, {c_lat}), got {tuple(K.shape)}")
        k_mat = K.to(dtype=dtype, device=device)

    w_pinv = torch.linalg.pinv(w_mat)  # (C, 3)
    eye_c = torch.eye(c_lat, dtype=dtype, device=device)
    m_mat = w_pinv @ A @ w_mat + (eye_c - w_pinv @ w_mat) @ k_mat
    m_vec = w_pinv @ (A @ fit.c + b - fit.c)
    return m_mat, m_vec


def apply_channel_affine(z: torch.Tensor, M: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Apply the pointwise channel-affine map ``z_p ↦ M z_p + m`` (plan §3.2).

    Args:
        z: (B, C, h, w) latents.
        M: (C, C) channel-mixing matrix.
        m: (C,) channel offset.

    Returns:
        (B, C, h, w) transformed latents.
    """
    return torch.einsum("dc,bchw->bdhw", M, z) + m[None, :, None, None]


def push_forward_posterior(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    M: torch.Tensor,
    m: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Push a diagonal Gaussian posterior through the affine operator (plan §3.4).

    The pushed-forward distribution of ``z ~ N(mu, diag(sigma²))`` under
    ``z ↦ M z + m`` is Gaussian with mean ``M mu + m`` and full covariance
    ``M diag(sigma²) Mᵀ``. Per the spec decision, the returned std is the
    *marginal* std of that push-forward,

        sigma' = sqrt(diag(M diag(sigma²) Mᵀ)) = sqrt((M∘M) sigma²),

    computed pointwise (NOT ``|M| sigma``, which is exact only for diagonal M).

    Args:
        mu: (B, C, h, w) posterior means.
        sigma: (B, C, h, w) posterior stds (positive).
        M: (C, C) channel-mixing matrix.
        m: (C,) channel offset.

    Returns:
        (mu', sigma'): pushed mean (B, C, h, w) and marginal std (B, C, h, w).
    """
    mu_pushed = apply_channel_affine(mu, M, m)
    sigma_pushed = torch.sqrt(torch.einsum("dc,bchw->bdhw", M * M, sigma * sigma))
    return mu_pushed, sigma_pushed
