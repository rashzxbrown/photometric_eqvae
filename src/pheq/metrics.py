"""pheq.metrics — equivariance-error battery (plan §5.3).

Color-difference metrics (sRGB → CIELAB → CIEDE2000 per Sharma, Wu & Dalal 2005),
decoder-side and latent-side equivariance errors, and a hue-histogram distance.

All image tensors are ``(B, 3, H, W)`` float in ``[0, 1]``; Lab tensors are
``(B, 3, H, W)`` with channels ``(L, a, b)``. No in-place ops anywhere: every
function is safe to differentiate through.
"""

from __future__ import annotations

import math
from typing import Callable, Literal, Union

import torch

__all__ = [
    "rgb_to_lab",
    "ciede2000",
    "mean_ciede2000",
    "ee_pix",
    "ee_lat",
    "hue_histogram_distance",
]

# ---------------------------------------------------------------------------
# sRGB -> CIELAB (D65)
# ---------------------------------------------------------------------------

# sRGB (linear) -> XYZ, D65 white, IEC 61966-2-1 primaries.
_SRGB_TO_XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)
# D65 reference white (2° observer), consistent with the row sums of the matrix.
_D65_WHITE = (0.95047, 1.00000, 1.08883)

# CIE constants (exact rationals): eps = (6/29)^3, kappa = (29/3)^3.
_LAB_EPS = 216.0 / 24389.0
_LAB_KAPPA = 24389.0 / 27.0


def rgb_to_lab(img: torch.Tensor) -> torch.Tensor:
    """Convert sRGB images in ``[0, 1]`` to CIELAB (D65 white point).

    Pipeline (plan §5.3, standard sRGB/CIE definitions): sRGB inverse-gamma
    expansion (IEC 61966-2-1), linear RGB → XYZ via the D65 sRGB matrix, then
    XYZ → Lab with the exact CIE constants eps = 216/24389, kappa = 24389/27.
    Differentiable-friendly: no in-place operations.

    Out-of-range inputs (the pre-clip convention of plan §3.1 routinely
    produces values below 0 / above 1) take the linear branch below the sRGB
    knee; the power branch is evaluated on a base clamped to ``>= 0`` so that
    values below -0.055 never produce NaN in the *unselected* ``torch.where``
    branch (whose backward would otherwise poison the gradient with
    ``NaN * 0 = NaN``). Forward values are unchanged: the clamp only binds
    where the linear branch is selected anyway.

    Args:
        img: ``(B, 3, H, W)`` sRGB tensor with values in ``[0, 1]``.

    Returns:
        ``(B, 3, H, W)`` tensor with channels ``(L*, a*, b*)``.
    """
    # --- sRGB gamma expansion ---------------------------------------------
    low = img / 12.92
    # Clamp ONLY the power-branch base: (negative)^2.4 is NaN, and torch.where
    # backward propagates NaN from the unselected branch (NaN * 0 = NaN).
    high = ((torch.clamp(img, min=0.0) + 0.055) / 1.055) ** 2.4
    lin = torch.where(img <= 0.04045, low, high)

    # --- linear RGB -> XYZ --------------------------------------------------
    mat = torch.tensor(_SRGB_TO_XYZ, dtype=img.dtype, device=img.device)
    xyz = torch.einsum("ij,bjhw->bihw", mat, lin)

    # --- normalize by white point ------------------------------------------
    white = torch.tensor(_D65_WHITE, dtype=img.dtype, device=img.device)
    t = xyz / white[None, :, None, None]

    # --- CIE f(t) ------------------------------------------------------------
    # cbrt via sign-safe pow (t >= 0 here, but clamp guards grad at exact 0).
    f_cube = t.clamp(min=_LAB_EPS) ** (1.0 / 3.0)
    f_lin = (_LAB_KAPPA * t + 16.0) / 116.0
    f = torch.where(t > _LAB_EPS, f_cube, f_lin)

    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    lum = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return torch.stack((lum, a, b), dim=1)


# ---------------------------------------------------------------------------
# CIEDE2000 (Sharma, Wu & Dalal 2005)
# ---------------------------------------------------------------------------

_POW7_25 = 25.0 ** 7

#: Epsilon added inside square roots whose radicand can be exactly zero
#: (identical/achromatic pixels): sqrt has an infinite derivative at 0, so
#: the backward pass would otherwise produce NaN at any pixel where the two
#: inputs match bitwise. Forward bias is at most sqrt(eps) = 1e-8 — orders of
#: magnitude below the 1e-3 tolerance of the Sharma reference pairs.
_SQRT_EPS = 1e-16


def _safe_sqrt(x: torch.Tensor) -> torch.Tensor:
    """sqrt with finite gradient at 0 (and guard against tiny negative
    float residue): ``sqrt(clamp(x, 0) + eps)``. Bias <= 1e-8."""
    return torch.sqrt(torch.clamp(x, min=0.0) + _SQRT_EPS)


def ciede2000(lab1: torch.Tensor, lab2: torch.Tensor) -> torch.Tensor:
    """Per-pixel CIEDE2000 color difference ΔE00 (plan §5.3).

    Implements Sharma, Wu & Dalal (2005), "The CIEDE2000 Color-Difference
    Formula: Implementation Notes, Supplementary Test Data, and Mathematical
    Observations", exactly: the G chroma compensation (eq. 4-6), hue angles in
    ``[0°, 360°)`` (eq. 7), the >180° branch logic for the hue difference
    (eq. 10) and mean hue (eq. 14), the T weighting (eq. 15), the Δθ/R_C/R_T
    rotation term (eq. 17-19, 21), and the S_L/S_C/S_H weights (eq. 18-20,
    numbering per paper). Parametric factors k_L = k_C = k_H = 1.

    Computation stays in the input dtype (pass float64 tensors for
    reference-grade precision; float32 is fine for image batches).

    Gradient safety: every square root whose radicand can reach exactly zero
    (identical pixels, achromatic colors) goes through :func:`_safe_sqrt`,
    and the ``atan2`` hue angles are evaluated on a guarded input at exactly
    achromatic pixels (``atan2(0, 0)`` has NaN partials) — so backward is
    finite even when the two inputs match bitwise, at a forward bias of at
    most ``sqrt(_SQRT_EPS)`` = 1e-8.

    Args:
        lab1: ``(B, 3, H, W)`` CIELAB tensor.
        lab2: ``(B, 3, H, W)`` CIELAB tensor.

    Returns:
        ``(B, H, W)`` tensor of ΔE00 values.
    """
    l1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
    l2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]

    # Step 1: C', h' (eq. 2-7).
    c1 = _safe_sqrt(a1 * a1 + b1 * b1)
    c2 = _safe_sqrt(a2 * a2 + b2 * b2)
    c_bar = 0.5 * (c1 + c2)
    c_bar7 = c_bar ** 7
    g = 0.5 * (1.0 - _safe_sqrt(c_bar7 / (c_bar7 + _POW7_25)))
    a1p = (1.0 + g) * a1
    a2p = (1.0 + g) * a2
    c1p = _safe_sqrt(a1p * a1p + b1 * b1)
    c2p = _safe_sqrt(a2p * a2p + b2 * b2)

    # atan2(0, 0) = 0 in torch, matching the paper's h' = 0 convention (eq. 7)
    # — but its PARTIALS at (0, 0) are NaN, so achromatic pixels evaluate
    # atan2(0, 1) instead (same forward value, finite backward).
    achro1 = (a1p == 0.0) & (b1 == 0.0)
    achro2 = (a2p == 0.0) & (b2 == 0.0)
    a1p_safe = torch.where(achro1, torch.ones_like(a1p), a1p)
    a2p_safe = torch.where(achro2, torch.ones_like(a2p), a2p)
    h1p = torch.rad2deg(torch.atan2(b1, a1p_safe)) % 360.0
    h2p = torch.rad2deg(torch.atan2(b2, a2p_safe)) % 360.0

    # Step 2: ΔL', ΔC', Δh', ΔH' (eq. 8-11).
    dl = l2 - l1
    dc = c2p - c1p

    hd = h2p - h1p
    dh = torch.where(hd > 180.0, hd - 360.0, torch.where(hd < -180.0, hd + 360.0, hd))
    chroma_prod = c1p * c2p
    dh = torch.where(chroma_prod == 0.0, torch.zeros_like(dh), dh)
    dhh = 2.0 * _safe_sqrt(chroma_prod) * torch.sin(torch.deg2rad(0.5 * dh))

    # Step 3: means and weights (eq. 12-21).
    l_bar = 0.5 * (l1 + l2)
    cp_bar = 0.5 * (c1p + c2p)

    habs = torch.abs(h1p - h2p)
    hsum = h1p + h2p
    h_bar = torch.where(
        habs <= 180.0,
        0.5 * hsum,
        torch.where(hsum < 360.0, 0.5 * (hsum + 360.0), 0.5 * (hsum - 360.0)),
    )
    h_bar = torch.where(chroma_prod == 0.0, hsum, h_bar)

    h_bar_rad = torch.deg2rad(h_bar)
    t = (
        1.0
        - 0.17 * torch.cos(h_bar_rad - math.radians(30.0))
        + 0.24 * torch.cos(2.0 * h_bar_rad)
        + 0.32 * torch.cos(3.0 * h_bar_rad + math.radians(6.0))
        - 0.20 * torch.cos(4.0 * h_bar_rad - math.radians(63.0))
    )

    dtheta = 30.0 * torch.exp(-(((h_bar - 275.0) / 25.0) ** 2))
    cp_bar7 = cp_bar ** 7
    rc = 2.0 * _safe_sqrt(cp_bar7 / (cp_bar7 + _POW7_25))
    rt = -torch.sin(torch.deg2rad(2.0 * dtheta)) * rc

    l50 = (l_bar - 50.0) ** 2
    sl = 1.0 + 0.015 * l50 / torch.sqrt(20.0 + l50)
    sc = 1.0 + 0.045 * cp_bar
    sh = 1.0 + 0.015 * cp_bar * t

    # Step 4: ΔE00 (eq. 22), k_L = k_C = k_H = 1.
    tl = dl / sl
    tc = dc / sc
    th = dhh / sh
    return _safe_sqrt(tl * tl + tc * tc + th * th + rt * tc * th)


def mean_ciede2000(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Mean CIEDE2000 between two sRGB image batches (plan §5.3).

    Converts both inputs from sRGB to CIELAB internally (:func:`rgb_to_lab`)
    then averages :func:`ciede2000` over batch and pixels.

    Args:
        img1: ``(B, 3, H, W)`` sRGB tensor in ``[0, 1]``.
        img2: ``(B, 3, H, W)`` sRGB tensor in ``[0, 1]``.

    Returns:
        Scalar tensor: mean ΔE00 over batch and pixels.
    """
    return ciede2000(rgb_to_lab(img1), rgb_to_lab(img2)).mean()


# ---------------------------------------------------------------------------
# Equivariance errors
# ---------------------------------------------------------------------------


def _apply_channel_affine(z: torch.Tensor, m_mat: torch.Tensor, m_vec: torch.Tensor) -> torch.Tensor:
    # Channel-affine action on latents, identical to the binding definition of
    # pheq.analytic.apply_channel_affine in SPEC.md (kept local so metrics has
    # no import-time dependency on the sibling module).
    return torch.einsum("dc,bchw->bdhw", m_mat, z) + m_vec[None, :, None, None]


def ee_pix(
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    M: torch.Tensor,
    m: torch.Tensor,
    x_aug: torch.Tensor,
    metric: Union[Literal["l2"], Literal["ciede2000"]] = "l2",
) -> torch.Tensor:
    """Decoder-side equivariance error (plan §5.3, primary EE).

    Computes ``metric(decode_fn(apply_channel_affine(z, M, m)), x_aug)`` — the
    discrepancy between decoding the operator-transformed latent and the
    pixel-space-augmented target τ_a(x).

    Args:
        decode_fn: Maps latents ``(B, C, h, w)`` to images ``(B, 3, H, W)``.
        z: Latent batch ``(B, C, h, w)``.
        M: Channel matrix ``(C, C)``.
        m: Channel bias ``(C,)``.
        x_aug: Target images ``(B, 3, H, W)`` (the pixel-space augmentation).
        metric: ``'l2'`` — mean over batch and pixels of the per-pixel
            Euclidean RGB distance; ``'ciede2000'`` — :func:`mean_ciede2000`.

    Gradient safety: the per-pixel distances use :func:`_safe_sqrt`, so the
    backward pass stays finite at pixels where prediction and target match
    bitwise (sqrt'(0) is infinite; forward bias <= 1e-8 per pixel).

    Returns:
        Scalar tensor.
    """
    x_hat = decode_fn(_apply_channel_affine(z, M, m))
    if metric == "l2":
        diff = x_hat - x_aug
        return _safe_sqrt((diff * diff).sum(dim=1)).mean()
    if metric == "ciede2000":
        return mean_ciede2000(x_hat, x_aug)
    raise ValueError(f"unknown metric {metric!r}; expected 'l2' or 'ciede2000'")


def ee_lat(z_op: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """Latent-side equivariance error ``‖z_op − z_target‖ / ‖z_target‖``.

    DIAGNOSTIC ONLY — never use as a training loss. Collapse caveat (plan
    §3.5): **Never** include ‖T_a(E(x)) − E(τ_a(x))‖² in any training loss
    (EQ-VAE's documented collapse mode); latent-side EE is evaluation-only.

    Args:
        z_op: Operator-transformed latents ``T_a(E(x))``, ``(B, C, h, w)``.
        z_target: Encoded augmented images ``E(τ_a(x))``, ``(B, C, h, w)``.

    Returns:
        Scalar tensor: Frobenius-norm relative error over the whole batch.
    """
    return torch.linalg.vector_norm(z_op - z_target) / torch.linalg.vector_norm(z_target)


# ---------------------------------------------------------------------------
# Hue histogram distance
# ---------------------------------------------------------------------------


def _rgb_to_hue(img: torch.Tensor) -> torch.Tensor:
    # HSV hue channel in [0, 1); hue of achromatic pixels is 0 by convention.
    r, g, b = img[:, 0], img[:, 1], img[:, 2]
    maxc = torch.max(torch.max(r, g), b)
    minc = torch.min(torch.min(r, g), b)
    delta = maxc - minc
    safe = torch.where(delta == 0.0, torch.ones_like(delta), delta)
    hr = ((g - b) / safe) % 6.0
    hg = (b - r) / safe + 2.0
    hb = (r - g) / safe + 4.0
    hue = torch.where(maxc == r, hr, torch.where(maxc == g, hg, hb))
    hue = torch.where(delta == 0.0, torch.zeros_like(hue), hue)
    return hue / 6.0


def hue_histogram_distance(img1: torch.Tensor, img2: torch.Tensor, bins: int = 64) -> torch.Tensor:
    """L1 distance between HSV-hue histograms of two image batches (plan §5.3).

    Hue is the HSV hue channel in ``[0, 1)`` (achromatic pixels contribute
    hue 0). Each batch's pixels are pooled into a single ``bins``-bin
    histogram over ``[0, 1]``, normalized to sum to 1, and the distance is
    ``Σ |p1 − p2|`` (range ``[0, 2]``).

    Args:
        img1: ``(B, 3, H, W)`` sRGB tensor in ``[0, 1]``.
        img2: ``(B, 3, H, W)`` sRGB tensor in ``[0, 1]``.
        bins: Number of histogram bins.

    Returns:
        Scalar tensor.
    """
    h1 = _rgb_to_hue(img1).reshape(-1)
    h2 = _rgb_to_hue(img2).reshape(-1)
    p1 = torch.histc(h1.float(), bins=bins, min=0.0, max=1.0)
    p2 = torch.histc(h2.float(), bins=bins, min=0.0, max=1.0)
    p1 = p1 / p1.sum()
    p2 = p2 / p2.sum()
    return torch.abs(p1 - p2).sum()
