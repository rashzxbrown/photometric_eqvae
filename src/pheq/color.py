"""Canonicalized photometric family: pointwise color-affine maps in Aff(3).

Implements plan §3.1: every photometric factor is a pointwise map
``x_p -> A @ x_p + b`` on pixel colors ``x_p ∈ R^3``. All factor constructors
return an ``(A, b)`` pair with ``A: (3, 3)`` and ``b: (3,)``. Images are
``(B, 3, H, W)`` in ``[0, 1]``; tensors are float32 unless a dtype is requested.

Conventions (binding):
- Luma weights are BT.601 (:data:`LUMA`), matching torchvision Grayscale.
- Contrast is anchored at the FIXED gray ``g = 0.5`` — never the image mean.
  The image-mean version is not a pointwise map and breaks composition
  (plan §3.1).
- ``compose(f, g)`` applies ``f`` first, then ``g`` (result = g∘f).
- ``apply_affine`` defaults to ``clip=False``: equivariance targets are
  computed pre-clip; clipping is *measured* via :func:`clipped_fraction`.

All matrices are constructed in float64 internally and cast to the requested
dtype, so group/monoid identities hold to float64 precision when
``dtype=torch.float64`` is requested.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "LUMA",
    "YIQ_FROM_RGB",
    "RGB_FROM_YIQ",
    "brightness_affine",
    "contrast_affine",
    "saturation_affine",
    "hue_affine",
    "compose",
    "apply_affine",
    "clipped_fraction",
    "PhotoParams",
]

#: BT.601 luma weights (matches torchvision Grayscale). Plan §3.1.
LUMA: tuple[float, float, float] = (0.299, 0.587, 0.114)

#: RGB -> YIQ matrix (FCC NTSC convention). The Y row is :data:`LUMA`; the I
#: and Q chroma rows sum to zero, so the gray vector ``ones(3)`` maps exactly
#: onto the Y (luma) axis — hue rotations therefore fix gray exactly. Plan §3.1.
YIQ_FROM_RGB: torch.Tensor = torch.tensor(
    [
        [0.299, 0.587, 0.114],
        [0.595716, -0.274453, -0.321263],
        [0.211456, -0.522591, 0.311135],
    ],
    dtype=torch.float64,
)

#: YIQ -> RGB matrix, the exact float64 inverse of :data:`YIQ_FROM_RGB`.
RGB_FROM_YIQ: torch.Tensor = torch.linalg.inv(YIQ_FROM_RGB)

_LUMA_T = torch.tensor(LUMA, dtype=torch.float64)


def brightness_affine(
    beta: float, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Brightness factor: ``A = beta * I``, ``b = 0``.

    Abelian group (R_+, x); plan §3.1.
    """
    a = beta * torch.eye(3, dtype=torch.float64)
    return a.to(dtype), torch.zeros(3, dtype=dtype)


def contrast_affine(
    gamma: float, anchor: float = 0.5, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Contrast anchored at fixed gray ``anchor`` (default 0.5, NOT the image
    mean): ``A = gamma * I``, ``b = (1 - gamma) * anchor * ones(3)``.

    With the fixed anchor, composition is exact: (γ1)∘(γ2) = (γ1·γ2) including
    the bias. torchvision's image-mean contrast is not a pointwise map and
    breaks this; plan §3.1.
    """
    a = gamma * torch.eye(3, dtype=torch.float64)
    b = (1.0 - gamma) * anchor * torch.ones(3, dtype=torch.float64)
    return a.to(dtype), b.to(dtype)


def saturation_affine(
    s: float, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Saturation: ``A = s*I + (1-s) * ones(3) @ LUMA^T`` (rank-1 mixing), ``b = 0``.

    Since ``LUMA @ ones(3) = 1``, this is an abelian monoid,
    ``M(s1) M(s2) = M(s1*s2)``, with non-invertible absorbing element ``s = 0``
    (BT.601 grayscale); plan §3.1.
    """
    a = s * torch.eye(3, dtype=torch.float64) + (1.0 - s) * torch.outer(
        torch.ones(3, dtype=torch.float64), _LUMA_T
    )
    return a.to(dtype), torch.zeros(3, dtype=dtype)


def hue_affine(
    theta: float, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hue: rotation by ``theta`` (radians) about the luma axis, via the YIQ
    convention ``A = T_yiq^{-1} @ R_theta @ T_yiq``; ``b = 0``.

    ``R_theta`` rotates the (I, Q) chroma plane and is the identity on Y, so
    ``A`` is a rotation *embedded in GL(3)* (plan §3.1): det(A) = 1,
    ``A(θ1) A(θ2) = A(θ1 + θ2)``, and ``A`` is orthogonal in the YIQ metric
    (``T_yiq A T_yiq^{-1} ∈ SO(3)``), though not in the Euclidean RGB metric
    because ``T_yiq`` is not orthogonal. Because the I and Q rows of
    ``T_yiq`` sum to zero, the gray vector ``ones(3)`` lies on the rotation
    axis and is fixed exactly: ``A @ ones(3) = ones(3)``. Luma is preserved:
    ``LUMA @ A = LUMA``.

    Direction convention (binding — the algebraic identities above hold for
    BOTH rotation signs, so the sign must be pinned separately): positive
    ``theta`` shifts hue in the direction of INCREASING HSV hue,
    red → yellow → green — i.e. it linearizes torchvision's
    ``adjust_hue(img, theta / (2π))`` with a POSITIVE hue factor. Concretely
    the rotation acts as ``(I, Q) ↦ (cos θ · I + sin θ · Q,
    −sin θ · I + cos θ · Q)`` (clockwise in the (I, Q) plane, which is the
    increasing-HSV-hue direction). Pinned by a directional reference test in
    tests/test_color.py.
    """
    c, s = math.cos(theta), math.sin(theta)
    r = torch.eye(3, dtype=torch.float64)
    r[1, 1] = c
    r[1, 2] = s
    r[2, 1] = -s
    r[2, 2] = c
    a = RGB_FROM_YIQ @ r @ YIQ_FROM_RGB
    return a.to(dtype), torch.zeros(3, dtype=dtype)


def compose(
    *ops: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold ``(A_i, b_i)`` pairs left-to-right in homogeneous coordinates.

    Binding order convention (SPEC.md): ``compose(f, g)`` applies ``f`` FIRST,
    then ``g`` — i.e. the result is ``g ∘ f`` with
    ``A = A_g @ A_f`` and ``b = A_g @ b_f + b_g``. Plan §3.1: the four factors
    generate a subgroup/monoid of Aff(3) under this homogeneous-coordinate
    product.

    With no arguments, returns the identity ``(I_3, 0)`` in float32.
    """
    if not ops:
        return torch.eye(3, dtype=torch.float32), torch.zeros(3, dtype=torch.float32)
    a, b = ops[0]
    a = a.clone()
    b = b.clone()
    for a_next, b_next in ops[1:]:
        b = a_next @ b + b_next
        a = a_next @ a
    return a, b


def apply_affine(
    img: torch.Tensor, A: torch.Tensor, b: torch.Tensor, clip: bool = False
) -> torch.Tensor:
    """Apply the pointwise color-affine map ``rgb -> A @ rgb + b`` to an image.

    ``img`` is ``(B, 3, H, W)``. Default ``clip=False``: equivariance targets
    are computed pre-clip (plan §3.1); pass ``clip=True`` to clamp to [0, 1].
    ``A`` and ``b`` are cast to the image's dtype and device.
    """
    a = A.to(dtype=img.dtype, device=img.device)
    bias = b.to(dtype=img.dtype, device=img.device)
    out = torch.einsum("dc,bchw->bdhw", a, img) + bias[None, :, None, None]
    if clip:
        out = out.clamp(0.0, 1.0)
    return out


def clipped_fraction(img: torch.Tensor, A: torch.Tensor, b: torch.Tensor) -> float:
    """Fraction of pixel values outside [0, 1] after the (unclipped) map.

    Plan §3.1: clipping is *measured* per magnitude bin, never asserted rare.
    The fraction is over all ``B * 3 * H * W`` values.
    """
    out = apply_affine(img, A, b, clip=False)
    outside = (out < 0.0) | (out > 1.0)
    return float(outside.to(torch.float32).mean().item())


@dataclass
class PhotoParams:
    """One photometric group element (plan §3.1 ranges, §3.3 coordinates).

    Fields default to the identity: ``beta = gamma = sat = 1.0``, ``hue = 0.0``.
    """

    beta: float = 1.0
    gamma: float = 1.0
    sat: float = 1.0
    hue: float = 0.0

    def phi(self) -> torch.Tensor:
        """Canonical coordinates ``(log beta, log gamma, log sat, hue)`` as a
        float32 ``(4,)`` tensor (plan §3.3).

        ``sat = 0`` (the absorbing grayscale element, outside the group) maps
        to ``-inf``.
        """
        logs = torch.log(
            torch.tensor([self.beta, self.gamma, self.sat], dtype=torch.float32)
        )
        return torch.cat([logs, torch.tensor([self.hue], dtype=torch.float32)])

    def affine(
        self, dtype: torch.dtype = torch.float32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Composed ``(A, b)`` in the fixed factor order
        brightness → contrast → saturation → hue (brightness applied first,
        per the :func:`compose` convention). Plan §3.1.
        """
        return compose(
            brightness_affine(self.beta, dtype=dtype),
            contrast_affine(self.gamma, dtype=dtype),
            saturation_affine(self.sat, dtype=dtype),
            hue_affine(self.hue, dtype=dtype),
        )

    @classmethod
    def sample(
        cls, rng: torch.Generator, n: int, ranges: dict | None = None
    ) -> list["PhotoParams"]:
        """Sample ``n`` params from the plan §3.1 ranges, seeded by ``rng``.

        Default ranges (v1): ``beta, gamma ∈ [0.6, 1.4]``, ``sat ∈ [0.05, 1.5]``,
        ``hue ∈ [-π/4, π/4]``; each factor is active with probability 0.5
        (inactive factors take their identity value). Sampling is vectorized:
        two ``(n, 4)`` draws from ``rng`` regardless of ``n``.

        v2 (SPEC2 design decisions — the single permitted v1 edit): optional
        ``ranges`` maps a subset of the field names
        ``{"beta", "gamma", "sat", "hue"}`` to ``(lo, hi)`` pairs overriding
        the defaults above; unknown keys raise ``ValueError``. ``ranges=None``
        keeps the v1 ranges exactly (backwards-compatible, including bit-exact
        RNG consumption). v2 conditions pass the TIGHTENED ranges explicitly
        (``pheq.conditions.DEFAULT_PHOTO_RANGES``: beta, gamma ∈ [0.7, 1.3]).
        """
        bounds = {
            "beta": (0.6, 1.4),
            "gamma": (0.6, 1.4),
            "sat": (0.05, 1.5),
            "hue": (-math.pi / 4, math.pi / 4),
        }
        if ranges is not None:
            unknown = set(ranges) - set(bounds)
            if unknown:
                raise ValueError(
                    f"unknown range key(s) {sorted(unknown)}; "
                    f"valid keys: {sorted(bounds)}"
                )
            bounds.update(ranges)
        order = ("beta", "gamma", "sat", "hue")
        lo = torch.tensor([bounds[k][0] for k in order], dtype=torch.float64)
        hi = torch.tensor([bounds[k][1] for k in order], dtype=torch.float64)
        u = torch.rand(n, 4, generator=rng, dtype=torch.float64)
        vals = lo + u * (hi - lo)
        active = torch.rand(n, 4, generator=rng, dtype=torch.float64) < 0.5
        identity = torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float64)
        vals = torch.where(active, vals, identity)
        return [
            cls(beta=float(v[0]), gamma=float(v[1]), sat=float(v[2]), hue=float(v[3]))
            for v in vals
        ]
