"""Tests for pheq.color per SPEC.md "pheq/color.py".

Notes on the hue "rotation" checks: the spec mandates the YIQ construction
A = T^{-1} R T. Because T_yiq is not Euclidean-orthogonal (its chroma rows sum
to zero to fix gray, which is incompatible with orthogonality to the luma row),
A is a rotation *embedded in GL(3)* (plan §3.1): we assert det(A) = 1, the
one-parameter homomorphism, unit-modulus eigenvalues, and exact orthogonality
in the YIQ metric (T A T^{-1} ∈ SO(3)) — plus the gray-vector fix A @ 1 = 1.
"""

import math

import torch

from pheq.color import (
    LUMA,
    RGB_FROM_YIQ,
    YIQ_FROM_RGB,
    PhotoParams,
    apply_affine,
    brightness_affine,
    clipped_fraction,
    compose,
    contrast_affine,
    hue_affine,
    saturation_affine,
)

F64 = torch.float64


# ---------------------------------------------------------------------------
# brightness / contrast composition (exact, incl. bias)
# ---------------------------------------------------------------------------


def test_brightness_compose_exact() -> None:
    b1, b2 = 0.7, 1.3
    a_comp, bias_comp = compose(brightness_affine(b1, F64), brightness_affine(b2, F64))
    a_ref, bias_ref = brightness_affine(b1 * b2, F64)
    assert torch.allclose(a_comp, a_ref, atol=1e-14, rtol=0)
    assert torch.allclose(bias_comp, bias_ref, atol=1e-14, rtol=0)


def test_contrast_compose_exact_including_bias() -> None:
    g1, g2 = 0.8, 1.25
    a_comp, b_comp = compose(contrast_affine(g1, dtype=F64), contrast_affine(g2, dtype=F64))
    a_ref, b_ref = contrast_affine(g1 * g2, dtype=F64)
    assert torch.allclose(a_comp, a_ref, atol=1e-14, rtol=0)
    # the bias must also collapse: (1 - g1*g2) * 0.5 * ones
    assert torch.allclose(b_comp, b_ref, atol=1e-14, rtol=0)


def test_compose_order_convention_f_first() -> None:
    # brightness and contrast do NOT commute (bias differs), so this pins the
    # binding convention: compose(f, g) applies f first (result = g∘f).
    f = brightness_affine(1.3, F64)
    g = contrast_affine(0.7, dtype=F64)
    a, b = compose(f, g)
    x = torch.tensor([0.2, 0.5, 0.9], dtype=F64)
    fx = f[0] @ x + f[1]
    gfx = g[0] @ fx + g[1]
    assert torch.allclose(a @ x + b, gfx, atol=1e-14, rtol=0)
    # and it differs from f∘g
    gx = g[0] @ x + g[1]
    fgx = f[0] @ gx + f[1]
    assert not torch.allclose(a @ x + b, fgx, atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------
# saturation monoid
# ---------------------------------------------------------------------------


def test_saturation_monoid_float64() -> None:
    for s1, s2 in [(0.3, 0.8), (1.2, 1.4), (0.05, 1.5), (0.0, 0.7), (0.9, 0.0)]:
        m1, _ = saturation_affine(s1, F64)
        m2, _ = saturation_affine(s2, F64)
        m12, _ = saturation_affine(s1 * s2, F64)
        assert torch.allclose(m1 @ m2, m12, atol=1e-14, rtol=0), (s1, s2)
        assert torch.allclose(m2 @ m1, m12, atol=1e-14, rtol=0), (s1, s2)


def test_saturation_zero_absorbing_idempotent() -> None:
    m0, _ = saturation_affine(0.0, F64)
    assert torch.allclose(m0 @ m0, m0, atol=1e-14, rtol=0)
    ms, _ = saturation_affine(0.6, F64)
    assert torch.allclose(ms @ m0, m0, atol=1e-14, rtol=0)
    assert torch.allclose(m0 @ ms, m0, atol=1e-14, rtol=0)


def test_saturation_identity_at_one() -> None:
    m1, b = saturation_affine(1.0, F64)
    assert torch.allclose(m1, torch.eye(3, dtype=F64), atol=1e-14, rtol=0)
    assert torch.equal(b, torch.zeros(3, dtype=F64))


# ---------------------------------------------------------------------------
# hue rotation
# ---------------------------------------------------------------------------


def test_hue_is_rotation() -> None:
    for theta in [0.0, 0.3, -math.pi / 4, math.pi / 4]:
        a, b = hue_affine(theta, F64)
        assert torch.equal(b, torch.zeros(3, dtype=F64))
        # det(A) = 1
        assert abs(float(torch.linalg.det(a)) - 1.0) < 1e-12
        # orthogonal in the YIQ metric: T A T^{-1} is exactly R_theta ∈ SO(3)
        r = YIQ_FROM_RGB @ a @ RGB_FROM_YIQ
        assert torch.allclose(r.T @ r, torch.eye(3, dtype=F64), atol=1e-12, rtol=0)
        # identity on the Y (luma) coordinate
        assert torch.allclose(r[0], torch.tensor([1.0, 0.0, 0.0], dtype=F64), atol=1e-12, rtol=0)
        # rotation similarity: all eigenvalues on the unit circle
        eigmod = torch.abs(torch.linalg.eigvals(a))
        assert torch.allclose(eigmod, torch.ones(3, dtype=F64), atol=1e-12, rtol=0)


def test_hue_one_parameter_homomorphism() -> None:
    t1, t2 = 0.35, -0.6
    a1, _ = hue_affine(t1, F64)
    a2, _ = hue_affine(t2, F64)
    a12, _ = hue_affine(t1 + t2, F64)
    assert torch.allclose(a1 @ a2, a12, atol=1e-12, rtol=0)
    assert torch.allclose(a2 @ a1, a12, atol=1e-12, rtol=0)


def test_hue_identity_at_zero() -> None:
    a, _ = hue_affine(0.0, F64)
    assert torch.allclose(a, torch.eye(3, dtype=F64), atol=1e-12, rtol=0)


def test_hue_fixes_gray_vector() -> None:
    ones = torch.ones(3, dtype=F64)
    for theta in [0.3, -math.pi / 4, math.pi / 4, 1.0]:
        a, _ = hue_affine(theta, F64)
        assert torch.allclose(a @ ones, ones, atol=1e-12, rtol=0)
    # float32 default dtype too
    a32, _ = hue_affine(0.5)
    assert torch.allclose(a32 @ torch.ones(3), torch.ones(3), atol=1e-6, rtol=0)


def test_hue_preserves_luma() -> None:
    w = torch.tensor(LUMA, dtype=F64)
    for theta in [0.3, -0.7]:
        a, _ = hue_affine(theta, F64)
        assert torch.allclose(w @ a, w, atol=1e-12, rtol=0)


def _hsv_hue(rgb: list) -> float:
    """HSV hue in [0, 1) from the standard definition (independent reference)."""
    r, g, b = rgb
    mx, mn = max(rgb), min(rgb)
    d = mx - mn
    if d == 0:
        return 0.0
    if mx == r:
        return (((g - b) / d) % 6.0) / 6.0
    if mx == g:
        return ((b - r) / d + 2.0) / 6.0
    return ((r - g) / d + 4.0) / 6.0


def test_hue_direction_reference_values() -> None:
    """Directional pin: every other hue assertion in this file also passes for
    the sign-flipped rotation (theta -> -theta), so the DIRECTION must be
    pinned by reference values. Convention (hue_affine docstring): positive
    theta = torchvision adjust_hue with POSITIVE factor theta/(2*pi), i.e.
    increasing HSV hue (red -> yellow -> green). A(0.3) @ e_R hardcoded to
    4 decimals from the pinned implementation."""
    a, _ = hue_affine(0.3, F64)
    red = torch.tensor([1.0, 0.0, 0.0], dtype=F64)
    expected = torch.tensor([0.9191, 0.1103, -0.3559], dtype=F64)
    assert torch.allclose(a @ red, expected, atol=1e-4, rtol=0)
    # Single sign pin (the one-parameter homomorphism forces the rest): the
    # R -> G leak must be POSITIVE for positive theta (red rotates toward
    # yellow/green, never toward magenta/blue).
    assert float(a[1, 0]) > 0.0


def test_hue_direction_matches_hsv_hue_shift() -> None:
    """Positive theta must INCREASE the HSV hue of an in-gamut color by
    approximately theta / (2*pi) (torchvision adjust_hue convention; the YIQ
    rotation is its linearization, so the match is approximate)."""
    x = torch.tensor([0.6, 0.4, 0.3], dtype=F64)
    theta = 0.15
    a, _ = hue_affine(theta, F64)
    h0 = _hsv_hue([float(v) for v in x])
    h1 = _hsv_hue([float(v) for v in (a @ x)])
    shift = (h1 - h0 + 0.5) % 1.0 - 0.5  # signed circular difference
    expected = theta / (2.0 * math.pi)
    assert shift > 0.0, "positive theta must increase HSV hue"
    assert abs(shift - expected) < 0.35 * expected  # linearization tolerance


# ---------------------------------------------------------------------------
# image-level application
# ---------------------------------------------------------------------------


def test_saturation_zero_is_bt601_grayscale() -> None:
    torch.manual_seed(0)
    img = torch.rand(2, 3, 5, 7)
    a, b = saturation_affine(0.0)
    out = apply_affine(img, a, b)
    luma = LUMA[0] * img[:, 0] + LUMA[1] * img[:, 1] + LUMA[2] * img[:, 2]
    expected = luma[:, None, :, :].expand(-1, 3, -1, -1)
    assert torch.allclose(out, expected, atol=1e-6, rtol=0)


def test_apply_affine_matches_pixel_loop() -> None:
    torch.manual_seed(1)
    img = torch.rand(2, 3, 2, 3)
    a = torch.randn(3, 3)
    b = torch.randn(3)
    out = apply_affine(img, a, b)
    expected = torch.empty_like(img)
    for n in range(img.shape[0]):
        for i in range(img.shape[2]):
            for j in range(img.shape[3]):
                expected[n, :, i, j] = a @ img[n, :, i, j] + b
    assert torch.allclose(out, expected, atol=1e-6, rtol=0)


def test_apply_affine_clip() -> None:
    img = torch.tensor([0.1, 0.5, 0.9]).view(1, 3, 1, 1)
    a, b = brightness_affine(2.0)
    unclipped = apply_affine(img, a, b, clip=False)
    assert float(unclipped.max()) > 1.0  # pre-clip target keeps overshoot
    clipped = apply_affine(img, a, b, clip=True)
    assert float(clipped.min()) >= 0.0 and float(clipped.max()) <= 1.0
    assert torch.allclose(clipped, unclipped.clamp(0.0, 1.0), atol=0, rtol=0)


def test_clipped_fraction() -> None:
    # half the values map outside [0, 1] under brightness 1.2
    img = torch.tensor([0.9, 0.9, 0.9, 0.1, 0.1, 0.1]).view(1, 3, 2, 1)
    a, b = brightness_affine(1.2)
    assert abs(clipped_fraction(img, a, b) - 0.5) < 1e-6
    a_id, b_id = brightness_affine(1.0)
    assert clipped_fraction(img, a_id, b_id) == 0.0


# ---------------------------------------------------------------------------
# PhotoParams
# ---------------------------------------------------------------------------


def test_photoparams_default_is_identity() -> None:
    p = PhotoParams()
    a, b = p.affine(dtype=F64)
    assert torch.allclose(a, torch.eye(3, dtype=F64), atol=1e-12, rtol=0)
    assert torch.allclose(b, torch.zeros(3, dtype=F64), atol=1e-12, rtol=0)
    assert torch.allclose(p.phi(), torch.zeros(4), atol=0, rtol=0)


def test_photoparams_phi() -> None:
    p = PhotoParams(beta=2.0, gamma=0.5, sat=1.0, hue=0.3)
    expected = torch.tensor([math.log(2.0), math.log(0.5), 0.0, 0.3])
    assert torch.allclose(p.phi(), expected, atol=1e-7, rtol=0)
    assert p.phi().dtype == torch.float32


def test_photoparams_affine_factor_order() -> None:
    # brightness -> contrast -> saturation -> hue, brightness applied FIRST
    p = PhotoParams(beta=1.2, gamma=0.8, sat=0.5, hue=0.3)
    a, b = p.affine(dtype=F64)
    x = torch.tensor([0.2, 0.6, 0.9], dtype=F64)
    y = x.clone()
    for fa, fb in (
        brightness_affine(p.beta, F64),
        contrast_affine(p.gamma, dtype=F64),
        saturation_affine(p.sat, F64),
        hue_affine(p.hue, F64),
    ):
        y = fa @ y + fb
    assert torch.allclose(a @ x + b, y, atol=1e-12, rtol=0)


def test_photoparams_sample_seedable() -> None:
    rng1 = torch.Generator().manual_seed(1234)
    rng2 = torch.Generator().manual_seed(1234)
    ps1 = PhotoParams.sample(rng1, 32)
    ps2 = PhotoParams.sample(rng2, 32)
    assert len(ps1) == 32
    assert ps1 == ps2  # dataclass equality, bit-identical under the same seed
    rng3 = torch.Generator().manual_seed(9999)
    ps3 = PhotoParams.sample(rng3, 32)
    assert ps1 != ps3


def test_photoparams_sample_ranges_and_gating() -> None:
    rng = torch.Generator().manual_seed(0)
    ps = PhotoParams.sample(rng, 256)
    n_active = {"beta": 0, "gamma": 0, "sat": 0, "hue": 0}
    for p in ps:
        for name, val, lo, hi, ident in (
            ("beta", p.beta, 0.6, 1.4, 1.0),
            ("gamma", p.gamma, 0.6, 1.4, 1.0),
            ("sat", p.sat, 0.05, 1.5, 1.0),
            ("hue", p.hue, -math.pi / 4, math.pi / 4, 0.0),
        ):
            if val == ident:
                continue  # inactive factor: exactly the identity value
            n_active[name] += 1
            assert lo <= val <= hi, (name, val)
    # each factor is active ~w.p. 0.5: both branches must occur in 256 draws
    for name, count in n_active.items():
        assert 0 < count < 256, (name, count)
