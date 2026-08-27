"""Tests for pheq.metrics (SPEC.md 'pheq/metrics.py' section)."""

import math

import pytest
import torch

from pheq.metrics import (
    ciede2000,
    ee_lat,
    ee_pix,
    hue_histogram_distance,
    mean_ciede2000,
    rgb_to_lab,
)

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# CIEDE2000 reference data — Sharma, Wu & Dalal (2005), Table 1
# "The CIEDE2000 Color-Difference Formula: Implementation Notes,
#  Supplementary Test Data, and Mathematical Observations"
# (L1, a1, b1, L2, a2, b2, expected ΔE00), k_L = k_C = k_H = 1.
# Coverage: pairs 1-6 exercise the G compensation and the R_T rotation term
# (blue region, h̄' ≈ 270-275°); pairs 9-15 exercise both branches of the
# Δh' > 180° / mean-hue logic (near-axis colors straddling the a' axis);
# pairs 16-20 exercise the arctan quadrant handling; 21-24 are unit-ΔE00
# pairs; 25-34 are small-magnitude natural pairs.
# ---------------------------------------------------------------------------
SHARMA_PAIRS = [
    (50.0000, 2.6772, -79.7751, 50.0000, 0.0000, -82.7485, 2.0425),
    (50.0000, 3.1571, -77.2803, 50.0000, 0.0000, -82.7485, 2.8615),
    (50.0000, 2.8361, -74.0200, 50.0000, 0.0000, -82.7485, 3.4412),
    (50.0000, -1.3802, -84.2814, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, -1.1848, -84.8006, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, -0.9009, -85.5211, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, 0.0000, 0.0000, 50.0000, -1.0000, 2.0000, 2.3669),
    (50.0000, -1.0000, 2.0000, 50.0000, 0.0000, 0.0000, 2.3669),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0009, 7.1792),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0010, 7.1792),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0011, 7.2195),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0012, 7.2195),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0009, -2.4900, 4.8045),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0010, -2.4900, 4.8045),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0011, -2.4900, 4.7461),
    (50.0000, 2.5000, 0.0000, 50.0000, 0.0000, -2.5000, 4.3065),
    (50.0000, 2.5000, 0.0000, 73.0000, 25.0000, -18.0000, 27.1492),
    (50.0000, 2.5000, 0.0000, 61.0000, -5.0000, 29.0000, 22.8977),
    (50.0000, 2.5000, 0.0000, 56.0000, -27.0000, -3.0000, 31.9030),
    (50.0000, 2.5000, 0.0000, 58.0000, 24.0000, 15.0000, 19.4535),
    (50.0000, 2.5000, 0.0000, 50.0000, 3.1736, 0.5854, 1.0000),
    (50.0000, 2.5000, 0.0000, 50.0000, 3.2972, 0.0000, 1.0000),
    (50.0000, 2.5000, 0.0000, 50.0000, 1.8634, 0.5757, 1.0000),
    (50.0000, 2.5000, 0.0000, 50.0000, 3.2592, 0.3350, 1.0000),
    (60.2574, -34.0099, 36.2677, 60.4626, -34.1751, 39.4387, 1.2644),
    (63.0109, -31.0961, -5.8663, 62.8187, -29.7946, -4.0864, 1.2630),
    (61.2901, 3.7196, -5.3901, 61.4292, 2.2480, -4.9620, 1.8731),
    (35.0831, -44.1164, 3.7933, 35.0232, -40.0716, 1.5901, 1.8645),
    (22.7233, 20.0904, -46.6940, 23.0331, 14.9730, -42.5619, 2.0373),
    (36.4612, 47.8580, 18.3852, 36.2715, 50.5065, 21.2231, 1.4146),
    (90.8027, -2.0831, 1.4410, 91.1528, -1.6435, 0.0447, 1.4441),
    (90.9257, -0.5406, -0.9208, 88.6381, -0.8985, -0.7239, 1.5381),
    (6.7747, -0.2908, -2.4247, 5.8714, -0.0985, -2.2286, 0.6377),
    (2.0776, 0.0795, -1.1350, 0.9033, -0.0636, -0.5514, 0.9082),
]


def _pairs_to_lab_tensors():
    data = torch.tensor(SHARMA_PAIRS, dtype=torch.float64)  # (34, 7)
    lab1 = data[:, 0:3].reshape(-1, 3, 1, 1)
    lab2 = data[:, 3:6].reshape(-1, 3, 1, 1)
    expected = data[:, 6]
    return lab1, lab2, expected


class TestCiede2000:
    def test_sharma_reference_pairs(self):
        """All 34 published pairs to 4 decimals, tolerance 1e-3."""
        lab1, lab2, expected = _pairs_to_lab_tensors()
        de = ciede2000(lab1, lab2).reshape(-1)
        assert de.shape == expected.shape
        err = (de - expected).abs()
        assert err.max().item() < 1e-3, (
            f"max |ΔE00 error| = {err.max().item():.6f} at pair "
            f"{int(err.argmax().item()) + 1}"
        )

    def test_hue_branch_pairs_differ(self):
        """Pairs 10 vs 11 (and 14 vs 15) differ ONLY in the h' branch taken."""
        lab1, lab2, _ = _pairs_to_lab_tensors()
        de = ciede2000(lab1, lab2).reshape(-1)
        assert abs(de[9].item() - 7.1792) < 1e-3
        assert abs(de[10].item() - 7.2195) < 1e-3
        assert abs(de[13].item() - 4.8045) < 1e-3
        assert abs(de[14].item() - 4.7461) < 1e-3

    def test_output_shape(self):
        lab1 = torch.randn(2, 3, 5, 7, dtype=torch.float64) * 20.0
        lab2 = torch.randn(2, 3, 5, 7, dtype=torch.float64) * 20.0
        assert ciede2000(lab1, lab2).shape == (2, 5, 7)

    def test_identical_inputs_zero(self):
        lab = torch.tensor([50.0, 10.0, -10.0]).reshape(1, 3, 1, 1)
        assert ciede2000(lab, lab).abs().max().item() < 1e-6

    def test_achromatic_pair_no_nan(self):
        """Both colors on the gray axis: C' = 0, ΔE00 = ΔL'/S_L, no NaN."""
        lab1 = torch.tensor([50.0, 0.0, 0.0], dtype=torch.float64).reshape(1, 3, 1, 1)
        lab2 = torch.tensor([60.0, 0.0, 0.0], dtype=torch.float64).reshape(1, 3, 1, 1)
        de = ciede2000(lab1, lab2)
        assert torch.isfinite(de).all()
        # ΔL' = 10, L̄' = 55, S_L = 1 + 0.015*25/sqrt(45)
        expected = 10.0 / (1.0 + 0.015 * 25.0 / math.sqrt(45.0))
        assert abs(de.item() - expected) < 1e-6


class TestRgbToLab:
    def test_white(self):
        img = torch.ones(1, 3, 1, 1, dtype=torch.float64)
        lab = rgb_to_lab(img)
        assert abs(lab[0, 0, 0, 0].item() - 100.0) < 1e-2
        assert abs(lab[0, 1, 0, 0].item()) < 1e-2
        assert abs(lab[0, 2, 0, 0].item()) < 1e-2

    def test_black(self):
        img = torch.zeros(1, 3, 1, 1, dtype=torch.float64)
        lab = rgb_to_lab(img)
        assert abs(lab[0, 0, 0, 0].item()) < 1e-6
        assert abs(lab[0, 1, 0, 0].item()) < 1e-6
        assert abs(lab[0, 2, 0, 0].item()) < 1e-6

    def test_mid_gray(self):
        """sRGB (0.5, 0.5, 0.5) — skimage rgb2lab reference: L = 53.3890."""
        img = torch.full((1, 3, 1, 1), 0.5, dtype=torch.float64)
        lab = rgb_to_lab(img)
        assert abs(lab[0, 0, 0, 0].item() - 53.3890) < 5e-3
        assert abs(lab[0, 1, 0, 0].item()) < 1e-2
        assert abs(lab[0, 2, 0, 0].item()) < 1e-2

    def test_red(self):
        """sRGB (1, 0, 0) — skimage/colormath reference (53.2408, 80.0925, 67.2032)."""
        img = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64).reshape(1, 3, 1, 1)
        lab = rgb_to_lab(img)
        assert abs(lab[0, 0, 0, 0].item() - 53.2408) < 5e-3
        assert abs(lab[0, 1, 0, 0].item() - 80.0925) < 5e-3
        assert abs(lab[0, 2, 0, 0].item() - 67.2032) < 5e-3

    def test_shape_and_no_inplace(self):
        torch.manual_seed(1)
        img = torch.rand(2, 3, 4, 5)
        img_copy = img.clone()
        lab = rgb_to_lab(img)
        assert lab.shape == (2, 3, 4, 5)
        assert torch.equal(img, img_copy)  # no in-place modification

    def test_differentiable(self):
        torch.manual_seed(2)
        img = torch.rand(1, 3, 4, 4, dtype=torch.float64, requires_grad=True)
        loss = mean_ciede2000(img, img.detach() * 0.9 + 0.05)
        loss.backward()
        assert img.grad is not None
        assert torch.isfinite(img.grad).all()

    def test_gradient_finite_on_out_of_range_input(self):
        """Pre-clip pipeline values go below 0 (plan §3.1); for pixels below
        -0.055 the unselected power branch used to produce NaN * 0 = NaN in
        backward. Forward must be unchanged; gradient must be finite."""
        img = torch.tensor([-0.2, 0.5, 0.5]).reshape(1, 3, 1, 1).requires_grad_(True)
        lab = rgb_to_lab(img)
        assert torch.isfinite(lab).all()
        lab.sum().backward()
        assert torch.isfinite(img.grad).all()

        img2 = torch.tensor([-0.2, 0.5, 0.5]).reshape(1, 3, 1, 1).requires_grad_(True)
        loss = mean_ciede2000(img2, torch.full((1, 3, 1, 1), 0.5))
        loss.backward()
        assert torch.isfinite(img2.grad).all()


class TestMeanCiede2000:
    def test_scalar_and_zero_on_identical(self):
        torch.manual_seed(3)
        img = torch.rand(2, 3, 8, 8)
        out = mean_ciede2000(img, img)
        assert out.ndim == 0
        assert out.abs().item() < 1e-4

    def test_positive_on_different(self):
        torch.manual_seed(4)
        img1 = torch.rand(2, 3, 8, 8)
        img2 = (img1 * 0.7 + 0.1).clamp(0, 1)
        assert mean_ciede2000(img1, img2).item() > 0.1


class TestEePix:
    def test_l2_known_value(self):
        """Identity decoder, identity operator, target shifted by c per channel
        → per-pixel Euclidean distance = c * sqrt(3) exactly, scalar output."""
        torch.manual_seed(5)
        z = torch.rand(2, 3, 4, 4)
        M = torch.eye(3)
        m = torch.zeros(3)
        c = 0.25
        x_aug = z + c
        err = ee_pix(lambda t: t, z, M, m, x_aug, metric="l2")
        assert err.ndim == 0
        assert abs(err.item() - c * math.sqrt(3.0)) < 1e-6

    def test_zero_when_operator_matches(self):
        """decode(M z + m) == x_aug exactly → EE = 0 for both metrics."""
        torch.manual_seed(6)
        z = torch.rand(2, 3, 4, 4) * 0.5 + 0.25
        M = 0.8 * torch.eye(3)
        m = 0.05 * torch.ones(3)
        x_aug = torch.einsum("dc,bchw->bdhw", M, z) + m[None, :, None, None]
        assert ee_pix(lambda t: t, z, M, m, x_aug, metric="l2").item() < 1e-6
        assert ee_pix(lambda t: t, z, M, m, x_aug, metric="ciede2000").item() < 1e-3

    def test_ciede2000_metric_scalar(self):
        torch.manual_seed(7)
        z = torch.rand(2, 3, 4, 4) * 0.5 + 0.25
        x_aug = torch.rand(2, 3, 4, 4)
        err = ee_pix(lambda t: t.clamp(0, 1), z, torch.eye(3), torch.zeros(3), x_aug, metric="ciede2000")
        assert err.ndim == 0
        assert err.item() >= 0.0

    def test_unknown_metric_raises(self):
        z = torch.rand(1, 3, 2, 2)
        with pytest.raises(ValueError):
            ee_pix(lambda t: t, z, torch.eye(3), torch.zeros(3), z, metric="lpips")

    def test_gradient_finite_at_exact_match(self):
        """sqrt'(0) is infinite, so pixels where prediction equals target
        BITWISE (identical inputs, flat saturated regions) used to make the
        gradient NaN for both metrics despite a finite forward value."""
        torch.manual_seed(12)
        z = torch.rand(2, 3, 4, 4) * 0.5 + 0.25
        x_aug = z.clone()  # exact match at every pixel
        for metric in ("l2", "ciede2000"):
            z_in = z.clone().requires_grad_(True)
            err = ee_pix(lambda t: t, z_in, torch.eye(3), torch.zeros(3), x_aug, metric=metric)
            assert torch.isfinite(err)
            err.backward()
            assert torch.isfinite(z_in.grad).all(), metric

    def test_gradient_finite_at_achromatic_exact_match(self):
        """Gray pixels additionally hit C' = 0 (sqrt at 0 in the chroma terms)
        and atan2(0, 0) (NaN partials) inside ciede2000."""
        img = torch.full((1, 3, 2, 2), 0.5).requires_grad_(True)
        loss = mean_ciede2000(img, torch.full((1, 3, 2, 2), 0.5))
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(img.grad).all()


class TestEeLat:
    def test_zero_on_equal(self):
        torch.manual_seed(8)
        z = torch.randn(2, 4, 8, 8)
        out = ee_lat(z, z)
        assert out.ndim == 0
        assert out.item() == 0.0

    def test_relative_normalization(self):
        """ee_lat(z + d, z) = ||d|| / ||z||; invariant to global rescaling."""
        torch.manual_seed(9)
        z = torch.randn(2, 4, 8, 8)
        d = torch.randn(2, 4, 8, 8) * 0.1
        expected = (d.norm() / z.norm()).item()
        assert abs(ee_lat(z + d, z).item() - expected) < 1e-6
        scaled = ee_lat(3.0 * (z + d), 3.0 * z).item()
        assert abs(scaled - expected) < 1e-6

    def test_docstring_carries_collapse_caveat(self):
        doc = ee_lat.__doc__
        assert doc is not None
        assert "DIAGNOSTIC ONLY" in doc
        assert "never use as a training loss" in doc.lower()
        assert "collapse" in doc.lower()


class TestHueHistogramDistance:
    def test_identical_zero(self):
        torch.manual_seed(10)
        img = torch.rand(2, 3, 8, 8)
        assert hue_histogram_distance(img, img).item() < 1e-6

    def test_disjoint_hues_max_distance(self):
        """Pure red vs pure green images: disjoint histograms → L1 = 2."""
        red = torch.zeros(1, 3, 8, 8)
        red[:, 0] = 1.0
        green = torch.zeros(1, 3, 8, 8)
        green[:, 1] = 1.0
        d = hue_histogram_distance(red, green, bins=64)
        assert abs(d.item() - 2.0) < 1e-6

    def test_scalar_and_bins(self):
        torch.manual_seed(11)
        img1 = torch.rand(2, 3, 8, 8)
        img2 = torch.rand(2, 3, 8, 8)
        d = hue_histogram_distance(img1, img2, bins=16)
        assert d.ndim == 0
        assert 0.0 <= d.item() <= 2.0
