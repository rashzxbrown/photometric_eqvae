"""Tests for pheq.losses per SPEC2.md "losses.py".

Offline-first: the L1-only fallback and kl_loss run with no downloads; the
real-LPIPS branch is guarded by ``pytest.importorskip('lpips')`` semantics
plus a weight-availability check (the VGG16 backbone is fetched from the
torch hub on first use — offline CI must pass, SPEC2 ground rules).
"""

import math
import sys
from pathlib import Path

import pytest
import torch

from pheq.losses import ReconLoss, kl_loss


def _lpips_ready() -> bool:
    """True iff the lpips package imports AND the VGG16 backbone is cached.

    Constructing ``lpips.LPIPS(net='vgg')`` downloads the torchvision VGG16
    weights into the torch hub cache when absent; checking the cache instead
    of constructing keeps this test file strictly offline.
    """
    try:
        import lpips  # noqa: F401
    except Exception:
        return False
    hub = Path(torch.hub.get_dir()) / "checkpoints"
    return hub.is_dir() and any(hub.glob("vgg16-*.pth"))


requires_lpips = pytest.mark.skipif(
    not _lpips_ready(),
    reason="lpips package or cached VGG16 backbone weights unavailable (offline)",
)


# ---------------------------------------------------------------------------
# L1-only fallback (no lpips available)
# ---------------------------------------------------------------------------


def test_fallback_l1_only(monkeypatch: pytest.MonkeyPatch) -> None:
    # None in sys.modules makes `import lpips` raise ImportError,
    # deterministically simulating an unavailable package.
    monkeypatch.setitem(sys.modules, "lpips", None)
    loss = ReconLoss(require_lpips=False)
    assert loss.lpips_active is False
    assert loss.lpips_net is None
    gen = torch.Generator().manual_seed(0)
    target = torch.rand((2, 3, 16, 16), generator=gen)
    pred = torch.rand((2, 3, 16, 16), generator=gen)
    expected = (pred - target).abs().mean()
    assert torch.allclose(loss(pred, target), expected)


def test_fallback_l1_is_unclamped(monkeypatch: pytest.MonkeyPatch) -> None:
    # The [-0.1, 1.1] clamp is LPIPS-only (SPEC2 care point): a constant
    # offset of 10 (far outside [0, 1]) must yield L1 == 10 exactly.
    monkeypatch.setitem(sys.modules, "lpips", None)
    loss = ReconLoss(require_lpips=False)
    target = torch.zeros(1, 3, 8, 8)
    pred = target + 10.0
    assert torch.allclose(loss(pred, target), torch.tensor(10.0))


def test_require_lpips_true_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "lpips", None)
    with pytest.raises(RuntimeError):
        ReconLoss()  # default require_lpips=True


# ---------------------------------------------------------------------------
# kl_loss: takes SIGMA (vae.encode_moments convention), not logvar
# ---------------------------------------------------------------------------


def test_kl_of_standard_normal_moments_is_zero() -> None:
    mu = torch.zeros(4, 4, 8, 8)
    sigma = torch.ones(4, 4, 8, 8)
    assert torch.allclose(kl_loss(mu, sigma), torch.tensor(0.0))


def test_kl_hand_computed() -> None:
    # Per-element KL densities 0.5*(mu^2 + s^2 - 1 - log s^2):
    #   (mu=0.5,  s=2.0): 0.5*(0.25 + 4.00 - 1 - ln 4)    = 0.5*1.86370564
    #   (mu=-1.0, s=0.5): 0.5*(1.00 + 0.25 - 1 + ln 4)    = 0.5*1.63629436
    # ONE batch element with two latent dims: per-batch-element SUM, then
    # mean over the batch of 1 = 0.5 * (1.86370564 + 1.63629436) = 1.75.
    # (The global-mean misreading would give 0.875 — half of this.)
    mu = torch.tensor([[0.5, -1.0]])
    sigma = torch.tensor([[2.0, 0.5]])
    expected = 0.5 * ((0.25 + 4.0 - 1.0 - math.log(4.0)) + (1.0 + 0.25 - 1.0 + math.log(4.0)))
    assert abs(expected - 1.75) < 1e-12  # the hand computation itself
    assert torch.allclose(kl_loss(mu, sigma), torch.tensor(1.75), atol=1e-6)


def test_kl_sums_over_latent_dims_means_over_batch() -> None:
    # Pin the LDM/SD-VAE reduction (sum over C,h,w; mean over B) — the
    # convention lambda_kl = 1e-6 was calibrated for. A global mean over all
    # elements would be C*h*w times smaller and MUST fail this test.
    gen = torch.Generator().manual_seed(0)
    mu = torch.randn((2, 4, 8, 8), generator=gen)
    sigma = 0.3 + torch.rand((2, 4, 8, 8), generator=gen)
    density = 0.5 * (mu.pow(2) + sigma.pow(2) - 1.0 - torch.log(sigma.pow(2)))
    expected = density.sum(dim=(1, 2, 3)).mean()
    got = kl_loss(mu, sigma)
    assert torch.allclose(got, expected, atol=1e-5)
    global_mean = density.mean()
    assert not torch.allclose(got, global_mean, rtol=0.5)  # 256x apart here


def test_kl_positive_for_nonstandard_moments() -> None:
    gen = torch.Generator().manual_seed(1)
    mu = torch.randn((3, 4, 4, 4), generator=gen)
    sigma = 0.3 + torch.rand((3, 4, 4, 4), generator=gen)
    assert float(kl_loss(mu, sigma)) > 0.0


def test_kl_sigma_not_logvar_convention() -> None:
    # If sigma were misread as logvar, kl(0, sigma=1) would be
    # 0.5*(e^1 - 1 - 1) != 0; the sigma convention gives exactly 0
    # (covered above) and a strictly positive value at sigma = e != 1 that
    # matches the sigma formula, not the logvar one.
    mu = torch.zeros(1, 1)
    sigma = torch.full((1, 1), math.e)
    expected_sigma_conv = 0.5 * (math.e**2 - 1.0 - 2.0)
    assert torch.allclose(kl_loss(mu, sigma), torch.tensor(expected_sigma_conv), atol=1e-5)


def test_kl_grad_flows() -> None:
    mu = torch.randn(2, 3, requires_grad=True)
    sigma = torch.rand(2, 3).add(0.5).requires_grad_(True)
    kl_loss(mu, sigma).backward()
    assert mu.grad is not None and torch.isfinite(mu.grad).all()
    assert sigma.grad is not None and torch.isfinite(sigma.grad).all()


# ---------------------------------------------------------------------------
# real LPIPS branch (guarded: skips offline / without cached weights)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def recon_loss() -> ReconLoss:
    pytest.importorskip("lpips")
    if not _lpips_ready():
        pytest.skip("VGG16 backbone weights not cached")
    return ReconLoss()


@requires_lpips
def test_lpips_identical_inputs_zero(recon_loss: ReconLoss) -> None:
    assert recon_loss.lpips_active is True
    gen = torch.Generator().manual_seed(0)
    x = torch.rand((1, 3, 64, 64), generator=gen)
    assert float(recon_loss(x, x)) == pytest.approx(0.0, abs=1e-6)


@requires_lpips
def test_lpips_term_added_and_weighted(recon_loss: ReconLoss) -> None:
    gen = torch.Generator().manual_seed(2)
    target = torch.rand((2, 3, 64, 64), generator=gen)
    pred = (target + 0.3 * torch.randn(target.shape, generator=gen)).clamp(0, 1)
    l1 = (pred - target).abs().mean()
    lp = recon_loss.lpips_term(pred, target)
    assert float(lp) > 0.0
    total = recon_loss(pred, target)
    assert torch.allclose(total, l1 + recon_loss.lambda_lpips * lp, atol=1e-6)
    # lambda_lpips scales the LPIPS term only.
    half = ReconLoss(lambda_lpips=0.5)
    assert torch.allclose(half(pred, target), l1 + 0.5 * lp, atol=1e-6)


@requires_lpips
def test_lpips_out_of_range_inputs_finite(recon_loss: ReconLoss) -> None:
    # Pre-clip targets can leave [0, 1]; the [-0.1, 1.1] clamp on the LPIPS
    # inputs must keep the loss finite even for wildly out-of-range preds.
    gen = torch.Generator().manual_seed(3)
    target = torch.rand((1, 3, 64, 64), generator=gen) * 1.6 - 0.3
    pred = torch.randn((1, 3, 64, 64), generator=gen) * 20.0
    assert torch.isfinite(recon_loss(pred, target))


@requires_lpips
def test_lpips_frozen_and_eval(recon_loss: ReconLoss) -> None:
    assert recon_loss.lpips_net is not None
    assert all(not p.requires_grad for p in recon_loss.lpips_net.parameters())
    recon_loss.train()  # must NOT flip the frozen net into train mode
    assert recon_loss.lpips_net.training is False
    # Gradients still flow THROUGH lpips to the prediction.
    gen = torch.Generator().manual_seed(4)
    target = torch.rand((1, 3, 64, 64), generator=gen)
    pred = torch.rand((1, 3, 64, 64), generator=gen).requires_grad_(True)
    recon_loss(pred, target).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
