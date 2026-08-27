"""Tests for pheq/vae.py (SPEC.md "pheq/vae.py" section).

Covers: ToyLinearAE exact round-trips, the planted (W, c) construction, and
fit_w recovery (the latter imports the concurrently developed pheq.analytic
inside the test so the rest of the file stays runnable before integration).
"""

import torch
import torch.nn.functional as F

from pheq.vae import ToyConvAE, ToyLinearAE

SEED = 1234


def _gen() -> torch.Generator:
    return torch.Generator().manual_seed(SEED)


def _smooth_block_images(n: int, size: int, gen: torch.Generator) -> torch.Tensor:
    """Random images that are block-constant on 2x2 blocks AND smooth across
    blocks: low-res color fields bilinear-upsampled to the latent grid, then
    nearest-upsampled to pixels.

    Block-constant puts them exactly in ToyLinearAE's decoder range (exact
    round-trip). fit_w's area downsample IS the 2x2 block-partition average,
    so the planted (W, c) is recoverable with R^2 ~ 1; the cross-block
    smoothness is kept as the easy/benign image regime (the hard,
    high-frequency regime is covered separately below).
    """
    lowres = torch.rand((n, 3, 3, 3), generator=gen) * 0.8 + 0.1
    latent_grid = F.interpolate(lowres, size=(size // 2, size // 2), mode="bilinear", align_corners=False)
    return F.interpolate(latent_grid, scale_factor=2, mode="nearest")


def test_toylinear_encode_decode_is_exact_identity_on_latents() -> None:
    ae = ToyLinearAE()
    z = torch.rand((4, 4, 8, 8), generator=_gen()) * 1.2 - 0.1
    z_rt = ae.encode(ae.decode(z))
    assert torch.allclose(z_rt, z, atol=1e-6)


def test_toylinear_round_trips_exactly_on_random_range_images() -> None:
    # Random images drawn from the decoder's range (see class docstring: with
    # 12->4 per-block compression, exactness on the range is the strongest
    # possible round-trip property; arbitrary pixel noise cannot round-trip).
    ae = ToyLinearAE()
    x = ae.decode(torch.rand((4, 4, 8, 8), generator=_gen()) * 1.2 - 0.1)
    assert torch.allclose(ae(x), x, atol=1e-6)


def test_toylinear_round_trips_exactly_on_random_block_constant_images() -> None:
    ae = ToyLinearAE()
    colors = torch.rand((4, 3, 8, 8), generator=_gen())
    x = F.interpolate(colors, scale_factor=2, mode="nearest")  # block-constant
    assert torch.allclose(ae(x), x, atol=1e-6)


def test_toylinear_decode_is_pointwise_affine_with_planted_w_c() -> None:
    # Direct check of the construction: the 2x2 block average of decode(z)
    # equals W z + c at every latent site (plan section 3.2, exact-linear case).
    ae = ToyLinearAE()
    w, c = ae.true_w()
    assert w.shape == (3, 4) and c.shape == (3,)
    z = torch.rand((4, 4, 8, 8), generator=_gen()) * 2.0 - 0.5
    block_avg = F.avg_pool2d(ae.decode(z), kernel_size=2)
    pred = torch.einsum("dc,bchw->bdhw", w, z) + c.view(1, 3, 1, 1)
    assert torch.allclose(block_avg, pred, atol=1e-6)


def test_toylinear_is_frozen_and_deterministic() -> None:
    ae1, ae2 = ToyLinearAE(), ToyLinearAE()
    assert all(not p.requires_grad for p in ae1.parameters())
    for p1, p2 in zip(ae1.parameters(), ae2.parameters()):
        assert torch.equal(p1, p2)


def test_fit_w_recovers_true_w_on_toylinear_latents() -> None:
    # May be un-runnable until integration: pheq.analytic is a sibling module
    # developed concurrently (import kept inside the test on purpose).
    from pheq.analytic import fit_w

    ae = ToyLinearAE()
    images = _smooth_block_images(8, 64, _gen())
    latents = ae.encode(images)
    fit = fit_w(latents, images)
    w_true, c_true = ae.true_w()

    assert fit.r2 > 0.999
    assert torch.allclose(fit.W, w_true, atol=0.02)
    assert torch.allclose(fit.c, c_true, atol=0.02)


def test_fit_w_unbiased_on_non_smooth_images() -> None:
    """Regression test for the downsample-estimator bias: white-noise latents
    decoded by ToyLinearAE give images with full cross-block high-frequency
    content (checkerboard detail channel active, no cross-block smoothness).
    The box/area downsample keeps each downsampled pixel inside its own
    latent site's block, so the planted (W, c) is recovered essentially
    exactly. An antialiased-bilinear downsample fails this test badly
    (W diagonal ~0.29 instead of 0.5, r2 ~0.90 — ~44% attenuation from
    adjacent-block mixing)."""
    from pheq.analytic import fit_w

    ae = ToyLinearAE()
    z = torch.randn((8, 4, 16, 16), generator=_gen())  # white-noise latents
    images = ae.decode(z)  # in decoder range, spatially rough
    latents = ae.encode(images)
    fit = fit_w(latents, images)
    w_true, c_true = ae.true_w()

    assert fit.r2 > 0.999
    assert torch.allclose(fit.W, w_true, atol=1e-4)
    assert torch.allclose(fit.c, c_true, atol=1e-4)


def test_toyconv_shapes_and_deterministic_seed() -> None:
    ae1, ae2 = ToyConvAE(seed=3), ToyConvAE(seed=3)
    for p1, p2 in zip(ae1.parameters(), ae2.parameters()):
        assert torch.equal(p1, p2)
    x = torch.rand((4, 3, 16, 16), generator=_gen())
    z = ae1.encode(x)
    assert z.shape == (4, 4, 8, 8)
    assert ae1(x).shape == x.shape
    mu, sigma = ae1.encode_moments(x)
    assert torch.equal(mu, z) and torch.count_nonzero(sigma) == 0


def test_load_sd_vae_rescales_to_native_range() -> None:
    """load_sd_vae must feed the SD-VAE its native [-1, 1] range while keeping
    the pheq [0, 1] convention at the API boundary (diffusers'
    VaeImageProcessor convention: 2x - 1 before encode, x/2 + 0.5 after
    decode, no clamp). Verified against a diffusers stub so the test runs
    offline: the stub encoder/decoder are identity maps, making the wrappers'
    rescaling directly observable."""
    import sys
    import types

    calls = {}

    class _Dist:
        def __init__(self, x):
            self.mean = x
            self.std = torch.ones_like(x)

    class _Enc:
        def __init__(self, x):
            self.latent_dist = _Dist(x)

    class _Dec:
        def __init__(self, z):
            self.sample = z

    class _FakeAutoencoderKL(torch.nn.Module):
        @classmethod
        def from_pretrained(cls, name):
            calls["name"] = name
            return cls()

        def encode(self, x):
            calls["encode_input"] = x
            return _Enc(x)

        def decode(self, z):
            return _Dec(z)

    stub = types.ModuleType("diffusers")
    stub.AutoencoderKL = _FakeAutoencoderKL
    saved = sys.modules.get("diffusers")
    sys.modules["diffusers"] = stub
    try:
        from pheq.vae import load_sd_vae

        vae = load_sd_vae(device="cpu")
        assert calls["name"] == "stabilityai/sd-vae-ft-mse"

        img = torch.rand((2, 3, 8, 8), generator=_gen())  # pheq range [0, 1]
        mu, sigma = vae.encode_moments(img)
        # The underlying VAE must see the native [-1, 1] input 2*img - 1.
        assert torch.allclose(calls["encode_input"], 2.0 * img - 1.0, atol=1e-6)
        assert torch.allclose(mu, 2.0 * img - 1.0, atol=1e-6)  # identity stub
        assert sigma.shape == mu.shape

        # Decode must map the VAE's ~[-1, 1] output back to [0, 1], unclamped.
        z = torch.tensor([-1.0, 0.0, 1.0, 3.0]).view(1, 4, 1, 1)
        out = vae.decode_latents(z)
        assert torch.allclose(out, torch.tensor([0.0, 0.5, 1.0, 2.0]).view(1, 4, 1, 1), atol=1e-6)
        assert float(out.max()) > 1.0  # no clamp (pre-clip convention)
    finally:
        if saved is None:
            del sys.modules["diffusers"]
        else:
            sys.modules["diffusers"] = saved


def test_toyconv_trains_in_seconds_on_synthetic_data() -> None:
    torch.manual_seed(SEED)
    ae = ToyConvAE(seed=0)
    x = torch.rand((8, 3, 16, 16), generator=_gen())
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-2)
    first = None
    for _ in range(30):
        optimizer.zero_grad()
        loss = F.mse_loss(ae(x), x)
        if first is None:
            first = loss.item()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.5 * first
