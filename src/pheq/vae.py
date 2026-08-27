"""Autoencoder access: SD-VAE loading plus toy autoencoders for offline tests.

Implements the ``pheq/vae.py`` section of SPEC.md.

- :func:`load_sd_vae` wraps the pretrained SD-VAE (``stabilityai/sd-vae-ft-mse``)
  whose latents admit the approximate linear color decoding of
  docs/research-plan.md §3.2 (the "latent RGB preview" map ``x̂_p ≈ W z_p + c``).
- :class:`ToyLinearAE` is an EXACTLY linear-affine autoencoder: it instantiates
  the exact-linear-decoder setting under which the closed-form operator of
  plan §3.2 is exact, with the planted ``(W, c)`` known analytically.
- :class:`ToyConvAE` is a small nonlinear conv AE used as a download-free
  stand-in for pipeline/smoke tests.

Conventions (SPEC.md): images are ``(B, 3, H, W)`` in ``[0, 1]``,
latents are ``(B, C, h, w)``, float32.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def load_sd_vae(device: str = "cpu") -> Any:
    """Load the pretrained SD-VAE ``stabilityai/sd-vae-ft-mse`` in eval mode.

    diffusers is imported lazily inside this function so that ``pheq``
    imports without it installed.

    Two helpers are attached to the returned model instance:

    - ``vae.encode_moments(img) -> (mu, sigma)``: posterior moments of the
      diagonal Gaussian encoder (plan §3.4 push-forward operates on these).
    - ``vae.decode_latents(z) -> img``: decode latents to RGB.

    Range convention: the pheq API boundary uses images in ``[0, 1]``
    (SPEC.md), but ``stabilityai/sd-vae-ft-mse`` is trained on inputs
    normalized to ``[-1, 1]`` (diffusers' ``VaeImageProcessor`` applies
    ``2x - 1`` before ``encode`` and ``x/2 + 0.5`` after ``decode``). The
    helpers perform exactly that rescaling, so callers keep the ``[0, 1]``
    convention and the VAE sees its native range. ``decode_latents`` does
    NOT clamp its output (pre-clip convention, plan §3.1).

    The ``decode_latents`` helper additionally carries a ``module`` attribute
    referencing the underlying ``AutoencoderKL`` so that
    :func:`pheq.oracle.invert_latent` can locate and freeze decoder
    parameters when handed the bare helper.

    Args:
        device: torch device string, e.g. ``'cpu'`` or ``'mps'``.

    Returns:
        The ``diffusers.AutoencoderKL`` instance (eval mode, on ``device``)
        with the two helpers attached.
    """
    from diffusers import AutoencoderKL  # lazy: package must import without diffusers

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    vae = vae.to(device).eval()

    def encode_moments(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # pheq images are [0, 1]; the SD-VAE's native input range is [-1, 1].
        posterior = vae.encode(2.0 * img - 1.0).latent_dist
        return posterior.mean, posterior.std

    def decode_latents(z: torch.Tensor) -> torch.Tensor:
        # SD-VAE decodes to ~[-1, 1]; map back to the pheq [0, 1] convention
        # without clamping (equivariance targets are pre-clip, plan §3.1).
        return vae.decode(z).sample / 2.0 + 0.5

    decode_latents.module = vae  # type: ignore[attr-defined]
    encode_moments.module = vae  # type: ignore[attr-defined]
    vae.encode_moments = encode_moments
    vae.decode_latents = decode_latents
    return vae


class ToyLinearAE(nn.Module):
    """An EXACTLY linear-affine toy autoencoder (C=4, f=2), for offline tests.

    Construction (documented per SPEC.md "pheq/vae.py"):

    The encoder is a fixed ``Conv2d(3, 4, kernel_size=2, stride=2)`` and the
    decoder a fixed ``ConvTranspose2d(4, 3, kernel_size=2, stride=2)``.
    Because stride equals kernel size, image pixels partition into disjoint
    2x2 blocks, each block interacting with exactly one latent site. Stack a
    block's pixels as a vector ``p ∈ R^12`` (3 channels x 4 positions). The
    decoder weight realizes ``p = V z + b̃_d`` with ``V ∈ R^{12x4}`` having
    ORTHONORMAL columns:

    - ``v_R, v_G, v_B``: value 1/2 on one RGB channel at all 4 positions
      ("flat" color directions; norm ``sqrt(4 * 1/4) = 1``).
    - ``v_D``: a detail direction, value ±1/2 on the R channel with the
      2x2 checkerboard sign pattern ``(+, -; -, +)`` (orthogonal to the flat
      directions since it sums to zero over the block).

    ``b̃_d`` is the per-channel decoder bias ``b_d ∈ R^3`` broadcast over the
    block; it lies in ``span(v_R, v_G, v_B)``. The encoder weight is ``Vᵀ``
    and its bias is ``b_e = -Vᵀ b̃_d = (-2 b_d, 0)``, so:

    - ``encode(decode(z)) = Vᵀ(V z + b̃_d) + b_e = z`` exactly (``VᵀV = I_4``),
    - ``decode(encode(x)) = V Vᵀ x - V Vᵀ b̃_d + b̃_d = x`` exactly for every
      image in the decoder's range (the affine subspace ``range(V) + b̃_d``;
      12→4 per-block compression makes exactness on arbitrary images
      impossible, so round-trip tests draw images from the range —
      e.g. ``decode`` of random latents, or block-constant color images,
      which lie in ``span(v_R, v_G, v_B) + b̃_d``).

    ``D(z)`` is pointwise-affine in ``z`` on the latent grid: the block
    average of ``decode(z)`` at latent site ``(i, j)`` equals ``W z_ij + c``
    with the planted, analytically known values

        ``W = [0.5 * I_3 | 0] ∈ R^{3x4}``,   ``c = b_d``,

    because averaging the flat directions gives 1/2 per channel and the
    checkerboard direction averages to zero (hence the structural zero fourth
    column: ANY orthonormal completion of the flat directions has zero block
    mean, so the detail channel never contributes to the downsampled RGB).
    This is the exact-linear-decoder setting of plan §3.2, in which the
    closed-form operator ``M_a = W⁺ A_a W + (I - W⁺W) K`` is exactly
    equivariant, and ``fit_w`` (pheq/analytic.py) can recover ``(W, c)``.

    All parameters are fixed (``requires_grad=False``); the construction is
    fully deterministic (no randomness). Input H, W must be even.
    """

    LATENT_CHANNELS: int = 4
    DOWNSAMPLE: int = 2

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, kernel_size=2, stride=2, bias=True)
        self.decoder = nn.ConvTranspose2d(4, 3, kernel_size=2, stride=2, bias=True)

        # Rows of Vᵀ (encoder) == columns of V (decoder); identical tensor
        # layout for both: (4, 3, 2, 2) = Conv2d (out, in, kh, kw)
        # and ConvTranspose2d (in, out, kh, kw).
        weight = torch.zeros(4, 3, 2, 2)
        weight[0, 0] = 0.5  # v_R: flat over the block on channel R
        weight[1, 1] = 0.5  # v_G
        weight[2, 2] = 0.5  # v_B
        weight[3, 0] = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])  # v_D checkerboard

        bias_d = torch.tensor([0.10, -0.05, 0.02])  # c in the planted (W, c)
        bias_e = torch.zeros(4)
        bias_e[:3] = -2.0 * bias_d  # -Vᵀ b̃_d (checkerboard component is 0)

        with torch.no_grad():
            self.encoder.weight.copy_(weight)
            self.encoder.bias.copy_(bias_e)
            self.decoder.weight.copy_(weight)
            self.decoder.bias.copy_(bias_d)
        for p in self.parameters():
            p.requires_grad_(False)

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, 3, H, W)`` images (H, W even) to ``(B, 4, H/2, W/2)`` latents."""
        return self.encoder(img)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode ``(B, 4, h, w)`` latents to ``(B, 3, 2h, 2w)`` images (pointwise-affine in z)."""
        return self.decoder(z)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Round trip ``decode(encode(img))``; exact on the decoder's range."""
        return self.decode(self.encode(img))

    def true_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the planted ``(W: (3, 4), c: (3,))`` with block-avg(decode(z)) = W z + c.

        See the class docstring for the derivation (plan §3.2 exact-linear case).
        """
        w = torch.zeros(3, 4)
        w[:, :3] = 0.5 * torch.eye(3)
        c = self.decoder.bias.detach().clone()
        return w, c

    # SD-VAE-compatible helpers so probes can swap --vae toy for --vae sd.
    def encode_moments(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic-'posterior' helper: (mu, sigma) = (encode(img), zeros)."""
        mu = self.encode(img)
        return mu, torch.zeros_like(mu)

    def decode_latents(self, z: torch.Tensor) -> torch.Tensor:
        """Alias of :meth:`decode` matching the SD-VAE helper name."""
        return self.decode(z)


class ToyConvAE(nn.Module):
    """Small nonlinear conv AE (3→16→C encoder, mirror decoder, SiLU), f=2.

    A download-free stand-in for pipeline tests (SPEC.md "pheq/vae.py"):
    trainable in seconds on synthetic data. Unlike :class:`ToyLinearAE` its
    decoder is NOT pointwise-affine, so it exercises the fitted/learned
    operator paths (plan §3.2–3.3) rather than the exact-linear identity.

    Initialization is deterministic given ``seed`` (global RNG state is
    forked, not mutated).
    """

    def __init__(self, channels: int = 4, hidden: int = 16, seed: int = 0) -> None:
        super().__init__()
        self.channels = channels
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = nn.Sequential(
                nn.Conv2d(3, hidden, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(hidden, channels, kernel_size=2, stride=2),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(channels, hidden, kernel_size=2, stride=2),
                nn.SiLU(),
                nn.Conv2d(hidden, 3, kernel_size=3, padding=1),
            )

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, 3, H, W)`` images (H, W even) to ``(B, C, H/2, W/2)`` latents."""
        return self.encoder(img)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode ``(B, C, h, w)`` latents to ``(B, 3, 2h, 2w)`` images."""
        return self.decoder(z)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Reconstruction ``decode(encode(img))``."""
        return self.decode(self.encode(img))

    # SD-VAE-compatible helpers so probes can swap --vae toy for --vae sd.
    def encode_moments(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic-'posterior' helper: (mu, sigma) = (encode(img), zeros)."""
        mu = self.encode(img)
        return mu, torch.zeros_like(mu)

    def decode_latents(self, z: torch.Tensor) -> torch.Tensor:
        """Alias of :meth:`decode` matching the SD-VAE helper name."""
        return self.decode(z)
