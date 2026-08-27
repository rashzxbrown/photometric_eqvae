"""Minimal faithful DiT for latent diffusion (SPEC2 dit.py).

DiT (Peebles & Xie, "Scalable Diffusion Models with Transformers") sized for
(4, 32, 32) latents — the Tier-1 proxy generative model (plan §5, DiT-S
proxy-diffusion FID on every checkpoint; plan-3month M2/M3).

Faithful to the reference implementation: patchify via strided conv, fixed 2D
sin-cos positional embedding, adaLN-Zero transformer blocks (SiLU → Linear
modulation projections, ZERO-initialized, producing shift/scale/gate for both
the attention and MLP branches), sinusoidal timestep embedding → MLP, learned
null-class embedding for label dropout, and a zero-initialized final layer
(adaLN + linear). Zero-init of the final linear makes the network output
EXACTLY zero at initialization (adaLN-Zero, tested).

Sprint simplifications (documented per SPEC2):
- eps-prediction ONLY: forward returns the (B, C, h, w) noise estimate; no
  learned-sigma head (the reference DiT's default). The sampler in
  pheq.diffusion uses the fixed DDPM posterior variance instead.
- cfg-free protocol (plan §5 Tier 2, EQ-VAE anchor): class dropout still
  trains the null class, but no guidance machinery is provided.
- ``num_classes = 0`` → unconditional: the label input is ignored entirely and
  no label embedder is created.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def timestep_embedding(
    t: torch.Tensor, dim: int, max_period: float = 10_000.0
) -> torch.Tensor:
    """Sinusoidal timestep embedding (reference DiT convention).

    Args:
        t: (B,) timesteps (int or float tensor).
        dim: embedding dimension (must be even).
        max_period: minimum frequency period.

    Returns:
        (B, dim) tensor ``[cos(t·ω), sin(t·ω)]`` with log-spaced ω.
    """
    if dim % 2 != 0:
        raise ValueError(f"timestep embedding dim must be even, got {dim}")
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t.to(torch.float32)[:, None] * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def _build_2d_sincos_pos_embed(dim: int, grid_size: int) -> torch.Tensor:
    """Fixed (non-learned) 2D sin-cos positional embedding (reference DiT).

    Each token at grid position (i, j) receives the concatenation of a 1D
    sin-cos embedding of j (dim/2) and of i (dim/2), each split [sin, cos].

    Args:
        dim: hidden size (must be divisible by 4).
        grid_size: tokens per side (input_size // patch_size).

    Returns:
        (1, grid_size**2, dim) float32 tensor.
    """
    if dim % 4 != 0:
        raise ValueError(f"pos-embed dim must be divisible by 4, got {dim}")
    quarter = dim // 4
    omega = 1.0 / (
        10_000.0 ** (torch.arange(quarter, dtype=torch.float64) / quarter)
    )
    pos = torch.arange(grid_size, dtype=torch.float64)
    grid_i, grid_j = torch.meshgrid(pos, pos, indexing="ij")
    out = []
    for g in (grid_j, grid_i):  # width-axis first, matching the reference
        ang = g.reshape(-1)[:, None] * omega[None, :]  # (T, dim/4)
        out.append(torch.sin(ang))
        out.append(torch.cos(ang))
    return torch.cat(out, dim=-1).to(torch.float32)[None]


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN modulation ``x * (1 + scale) + shift`` (per-token broadcast)."""
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class TimestepEmbedder(nn.Module):
    """Sinusoidal frequency embedding → 2-layer SiLU MLP (reference DiT)."""

    def __init__(self, hidden_size: int, freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(t, self.freq_dim))


class LabelEmbedder(nn.Module):
    """Class-label embedding with a LEARNED null class for dropout (SPEC2).

    The embedding table has ``num_classes + 1`` rows; row ``num_classes`` is
    the null (unconditional) class. During training each label is replaced by
    the null class with probability ``dropout_prob`` (classifier-free-guidance
    training recipe; sampling here stays cfg-free per SPEC2).
    """

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)

    def forward(self, y: torch.Tensor, train: bool) -> torch.Tensor:
        if train and self.dropout_prob > 0:
            drop = torch.rand(y.shape[0], device=y.device) < self.dropout_prob
            y = torch.where(drop, torch.full_like(y, self.num_classes), y)
        return self.embedding_table(y)


class Attention(nn.Module):
    """Standard multi-head self-attention (qkv + output projection)."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, H, T, hd)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(b, t, d)
        return self.proj(x)


class Mlp(nn.Module):
    """Transformer MLP: Linear → GELU(tanh) → Linear (reference DiT)."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DiTBlock(nn.Module):
    """adaLN-Zero transformer block (Peebles & Xie §3.2).

    The conditioning vector c produces 6 modulation signals
    (shift/scale/gate × attention/MLP) through a SiLU → Linear projection that
    is ZERO-initialized, so every residual branch is exactly zero at init and
    each block is the identity function.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa[:, None, :] * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp[:, None, :] * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """Final adaLN + linear projection, ZERO-initialized (adaLN-Zero)."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class DiT(nn.Module):
    """Diffusion Transformer for latents, eps-prediction only (SPEC2 dit.py).

    Args:
        input_size: latent spatial side h (= w); (C, h, w) latents.
        patch_size: patchify factor p (p = 2 → (h/p)² tokens).
        in_channels: latent channels C (out channels equal — eps only).
        hidden_size: transformer width.
        depth: number of DiTBlocks.
        num_heads: attention heads.
        mlp_ratio: MLP expansion factor.
        num_classes: label classes; 0 → unconditional (label input ignored,
            no label embedder created).
        class_dropout: probability of replacing a label with the learned null
            class during training (ignored when num_classes = 0).
    """

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_classes: int = 0,
        class_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_size % patch_size != 0:
            raise ValueError(
                f"input_size {input_size} not divisible by patch_size {patch_size}"
            )
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels  # eps only, no learned sigma (SPEC2)
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.patch_embed = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        grid_size = input_size // patch_size
        self.num_patches = grid_size * grid_size
        self.register_buffer(
            "pos_embed", _build_2d_sincos_pos_embed(hidden_size, grid_size)
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder: LabelEmbedder | None = (
            LabelEmbedder(num_classes, hidden_size, class_dropout)
            if num_classes > 0
            else None
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self._init_weights()

    def _init_weights(self) -> None:
        """Reference DiT initialization (adaLN-Zero).

        Xavier-uniform linears; patch-embed conv treated as a linear;
        normal(0.02) timestep MLP and label table; ZERO modulation projections
        in every block; ZERO final adaLN projection and final linear — the
        network output is exactly zero at init.
        """

        def basic(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(basic)
        w = self.patch_embed.weight
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, p·p·C) token predictions → (B, C, h, w) latent grid."""
        b = x.shape[0]
        p = self.patch_size
        g = self.input_size // p
        c = self.out_channels
        x = x.reshape(b, g, g, p, p, c)
        x = torch.einsum("bhwpqc->bchpwq", x)
        return x.reshape(b, c, g * p, g * p)

    def forward(
        self, z_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict eps for noised latents (SPEC2: eps only, no learned sigma).

        Args:
            z_t: (B, C, input_size, input_size) noised latents.
            t: (B,) diffusion timesteps.
            y: (B,) integer labels; REQUIRED when num_classes > 0, ignored
               (may be None) when num_classes = 0.

        Returns:
            (B, C, input_size, input_size) eps prediction.
        """
        if z_t.shape[-2:] != (self.input_size, self.input_size):
            raise ValueError(
                f"expected {self.input_size}x{self.input_size} latents, "
                f"got {tuple(z_t.shape[-2:])}"
            )
        x = self.patch_embed(z_t).flatten(2).transpose(1, 2) + self.pos_embed
        c = self.t_embedder(t)
        if self.y_embedder is not None:
            if y is None:
                raise ValueError(
                    f"labels y required for conditional DiT (num_classes="
                    f"{self.num_classes}); pass num_classes=0 for unconditional"
                )
            c = c + self.y_embedder(y, self.training)
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        return self.unpatchify(x)


def dit_s(
    num_classes: int = 0,
    input_size: int = 32,
    in_channels: int = 4,
    class_dropout: float = 0.1,
) -> DiT:
    """DiT-S/2: patch 2, hidden 384, depth 12, heads 6 (~33M params, SPEC2)."""
    return DiT(
        input_size=input_size,
        patch_size=2,
        in_channels=in_channels,
        hidden_size=384,
        depth=12,
        num_heads=6,
        num_classes=num_classes,
        class_dropout=class_dropout,
    )


def dit_b(
    num_classes: int = 0,
    input_size: int = 32,
    in_channels: int = 4,
    class_dropout: float = 0.1,
) -> DiT:
    """DiT-B/2: patch 2, hidden 768, depth 12, heads 12 (~130M params) — the
    convergence-pair scale of docs/plan-3month.md (the 'hero figure')."""
    return DiT(
        input_size=input_size, patch_size=2, in_channels=in_channels,
        hidden_size=768, depth=12, num_heads=12,
        num_classes=num_classes, class_dropout=class_dropout,
    )


def dit_tiny(
    num_classes: int = 0,
    input_size: int = 32,
    in_channels: int = 4,
    class_dropout: float = 0.1,
) -> DiT:
    """Test-scale DiT: patch 2, hidden 128, depth 6, heads 4 (SPEC2)."""
    return DiT(
        input_size=input_size,
        patch_size=2,
        in_channels=in_channels,
        hidden_size=128,
        depth=6,
        num_heads=4,
        num_classes=num_classes,
        class_dropout=class_dropout,
    )
