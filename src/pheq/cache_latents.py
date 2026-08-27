"""Latent shard cache + per-checkpoint normalization (SPEC2 cache_latents.py).

Encodes an image folder with a (fine-tuned) VAE checkpoint into ``.pt`` shards
of posterior MOMENTS, then writes ``stats.json`` with the per-channel mean/std
of mu over the whole set. Downstream DiT training normalizes with EXACTLY these
per-checkpoint statistics (plan §3.4: "the latent scale factor is recomputed
per fine-tuned checkpoint ... never reusing SD-VAE's 0.18215 across
checkpoints" — without this, gFID differences between conditions can be
dominated by mismatched latent scaling, a silent confound). stats.json is
therefore MANDATORY: :class:`LatentShardDataset` refuses to construct without
it, and pheq.train_dit refuses to run.

Caching convention (plan §3.4): shards store MOMENTS ``(mu, sigma)``; noise is
re-sampled fresh on every dataset access — never cache realized samples.

Storage is float16, all compute is float32 (SPEC2 care point): moments are
encoded in f32, stored f16, and read back as f32; the normalization statistics
are computed from the f32 moments (f64 accumulators for the streaming sums),
never from the quantized f16 copies.

CLI::

    uv run python -m pheq.cache_latents --ckpt CKPT --images DIR --out DIR
        [--batch 32] [--device cpu] [--shard-size 2048] [--size 256]
        [--class-from-dir]
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import Dataset

from pheq.data import ImageFolderDataset

# Canonical run-checkpoint -> eval-mode VAE resolver (SPEC2 run-checkpoint
# format). Private to pheq.fid but stable within this repo; reusing it keeps
# checkpoint-architecture resolution ('toy' vs 'sd') identical across
# fid.rfid, cache_latents, and train_dit's sampling hook.
from pheq.fid import _resolve_vae

__all__ = ["cache", "LatentShardDataset", "resolve_vae", "main"]

STATS_NAME = "stats.json"
SHARD_PATTERN = "shard_*.pt"

#: Floor applied to the per-channel std at normalization time so a dead
#: (zero-variance) latent channel cannot produce inf/NaN normalized latents.
STD_EPS = 1e-8


def resolve_vae(
    ckpt_or_vae: Any, device: str = "cpu"
) -> tuple[nn.Module, dict | None]:
    """Resolve a VAE module or run-checkpoint into ``(vae, ckpt_dict | None)``.

    Accepts an ``nn.Module`` (returned as-is, moved to ``device``, eval mode),
    a path to a SPEC2 run checkpoint, or an already-loaded run-checkpoint
    dict. The checkpoint dict (when there is one) is returned alongside the
    module so callers can record its ``condition``/``step`` provenance in
    stats.json.
    """
    ckpt: dict | None = None
    if isinstance(ckpt_or_vae, (str, Path)):
        ckpt = torch.load(str(ckpt_or_vae), map_location="cpu", weights_only=False)
    elif isinstance(ckpt_or_vae, dict):
        ckpt = ckpt_or_vae
    vae = _resolve_vae(ckpt if ckpt is not None else ckpt_or_vae, device)
    return vae, ckpt


def _manifest_hash(dataset: ImageFolderDataset) -> str:
    """sha256 over the sorted relative image paths (the dataset manifest).

    Ties stats.json to the exact image set it was computed on (SPEC2:
    "image_dir manifest hash"); content hashing would be slower and the
    reference sets are static during the sprint.
    """
    rel = "\n".join(str(p.relative_to(dataset.root).as_posix()) for p in dataset.paths)
    return hashlib.sha256(rel.encode("utf-8")).hexdigest()


def cache(
    ckpt_or_vae: Any,
    image_dir: str,
    out_dir: str,
    batch: int = 32,
    device: str = "cpu",
    shard_size: int = 2048,
    *,
    size: int = 256,
    class_from_dir: bool = False,
) -> Path:
    """Encode ALL images under ``image_dir`` into latent-moment shards.

    Writes ``out_dir/shard_{i:04d}.pt``, each a dict
    ``{"mu": (S, C, h, w) float16, "sigma": (S, C, h, w) float16,
    "label": (S,) int64}`` (S = ``shard_size`` except the last shard), then
    ``out_dir/stats.json`` with:

    - ``mean``/``std``: per-channel statistics of mu over the WHOLE set,
      computed from the f32 moments in one streaming pass (f64 sum/sum-of-
      squares accumulators; population std). These are the MANDATORY
      per-checkpoint normalization constants of plan §3.4.
    - ``condition``/``step``: provenance of the source run checkpoint (null
      when a bare module was passed).
    - ``manifest_hash``: sha256 of the sorted relative image paths.
    - ``shard_sizes``: per-shard sample counts (lets the dataset index shards
      lazily without opening them).
    - ``n_images``, ``latent_shape``, ``image_dir``, ``image_size``.

    Args:
        ckpt_or_vae: autoencoder module, run-checkpoint path, or loaded
            run-checkpoint dict (SPEC2 run-checkpoint format).
        image_dir: image folder (recursed by :class:`ImageFolderDataset`,
            deterministic sorted order).
        out_dir: output directory; pre-existing ``shard_*.pt``/``stats.json``
            are deleted first (stale shards would silently mix checkpoints).
        batch: encode chunk size.
        device: torch device string for encoding.
        shard_size: samples per shard (default 2048 per SPEC2).
        size: ImageFolderDataset resolution (keyword beyond the SPEC2 arg
            list so toy tests run tiny; default 256, the sprint resolution).
        class_from_dir: forward to ImageFolderDataset (ImageNet-100 layout).

    Returns:
        Path to the written ``stats.json``.
    """
    if shard_size < 1:
        raise ValueError(f"shard_size must be >= 1, got {shard_size}")
    dataset = ImageFolderDataset(image_dir, size=size, class_from_dir=class_from_dir)
    vae, ckpt = resolve_vae(ckpt_or_vae, device)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in list(out.glob(SHARD_PATTERN)) + [out / STATS_NAME]:
        if stale.exists():
            stale.unlink()

    # Streaming per-channel stats of mu (f32 values, f64 accumulators).
    ch_sum: torch.Tensor | None = None
    ch_sumsq: torch.Tensor | None = None
    n_sites = 0

    buf_mu: list[torch.Tensor] = []
    buf_sigma: list[torch.Tensor] = []
    buf_label: list[torch.Tensor] = []
    buffered = 0
    shard_idx = 0
    shard_sizes: list[int] = []
    latent_shape: tuple[int, ...] | None = None

    def _flush(n: int) -> None:
        """Write the first ``n`` buffered samples as one shard (f16 storage)."""
        nonlocal buf_mu, buf_sigma, buf_label, buffered, shard_idx
        mu = torch.cat(buf_mu)
        sigma = torch.cat(buf_sigma)
        label = torch.cat(buf_label)
        torch.save(
            {
                "mu": mu[:n].to(torch.float16),
                "sigma": sigma[:n].to(torch.float16),
                "label": label[:n].to(torch.int64),
            },
            out / f"shard_{shard_idx:04d}.pt",
        )
        shard_sizes.append(int(n))
        buf_mu = [mu[n:]] if mu.shape[0] > n else []
        buf_sigma = [sigma[n:]] if sigma.shape[0] > n else []
        buf_label = [label[n:]] if label.shape[0] > n else []
        buffered -= n
        shard_idx += 1

    with torch.no_grad():
        for start in range(0, len(dataset), batch):
            idx = range(start, min(start + batch, len(dataset)))
            pairs = [dataset[i] for i in idx]
            x = torch.stack([p[0] for p in pairs]).to(device)
            labels = torch.tensor([p[1] for p in pairs], dtype=torch.int64)
            mu, sigma = vae.encode_moments(x)
            mu = mu.detach().to("cpu", torch.float32)
            sigma = sigma.detach().to("cpu", torch.float32)
            if latent_shape is None:
                latent_shape = tuple(mu.shape[1:])
            # f64 streaming accumulators over the f32 mu (SPEC2: stats are
            # computed in f32 streaming — i.e. from the full-precision
            # moments, never the quantized f16 shard copies).
            m64 = mu.to(torch.float64)
            s = m64.sum(dim=(0, 2, 3))
            sq = (m64 * m64).sum(dim=(0, 2, 3))
            ch_sum = s if ch_sum is None else ch_sum + s
            ch_sumsq = sq if ch_sumsq is None else ch_sumsq + sq
            n_sites += mu.shape[0] * mu.shape[2] * mu.shape[3]

            buf_mu.append(mu)
            buf_sigma.append(sigma)
            buf_label.append(labels)
            buffered += mu.shape[0]
            while buffered >= shard_size:
                _flush(shard_size)
    if buffered > 0:
        _flush(buffered)

    assert ch_sum is not None and ch_sumsq is not None and latent_shape is not None
    mean = ch_sum / n_sites
    var = (ch_sumsq / n_sites - mean * mean).clamp_min(0.0)
    std = var.sqrt()

    stats = {
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "n_images": len(dataset),
        "latent_shape": [int(v) for v in latent_shape],
        "shard_sizes": shard_sizes,
        "condition": (ckpt or {}).get("condition"),
        "step": (ckpt or {}).get("step"),
        "manifest_hash": _manifest_hash(dataset),
        "image_dir": str(image_dir),
        "image_size": int(size),
    }
    stats_path = out / STATS_NAME
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    return stats_path


def load_stats(latent_dir: str) -> dict:
    """Load ``stats.json`` from a latent cache dir; raise if missing.

    Per-checkpoint normalization is MANDATORY (plan §3.4 / SPEC2 gap-review):
    a missing stats.json means the cache was never finalized, and training on
    unnormalized (or wrongly normalized) latents is the silent-scaling
    confound this file exists to prevent — so this is a hard error, never a
    fallback.
    """
    path = Path(latent_dir) / STATS_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {STATS_NAME} in {str(latent_dir)!r}: per-checkpoint latent "
            "normalization is mandatory (plan §3.4). Re-run pheq.cache_latents "
            "to (re)build the cache."
        )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class LatentShardDataset(Dataset):
    """Lazy dataset over cached latent-moment shards (SPEC2 cache_latents.py).

    ``__getitem__`` returns ``(z_normalized, label)`` with

        ``z = (mu + sigma * eps - mean) / std``,   ``eps ~ N(0, I)``,

    where ``eps`` is drawn FRESH on every access (plan §3.4 caching
    convention: caches store moments; noise is re-sampled) and mean/std are
    the per-checkpoint statistics from ``stats.json`` (missing stats.json ->
    :class:`FileNotFoundError` at construction — normalization is mandatory).
    Moments are stored f16 and computed in f32 (SPEC2 care point); ``z`` is
    float32. ``eps`` uses torch's global RNG (per-worker seeded by the
    DataLoader) — seed ``torch.manual_seed`` for deterministic single-process
    access.

    Shards are loaded lazily with an LRU cache (``max_cached_shards``,
    default 64 — at the standard shard_size 2048 and SD-VAE moments that is
    ~2 GB, enough to hold this project's whole 41K-image cache in RAM. The
    old default of 2 caused catastrophic thrash under shuffled sampling:
    ~240 disk shard-loads PER BATCH across 21 shards, capping DiT training
    at ~0.55 it/s on an idle H200 — data-bound, not GPU-bound);
    shard lengths come from stats.json's ``shard_sizes`` when present, else
    from a one-time peek at each shard (supports hand-written test shards).

    Attributes:
        mean / std: ``(C, 1, 1)`` float32 normalization constants (std
            floored at :data:`STD_EPS`).
        latent_shape: ``(C, h, w)`` of one sample.
        stats: the parsed stats.json dict.
    """

    def __init__(self, latent_dir: str, max_cached_shards: int = 64) -> None:
        self.root = Path(latent_dir)
        self.stats = load_stats(latent_dir)
        self.shard_paths: list[Path] = sorted(self.root.glob(SHARD_PATTERN), key=str)
        if not self.shard_paths:
            raise FileNotFoundError(
                f"no {SHARD_PATTERN} shards found in {str(latent_dir)!r}"
            )
        sizes = self.stats.get("shard_sizes")
        if sizes is not None and len(sizes) == len(self.shard_paths):
            self._sizes = [int(s) for s in sizes]
        else:  # hand-written cache without shard_sizes: peek once per shard
            self._sizes = [
                int(torch.load(p, map_location="cpu")["label"].shape[0])
                for p in self.shard_paths
            ]
        self._offsets: list[int] = []
        total = 0
        for s in self._sizes:
            total += s
            self._offsets.append(total)
        self._total = total

        mean = torch.tensor(self.stats["mean"], dtype=torch.float32)
        std = torch.tensor(self.stats["std"], dtype=torch.float32)
        self.mean = mean.view(-1, 1, 1)
        self.std = std.clamp_min(STD_EPS).view(-1, 1, 1)
        self.latent_shape: tuple[int, ...] = tuple(
            int(v) for v in self.stats.get("latent_shape", ())
        )
        self._max_cached = max(1, int(max_cached_shards))
        self._cache: OrderedDict[int, dict] = OrderedDict()

    def __len__(self) -> int:
        return self._total

    def _get_shard(self, i: int) -> dict:
        if i in self._cache:
            self._cache.move_to_end(i)
            return self._cache[i]
        shard = torch.load(self.shard_paths[i], map_location="cpu")
        self._cache[i] = shard
        while len(self._cache) > self._max_cached:
            self._cache.popitem(last=False)
        return shard

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if not -self._total <= index < self._total:
            raise IndexError(f"index {index} out of range for {self._total} samples")
        if index < 0:
            index += self._total
        i = bisect.bisect_right(self._offsets, index)
        j = index - (self._offsets[i - 1] if i > 0 else 0)
        shard = self._get_shard(i)
        mu = shard["mu"][j].to(torch.float32)  # f16 storage, f32 compute
        sigma = shard["sigma"][j].to(torch.float32)
        eps = torch.randn_like(mu)  # fresh noise per access (plan §3.4)
        z = (mu + sigma * eps - self.mean) / self.std
        return z, int(shard["label"][j])


def main(argv: list[str] | None = None) -> None:
    """CLI: cache latent shards + stats.json for a run checkpoint."""
    parser = argparse.ArgumentParser(
        description="Cache VAE latent moments as f16 shards + per-checkpoint "
        "normalization stats (plan §3.4)."
    )
    parser.add_argument("--ckpt", type=str, required=True,
                        help="run checkpoint path (SPEC2 run-checkpoint format)")
    parser.add_argument("--images", type=str, required=True,
                        help="image directory (recursed)")
    parser.add_argument("--out", type=str, required=True,
                        help="output directory for shards + stats.json")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--size", type=int, default=256,
                        help="ImageFolderDataset resolution (default 256)")
    parser.add_argument("--class-from-dir", action="store_true",
                        help="labels from immediate parent directory names")
    args = parser.parse_args(argv)

    stats_path = cache(
        args.ckpt,
        args.images,
        args.out,
        batch=args.batch,
        device=args.device,
        shard_size=args.shard_size,
        size=args.size,
        class_from_dir=args.class_from_dir,
    )
    with open(stats_path, encoding="utf-8") as fh:
        stats = json.load(fh)
    print(
        f"cached {stats['n_images']} images -> {len(stats['shard_sizes'])} shard(s) "
        f"in {args.out} (latent {tuple(stats['latent_shape'])})"
    )
    print(f"  mean = {[round(v, 4) for v in stats['mean']]}")
    print(f"  std  = {[round(v, 4) for v in stats['std']]}")
    print(f"  condition = {stats['condition']}  step = {stats['step']}")
    print(f"saved -> {stats_path}")


if __name__ == "__main__":
    main()
