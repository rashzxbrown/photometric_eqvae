"""Tests for pheq.cache_latents (SPEC2): shard round-trip, per-checkpoint
stats.json (mandatory, plan §3.4), fresh-eps resampling, missing-stats error."""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from pheq.cache_latents import LatentShardDataset, cache, load_stats, main
from pheq.data import ImageFolderDataset
from pheq.vae import ToyConvAE

SIZE = 16  # ToyConvAE f=2 -> latents (4, 8, 8)


def _write_images(root, n: int, seed: int = 0, subdirs=("a", "b")) -> None:
    """Deterministic random PNGs split across subdirs (for class_from_dir)."""
    gen = torch.Generator().manual_seed(seed)
    for i in range(n):
        d = root / subdirs[i % len(subdirs)]
        d.mkdir(exist_ok=True, parents=True)
        arr = (
            torch.rand(SIZE, SIZE, 3, generator=gen).mul(255).round().to(torch.uint8)
        )
        Image.fromarray(np.asarray(arr), mode="RGB").save(d / f"img_{i:03d}.png")


def _toy_run_ckpt(path, vae: ToyConvAE, condition: str = "b1", step: int = 123):
    """Minimal SPEC2 run checkpoint wrapping a ToyConvAE."""
    torch.save(
        {
            "vae": vae.state_dict(),
            "operator": None,
            "operator_kind": "none",
            "condition": condition,
            "step": step,
            "wfit": {"W": torch.zeros(3, 4), "c": torch.zeros(3)},
            "config": {"vae": "toy"},
        },
        path,
    )
    return path


@pytest.fixture()
def cached(tmp_path):
    """32 images cached through a ToyConvAE run checkpoint (shard_size=8)."""
    img_dir = tmp_path / "imgs"
    _write_images(img_dir, 32)
    vae = ToyConvAE(seed=0)
    ckpt = _toy_run_ckpt(tmp_path / "run.pt", vae)
    out = tmp_path / "latents"
    cache(
        str(ckpt), str(img_dir), str(out),
        batch=8, device="cpu", shard_size=8,
        size=SIZE, class_from_dir=True,
    )
    return img_dir, out, vae


def test_shard_format_and_stats(cached):
    img_dir, out, vae = cached
    shards = sorted(out.glob("shard_*.pt"))
    assert [p.name for p in shards] == [f"shard_{i:04d}.pt" for i in range(4)]
    for p in shards:
        shard = torch.load(p)
        assert shard["mu"].shape == (8, 4, 8, 8)
        assert shard["mu"].dtype == torch.float16  # f16 storage (SPEC2)
        assert shard["sigma"].shape == (8, 4, 8, 8)
        assert shard["sigma"].dtype == torch.float16
        assert shard["label"].shape == (8,)
        assert shard["label"].dtype == torch.int64

    stats = load_stats(str(out))
    assert stats["n_images"] == 32
    assert stats["latent_shape"] == [4, 8, 8]
    assert stats["shard_sizes"] == [8, 8, 8, 8]
    assert stats["condition"] == "b1" and stats["step"] == 123  # ckpt provenance
    assert len(stats["manifest_hash"]) == 64  # sha256 hex

    # Stats match direct f32 batch statistics of mu over the whole set.
    ds = ImageFolderDataset(str(img_dir), size=SIZE, class_from_dir=True)
    with torch.no_grad():
        mu, _ = vae.encode_moments(torch.stack([ds[i][0] for i in range(len(ds))]))
    ref_mean = mu.mean(dim=(0, 2, 3))
    ref_std = mu.var(dim=(0, 2, 3), unbiased=False).sqrt()  # population std
    assert torch.allclose(torch.tensor(stats["mean"]), ref_mean, atol=1e-5)
    assert torch.allclose(torch.tensor(stats["std"]), ref_std, atol=1e-5)

    # Labels reflect the parent-dir classes (a=0, b=1), 16 each.
    labels = torch.cat([torch.load(p)["label"] for p in shards])
    assert sorted(labels.tolist()) == [0] * 16 + [1] * 16


def test_dataset_roundtrip_normalized(cached):
    _img_dir, out, _vae = cached
    ds = LatentShardDataset(str(out))
    assert len(ds) == 32
    z, label = ds[0]
    assert z.shape == (4, 8, 8) and z.dtype == torch.float32
    assert label in (0, 1)
    assert ds[-1][0].shape == (4, 8, 8)  # negative indexing

    # ToyConvAE posteriors have sigma=0, so z = (mu - mean)/std exactly:
    # the normalized set must be ~zero-mean/unit-std per channel (f16
    # quantization of mu is the only error source).
    zs = torch.stack([ds[i][0] for i in range(len(ds))])
    assert torch.allclose(zs.mean(dim=(0, 2, 3)), torch.zeros(4), atol=1e-3)
    assert torch.allclose(
        zs.var(dim=(0, 2, 3), unbiased=False).sqrt(), torch.ones(4), atol=1e-2
    )


def _write_manual_cache(root, sigma_val: float = 1.0, n: int = 4) -> None:
    """Hand-written single-shard cache (identity normalization)."""
    root.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mu": torch.zeros(n, 4, 4, 4, dtype=torch.float16),
            "sigma": torch.full((n, 4, 4, 4), sigma_val, dtype=torch.float16),
            "label": torch.zeros(n, dtype=torch.int64),
        },
        root / "shard_0000.pt",
    )
    (root / "stats.json").write_text(
        json.dumps(
            {
                "mean": [0.0] * 4,
                "std": [1.0] * 4,
                "latent_shape": [4, 4, 4],
                "shard_sizes": [n],
            }
        )
    )


def test_fresh_eps_per_access(tmp_path):
    """Plan §3.4 caching convention: moments cached, noise RE-SAMPLED."""
    _write_manual_cache(tmp_path / "lat", sigma_val=1.0)
    ds = LatentShardDataset(str(tmp_path / "lat"))
    torch.manual_seed(0)
    z1, _ = ds[0]
    z2, _ = ds[0]
    assert not torch.allclose(z1, z2)  # same index, different noise
    # ...and deterministic under a reset global seed.
    torch.manual_seed(0)
    z1b, _ = ds[0]
    assert torch.allclose(z1, z1b)


def test_shard_sizes_fallback_peek(tmp_path):
    """stats.json without shard_sizes: dataset peeks shard lengths once."""
    _write_manual_cache(tmp_path / "lat", n=5)
    stats = json.loads((tmp_path / "lat" / "stats.json").read_text())
    del stats["shard_sizes"]
    (tmp_path / "lat" / "stats.json").write_text(json.dumps(stats))
    ds = LatentShardDataset(str(tmp_path / "lat"))
    assert len(ds) == 5


def test_missing_stats_raises(tmp_path):
    """stats.json is MANDATORY (plan §3.4): no silent unnormalized fallback."""
    lat = tmp_path / "lat"
    _write_manual_cache(lat)
    (lat / "stats.json").unlink()
    with pytest.raises(FileNotFoundError, match="stats.json"):
        LatentShardDataset(str(lat))
    with pytest.raises(FileNotFoundError, match="stats.json"):
        load_stats(str(lat))


def test_cache_from_bare_module_and_restale(tmp_path):
    """Bare nn.Module input -> null provenance; re-cache removes stale shards."""
    img_dir = tmp_path / "imgs"
    _write_images(img_dir, 8, subdirs=("only",))
    out = tmp_path / "lat"
    # Plant a stale shard that a re-cache must remove.
    _write_manual_cache(out)
    cache(ToyConvAE(seed=1), str(img_dir), str(out), batch=4, size=SIZE)
    stats = load_stats(str(out))
    assert stats["condition"] is None and stats["step"] is None
    assert stats["shard_sizes"] == [8]  # single shard (default shard_size)
    assert len(list(out.glob("shard_*.pt"))) == 1  # stale shard_0000 replaced
    ds = LatentShardDataset(str(out))
    assert len(ds) == 8


def test_cli_smoke(tmp_path, capsys):
    img_dir = tmp_path / "imgs"
    _write_images(img_dir, 8)
    ckpt = _toy_run_ckpt(tmp_path / "run.pt", ToyConvAE(seed=0))
    out = tmp_path / "lat"
    main(
        [
            "--ckpt", str(ckpt), "--images", str(img_dir), "--out", str(out),
            "--batch", "4", "--shard-size", "8", "--size", str(SIZE),
        ]
    )
    assert (out / "stats.json").is_file()
    assert "cached 8 images" in capsys.readouterr().out
