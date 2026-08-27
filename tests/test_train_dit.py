"""Tests for pheq.train_dit (SPEC2): offline tiny-shard training loop, resume,
mandatory stats.json refusal, sampling hook (un-normalize -> decode -> PNG),
FID hook with compute_fid monkeypatched, SIGTERM checkpoint save."""

import json
import os
import signal
import threading

import pytest
import torch

from pheq.train_dit import train
from pheq.vae import ToyConvAE

C, H = 4, 8  # latent shape (4, 8, 8) — ToyConvAE latents of 16x16 images


def _write_cache(root, n: int = 32, seed: int = 0, sigma: float = 0.1) -> None:
    """Synthetic latent cache: two shards + a consistent stats.json."""
    root.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(seed)
    mu = torch.randn(n, C, H, H, generator=gen) * 1.5 + 0.3
    half = n // 2
    for i, sl in enumerate((slice(0, half), slice(half, n))):
        torch.save(
            {
                "mu": mu[sl].to(torch.float16),
                "sigma": torch.full_like(mu[sl], sigma).to(torch.float16),
                "label": torch.zeros(mu[sl].shape[0], dtype=torch.int64),
            },
            root / f"shard_{i:04d}.pt",
        )
    (root / "stats.json").write_text(
        json.dumps(
            {
                "mean": mu.mean(dim=(0, 2, 3)).tolist(),
                "std": mu.var(dim=(0, 2, 3), unbiased=False).sqrt().tolist(),
                "latent_shape": [C, H, H],
                "shard_sizes": [half, n - half],
                "condition": "b1",
                "step": 0,
            }
        )
    )


def _toy_vae_ckpt(path):
    """SPEC2 run checkpoint wrapping a ToyConvAE (decodes (4,8,8) -> 16x16)."""
    torch.save(
        {
            "vae": ToyConvAE(seed=0).state_dict(),
            "operator": None,
            "operator_kind": "none",
            "condition": "b1",
            "step": 0,
            "wfit": {"W": torch.zeros(3, 4), "c": torch.zeros(3)},
            "config": {"vae": "toy"},
        },
        path,
    )
    return str(path)


def _monitor(out):
    return [json.loads(l) for l in (out / "monitor.jsonl").read_text().splitlines()]


def test_short_loop_no_nan(tmp_path):
    lat, out = tmp_path / "lat", tmp_path / "run"
    _write_cache(lat)
    ckpt = train(
        str(lat), str(out), arch="dit_tiny", steps=30, batch=8, lr=1e-3,
        device="cpu", seed=0, monitor_every=10,
    )
    assert ckpt.is_file()
    ck = torch.load(ckpt, weights_only=False)
    assert ck["step"] == 30 and ck["arch"] == "dit_tiny"
    assert set(ck) >= {"model", "ema", "opt", "step", "config", "gen_state"}
    lines = [r for r in _monitor(out) if r["level"] == "train"]
    assert [r["step"] for r in lines] == [10, 20, 30]
    for r in lines:
        assert "loss" in r and "loss_avg" in r
        assert torch.isfinite(torch.tensor(r["loss"]))


def test_resume_continues(tmp_path):
    lat, out = tmp_path / "lat", tmp_path / "run"
    _write_cache(lat)
    train(str(lat), str(out), arch="dit_tiny", steps=20, batch=8,
          device="cpu", seed=0, monitor_every=10)
    ckpt = train(str(lat), str(out), arch="dit_tiny", steps=40, batch=8,
                 device="cpu", seed=0, resume=True, monitor_every=10)
    ck = torch.load(ckpt, weights_only=False)
    assert ck["step"] == 40  # continued from 20, not restarted
    steps = [r["step"] for r in _monitor(out) if r["level"] == "train"]
    assert steps == [10, 20, 30, 40]  # loss continuity across the resume
    # Segment marker for consumers (hard-crash monitor.jsonl dedupe).
    resumes = [r["step"] for r in _monitor(out) if r["level"] == "resume"]
    assert resumes == [20]


def test_refuses_without_stats(tmp_path):
    """train_dit MUST refuse to run without stats.json (plan §3.4 gap-review)."""
    lat = tmp_path / "lat"
    _write_cache(lat)
    (lat / "stats.json").unlink()
    with pytest.raises(FileNotFoundError, match="stats.json"):
        train(str(lat), str(tmp_path / "run"), arch="dit_tiny", steps=2,
              batch=4, device="cpu")


def test_batch_larger_than_dataset(tmp_path):
    lat = tmp_path / "lat"
    _write_cache(lat, n=8)
    with pytest.raises(ValueError, match="batch"):
        train(str(lat), str(tmp_path / "run"), arch="dit_tiny", steps=2,
              batch=64, device="cpu")


def test_sampling_hook_pngs(tmp_path):
    from PIL import Image

    lat, out = tmp_path / "lat", tmp_path / "run"
    _write_cache(lat)
    vae_ckpt = _toy_vae_ckpt(tmp_path / "vae.pt")
    train(
        str(lat), str(out), arch="dit_tiny", steps=4, batch=4, device="cpu",
        seed=0, sample_every=2, vae_ckpt=vae_ckpt,
        timesteps=8, sample_steps=4, n_sample=4,
        monitor_every=2, save_every=2,
    )
    pngs = sorted((out / "samples").glob("step_*.png"))
    assert [p.name for p in pngs] == ["step_0000002.png", "step_0000004.png"]
    with Image.open(pngs[0]) as im:
        assert im.mode == "RGB"
        # 2x2 grid of 16x16 decodes, make_grid padding 2: 2*(16+2)+2 = 38.
        assert im.size == (38, 38)
    assert any(r["level"] == "sample" and r["step"] == 2 for r in _monitor(out))


def test_sampling_hook_skipped_without_vae(tmp_path):
    lat, out = tmp_path / "lat", tmp_path / "run"
    _write_cache(lat)
    train(str(lat), str(out), arch="dit_tiny", steps=2, batch=4, device="cpu",
          sample_every=2, timesteps=8, monitor_every=2, save_every=2)
    alerts = [r for r in _monitor(out) if r["level"] == "alert"]
    assert any("vae-ckpt" in r["msg"] for r in alerts)
    assert not (out / "samples").exists()


def test_fid_hook_monkeypatched(tmp_path, monkeypatch):
    lat, out = tmp_path / "lat", tmp_path / "run"
    _write_cache(lat)
    vae_ckpt = _toy_vae_ckpt(tmp_path / "vae.pt")
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    calls = {}

    def fake_compute_fid(dir_a, dir_b, mode="clean"):
        calls["args"] = (dir_a, dir_b)
        return 1.23

    monkeypatch.setattr("pheq.fid.compute_fid", fake_compute_fid)
    train(
        str(lat), str(out), arch="dit_tiny", steps=2, batch=4, device="cpu",
        seed=0, sample_every=0, fid_every=2, fid_ref_stats=str(ref_dir),
        n_fid=6, vae_ckpt=vae_ckpt, timesteps=8, sample_steps=2,
        monitor_every=2, save_every=2,
    )
    fid_lines = [r for r in _monitor(out) if r["level"] == "fid"]
    assert fid_lines and fid_lines[0]["fid"] == 1.23 and fid_lines[0]["n"] == 6
    gen_dir = out / "fid" / "step_0000002"
    assert len(list(gen_dir.glob("*.png"))) == 6  # decoded sample PNGs written
    assert calls["args"] == (str(gen_dir), str(ref_dir))


def test_class_conditional_loop(tmp_path):
    lat, out = tmp_path / "lat", tmp_path / "run"
    _write_cache(lat)
    ckpt = train(str(lat), str(out), arch="dit_tiny", steps=4, batch=4,
                 device="cpu", num_classes=2, monitor_every=2, save_every=2)
    assert torch.load(ckpt, weights_only=False)["step"] == 4
    # The all-zero label cache covers only class 0 of range(2): trainable,
    # but alert-logged (class 1's embedding stays init noise yet IS sampled
    # by the hooks — the without-`--class-from-dir` footgun).
    alerts = [r for r in _monitor(out) if r["level"] == "alert"]
    assert any("class" in r["msg"] for r in alerts)


def test_labels_exceeding_num_classes_refused(tmp_path):
    """Startup validation: a cached label >= num_classes would silently train
    the reserved null-class row (== num_classes) or IndexError mid-run
    (> num_classes) — refuse at train() startup instead."""
    lat = tmp_path / "lat"
    _write_cache(lat)
    shard_path = lat / "shard_0000.pt"
    shard = torch.load(shard_path, weights_only=False)
    shard["label"][0] = 2  # == num_classes: the reserved null row
    torch.save(shard, shard_path)
    with pytest.raises(ValueError, match="num_classes"):
        train(str(lat), str(tmp_path / "run"), arch="dit_tiny", steps=2,
              batch=4, device="cpu", num_classes=2)


def test_unconditional_ignores_labels(tmp_path):
    """num_classes=0 must not run the label scan (labels are ignored)."""
    lat = tmp_path / "lat"
    _write_cache(lat)
    shard_path = lat / "shard_0000.pt"
    shard = torch.load(shard_path, weights_only=False)
    shard["label"][:] = 99  # nonsense labels: irrelevant unconditionally
    torch.save(shard, shard_path)
    ckpt = train(str(lat), str(tmp_path / "run"), arch="dit_tiny", steps=2,
                 batch=4, device="cpu", num_classes=0, monitor_every=2,
                 save_every=2)
    assert torch.load(ckpt, weights_only=False)["step"] == 2


def test_sigterm_saves_and_exits_cleanly(tmp_path):
    """SLURM-preemption contract: SIGTERM -> checkpoint saved, clean return."""
    lat, out = tmp_path / "lat", tmp_path / "run"
    _write_cache(lat)
    # Safety net: if the timer fires after train() restored the previous
    # handler, a no-op handler must catch it instead of killing pytest.
    prev = signal.signal(signal.SIGTERM, lambda *_: None)
    timer = threading.Timer(0.4, lambda: os.kill(os.getpid(), signal.SIGTERM))
    try:
        timer.start()
        ckpt = train(
            str(lat), str(out), arch="dit_tiny", steps=2000, batch=4,
            device="cpu", seed=0, monitor_every=100, save_every=100_000,
        )
    finally:
        timer.cancel()
        signal.signal(signal.SIGTERM, prev)
    ck = torch.load(ckpt, weights_only=False)
    assert 0 < ck["step"] < 2000  # interrupted well before completion
    alerts = [r for r in _monitor(out) if r["level"] == "alert"]
    assert any("SIGTERM" in r["msg"] for r in alerts)
