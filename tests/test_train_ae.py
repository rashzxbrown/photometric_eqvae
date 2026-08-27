"""Tests for pheq.train_ae (SPEC2 "train_ae.py" — offline, ToyConvAE, tiny steps).

Covers the SPEC2 test list: the loop runs without NaN for
{b1, p1_analytic, p2_lie, b2lite, c1proxy(on-the-fly stats)}; checkpoint save
+ resume continuity; monitor.jsonl keys and step-0 snapshot; the SIGTERM
handler saves and exits 0 (signal raised in-process); and the
@pytest.mark.slow known-good-anchor test — the EE quick probe decreases over
200 steps for p1_analytic on a color-shift-friendly toy setup.

Everything runs on CPU with lambda_lpips = 0 (no LPIPS/VGG construction,
fully offline) and deterministic seeds.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import torch
from PIL import Image

from pheq.conditions import get_config
from pheq.probes._common import synthetic_images
from pheq.spectral import radial_power_spectrum, save_spectrum_stats
from pheq.train_ae import _drift_alert, train
from pheq.vae import ToyConvAE

REPO = Path(__file__).resolve().parents[1]

#: Monitor-line keys required by SPEC2 (step, recon L1, LPIPS, KL, per-channel
#: mu std, effective rank, spectral slope, clip fraction + EE quick probe).
MONITOR_KEYS = {
    "level", "step", "l1", "lpips", "kl", "mu_std",
    "eff_rank", "spectral_slope", "clip_fraction", "ee_quick",
}

#: The SPEC2 run-checkpoint consumer contract (eval_battery / cache_latents /
#: train_dit read exactly these).
CKPT_CONTRACT = {"vae", "operator", "operator_kind", "condition", "step", "wfit", "config"}


def _write_image_dir(root: Path, n: int, size: int, seed: int = 0) -> Path:
    """Save n deterministic synthetic RGB images as PNGs under root."""
    root.mkdir(parents=True, exist_ok=True)
    imgs = synthetic_images(n, size, seed=seed)
    for i in range(n):
        arr = (imgs[i] * 255.0).round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
        Image.fromarray(arr).save(root / f"img_{i:03d}.png")
    return root


@pytest.fixture(scope="session")
def data16(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """14 images at 16x16 — the smallest size with a valid spectral-slope fit."""
    return _write_image_dir(tmp_path_factory.mktemp("imgs16"), n=14, size=16)


@pytest.fixture(scope="session")
def data32(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """24 images at 32x32 for the CLI smoke and the slow anchor test."""
    return _write_image_dir(tmp_path_factory.mktemp("imgs32"), n=24, size=32)


def _tiny_config(name: str, **extra):
    """Registry config shrunk for offline CPU tests (LPIPS off, tiny shapes)."""
    overrides = dict(
        steps=30, image_size=16, batch_size=4, lambda_lpips=0.0, seed=0
    )
    overrides.update(extra)
    return get_config(name, **overrides)


def _monitor_lines(out_dir: Path) -> list[dict]:
    path = out_dir / "monitor.jsonl"
    assert path.exists(), "monitor.jsonl was not written"
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _spectral_stats_json(path: Path) -> str:
    """On-the-fly spectrum stats (SPEC2 c1proxy test convention)."""
    gen = torch.Generator().manual_seed(0)
    z = torch.randn(8, 4, 8, 8, generator=gen)
    freqs, power = radial_power_spectrum(z)
    save_spectrum_stats(path, freqs, power)
    return str(path)


# ---------------------------------------------------------------------------
# Loop runs without NaN for the SPEC2 condition set
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "condition", ["b1", "p1_analytic", "p2_lie", "b2lite", "c1proxy"]
)
def test_loop_runs_30_steps(condition: str, data16: Path, tmp_path: Path) -> None:
    extra = {}
    if condition == "c1proxy":
        extra["spectral_stats"] = _spectral_stats_json(tmp_path / "stats.json")
    cfg = _tiny_config(condition, **extra)
    out = tmp_path / "run"

    ckpt_path = train(cfg, str(data16), str(out), "cpu", vae="toy", workers=0)

    assert ckpt_path == out / "ckpt_latest.pt"
    assert ckpt_path.exists()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Run-checkpoint contract (SPEC2 design decisions).
    assert CKPT_CONTRACT <= set(ck)
    assert ck["condition"] == condition
    assert ck["operator_kind"] == cfg.operator_kind
    assert ck["step"] == 30
    assert ck["wfit"]["W"].shape == (3, 4)
    assert ck["wfit"]["c"].shape == (3,)
    assert ck["config"]["condition"] == condition
    if cfg.operator_kind in ("none", "analytic"):
        assert ck["operator"] is None
    else:
        assert ck["operator"] is not None
    # "Runs 30 steps without NaN" must hold for the TRAINED weights, not just
    # the step-0 monitor snapshot: every checkpoint tensor is finite and a
    # forward pass through the reloaded model stays finite.
    assert all(torch.isfinite(v).all() for v in ck["vae"].values()), (
        "non-finite VAE weights after 30 steps"
    )
    if ck["operator"] is not None:
        assert all(torch.isfinite(v).all() for v in ck["operator"].values()), (
            "non-finite operator weights after 30 steps"
        )
    # The VAE state must load into a fresh ToyConvAE (consumer usability).
    fresh = ToyConvAE(seed=cfg.seed)
    fresh.load_state_dict(ck["vae"])
    with torch.no_grad():
        x_probe = torch.rand(2, 3, cfg.image_size, cfg.image_size,
                             generator=torch.Generator().manual_seed(0))
        mu_probe, _ = fresh.encode_moments(x_probe)
        rec_probe = fresh.decode_latents(mu_probe)
    assert torch.isfinite(mu_probe).all() and torch.isfinite(rec_probe).all()
    # Completion sentinel: the full step budget was trained (SLURM scripts
    # gate eval_battery on this file; early saves must NOT write it).
    assert (out / "DONE").exists()
    assert (out / "DONE").read_text().strip() == "30"

    # Step-0 snapshot with the SPEC2 monitor keys, all finite where numeric.
    lines = _monitor_lines(out)
    snap = lines[0]
    assert snap["level"] == "monitor" and snap["step"] == 0
    assert MONITOR_KEYS <= set(snap)
    for key in ("l1", "lpips", "kl", "eff_rank"):
        assert snap[key] == snap[key], f"{key} is NaN"  # NaN != NaN
    assert all(v == v for v in snap["mu_std"])
    if cfg.tau_photo:
        assert isinstance(snap["ee_quick"], float) and snap["ee_quick"] == snap["ee_quick"]
        assert isinstance(snap["clip_fraction"], float)
    else:
        assert snap["ee_quick"] is None
        assert snap["clip_fraction"] is None


def test_p2_lie_operator_state_in_ckpt(data16: Path, tmp_path: Path) -> None:
    cfg = _tiny_config("p2_lie", steps=5)
    ckpt_path = train(cfg, str(data16), str(tmp_path / "run"), "cpu", vae="toy", workers=0)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "G" in ck["operator"]
    assert ck["operator"]["G"].shape == (4, 4, 4)


def test_wfit_path_is_used(data16: Path, tmp_path: Path) -> None:
    """--wfit payloads (pheq.probes.fit_w format) are loaded verbatim."""
    gen = torch.Generator().manual_seed(3)
    payload = {"W": torch.randn(3, 4, generator=gen), "c": torch.randn(3, generator=gen)}
    wfit_file = tmp_path / "wfit.pt"
    torch.save(payload, wfit_file)

    cfg = _tiny_config("p1_analytic", steps=3)
    ckpt_path = train(
        cfg, str(data16), str(tmp_path / "run"), "cpu",
        wfit_path=str(wfit_file), vae="toy", workers=0,
    )
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert torch.equal(ck["wfit"]["W"], payload["W"])
    assert torch.equal(ck["wfit"]["c"], payload["c"])


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------


def test_resume_continues_from_saved_step(data16: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    cfg8 = _tiny_config("b1", steps=8)
    p1 = train(cfg8, str(data16), str(out), "cpu", vae="toy", workers=0)
    assert torch.load(p1, map_location="cpu", weights_only=False)["step"] == 8

    # Resume with a larger step budget: continues from 8, ends at 16.
    cfg16 = _tiny_config("b1", steps=16)
    p2 = train(cfg16, str(data16), str(out), "cpu", vae="toy", workers=0)
    assert torch.load(p2, map_location="cpu", weights_only=False)["step"] == 16

    # Continuity: the resumed run did NOT restart — exactly one step-0
    # snapshot was ever written (a fresh run would have appended a second).
    zero_lines = [l for l in _monitor_lines(out) if l["level"] == "monitor" and l["step"] == 0]
    assert len(zero_lines) == 1
    # The resumed run wrote a segment marker so monitor.jsonl consumers can
    # discard stale lines from a hard-crashed predecessor.
    resumes = [l for l in _monitor_lines(out) if l["level"] == "resume"]
    assert [r["step"] for r in resumes] == [8]

    # A third call whose budget is already met returns immediately.
    p3 = train(cfg16, str(data16), str(out), "cpu", vae="toy", workers=0)
    assert torch.load(p3, map_location="cpu", weights_only=False)["step"] == 16


def test_resume_rejects_condition_mismatch(data16: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    train(_tiny_config("b1", steps=2), str(data16), str(out), "cpu", vae="toy", workers=0)
    with pytest.raises(ValueError, match="condition"):
        train(_tiny_config("b2lite", steps=4), str(data16), str(out), "cpu", vae="toy", workers=0)


# ---------------------------------------------------------------------------
# SIGTERM handler (SLURM preemption): saves and exits 0, in-process
# ---------------------------------------------------------------------------


def test_sigterm_saves_and_exits_zero(data16: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    cfg = _tiny_config("b1", steps=100_000)  # far more than 0.4 s of work
    timer = threading.Timer(0.4, os.kill, args=(os.getpid(), signal.SIGTERM))
    timer.start()
    try:
        with pytest.raises(SystemExit) as excinfo:
            train(cfg, str(data16), str(out), "cpu", vae="toy", workers=0)
    finally:
        timer.cancel()
    assert excinfo.value.code == 0
    ck = torch.load(out / "ckpt_latest.pt", map_location="cpu", weights_only=False)
    assert 0 < ck["step"] < 100_000
    # A preemption save is NOT completion: no DONE sentinel (SLURM scripts
    # would otherwise run eval_battery in the <=300 s grace window).
    assert not (out / "DONE").exists()
    # The handler was restored: a later SIGTERM must not be swallowed by a
    # stale train() handler (default disposition or pytest's own handler).
    assert signal.getsignal(signal.SIGTERM) is not None


def test_max_hours_saves_and_returns(data16: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    cfg = _tiny_config("b1", steps=100_000)
    path = train(
        cfg, str(data16), str(out), "cpu", vae="toy", workers=0, max_hours=1e-4
    )  # 0.36 s budget
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert 0 < ck["step"] < 100_000
    assert not (out / "DONE").exists()  # early save, not completion


# ---------------------------------------------------------------------------
# Monitor plumbing
# ---------------------------------------------------------------------------


def test_drift_alert_thresholds() -> None:
    ref = {"mu_std": [1.0, 2.0], "eff_rank": 3.0}
    ok = {"mu_std": [1.2, 1.8], "eff_rank": 3.5}  # all within 25%
    assert _drift_alert(ok, ref) == []
    bad_std = {"mu_std": [1.3, 2.0], "eff_rank": 3.0}  # +30% on channel 0
    assert any("mu_std[0]" in r for r in _drift_alert(bad_std, ref))
    bad_rank = {"mu_std": [1.0, 2.0], "eff_rank": 1.0}  # -67%
    assert any("eff_rank" in r for r in _drift_alert(bad_rank, ref))


# ---------------------------------------------------------------------------
# CLI (offline, --vae toy, < 60 s)
# ---------------------------------------------------------------------------


def test_cli_toy_offline(data32: Path, tmp_path: Path) -> None:
    out = tmp_path / "cli_run"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable, "-m", "pheq.train_ae",
        "--condition", "b1", "--data", str(data32), "--out", str(out),
        "--vae", "toy", "--steps", "5", "--device", "cpu", "--seed", "0",
        "--image-size", "32", "--batch-size", "4", "--workers", "0",
    ]
    result = subprocess.run(
        cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert (out / "ckpt_latest.pt").exists()
    assert (out / "monitor.jsonl").exists()
    assert "checkpoint" in result.stdout


# ---------------------------------------------------------------------------
# Known-good anchor (SPEC2): EE quick probe decreases for p1_analytic
# ---------------------------------------------------------------------------


def _run_anchor(data32: Path, out: Path, p_eq: float = 0.7) -> tuple[float, float]:
    """One 200-step p1_analytic anchor run; returns (ee0, ee200)."""
    cfg = get_config(
        "p1_analytic",
        steps=200,
        image_size=32,
        batch_size=8,
        lambda_lpips=0.0,
        lr=3e-3,
        p_eq=p_eq,
        seed=0,
    )
    train(cfg, str(data32), str(out), "cpu", vae="toy", workers=0)
    monitors = {
        l["step"]: l for l in _monitor_lines(out) if l["level"] == "monitor"
    }
    assert 0 in monitors and 200 in monitors
    ee0, ee200 = monitors[0]["ee_quick"], monitors[200]["ee_quick"]
    assert ee0 == ee0 and ee200 == ee200  # both finite (NaN != NaN)
    return ee0, ee200


@pytest.mark.slow
def test_anchor_ee_decreases_p1_analytic(
    data32: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole-loop anchor: on a color-shift-friendly toy setup (smooth
    synthetic color fields, high lr, eq branch taken often), 200 fine-tune
    steps of p1_analytic must reduce the decoder-side EE quick probe
    (frozen analytic operator, CIEDE2000) below its step-0 snapshot.

    The probe EE also shrinks under plain reconstruction training, so the
    raw decrease alone would not certify the equivariance wiring. Three runs
    on identical data/seed make the anchor discriminative (calibrated
    ratios ee200/ee0: correct 0.19, pure-recon baseline 0.22, INVERTED
    operator 0.46):

    1. correct run: ee200 < 0.3 * ee0 — a threshold the inverted-operator
       run does NOT meet;
    2. differential vs a p_eq = 0 baseline (same steps, pure recon): the
       eq branch must IMPROVE on the shared recon-driven decrease;
    3. negative control: with push_forward_posterior monkeypatched to the
       inverted operator z' = M^{-1}(z - m) (the canonical silent direction
       error), the run must FAIL both criteria above.
    """
    ee0, ee200 = _run_anchor(data32, tmp_path / "anchor")
    assert ee200 < ee0, f"EE quick probe did not decrease: {ee0:.3f} -> {ee200:.3f}"
    # Tightened threshold (calibrated): correct 0.19 passes, inverted 0.46
    # fails — this is no longer satisfiable by recon improvement + a wrong
    # operator direction.
    assert ee200 < 0.3 * ee0, f"EE decrease too small: {ee0:.3f} -> {ee200:.3f}"

    # (2) Differential: subtract the generic reconstruction improvement.
    base_ee0, base_ee200 = _run_anchor(data32, tmp_path / "base", p_eq=0.0)
    assert base_ee0 == ee0  # identical init => identical step-0 snapshot
    assert ee200 < base_ee200, (
        "equivariance training must beat pure-recon training on the EE "
        f"probe: p1 {ee200:.3f} !< baseline {base_ee200:.3f}"
    )

    # (3) Negative control: the inverted operator must NOT pass.
    import pheq.train_ae as train_ae_mod

    def _inverted_push_forward(mu, sigma, M, m):
        m_inv = torch.linalg.inv(M)
        mu_p = torch.einsum("ij,bjhw->bihw", m_inv, mu - m.view(1, -1, 1, 1))
        var_p = torch.einsum("ij,bjhw->bihw", m_inv * m_inv, sigma * sigma)
        return mu_p, var_p.clamp_min(1e-12).sqrt()

    monkeypatch.setattr(
        train_ae_mod, "push_forward_posterior", _inverted_push_forward
    )
    inv_ee0, inv_ee200 = _run_anchor(data32, tmp_path / "inverted")
    assert inv_ee0 == ee0
    assert inv_ee200 > 0.3 * inv_ee0, (
        "inverted-operator run unexpectedly met the anchor threshold "
        f"({inv_ee200:.3f} <= {0.3 * inv_ee0:.3f}): the anchor no longer "
        "discriminates the operator direction"
    )
    assert inv_ee200 > base_ee200, (
        "inverted operator should be WORSE than pure recon on the probe: "
        f"{inv_ee200:.3f} !> {base_ee200:.3f}"
    )
