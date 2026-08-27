"""Tests for pheq/fid.py (SPEC2.md "src/pheq/fid.py" section).

Fully OFFLINE per SPEC2 test-suite ground rules: cleanfid's FID computation is
never exercised — ``fid_available()`` guard logic is tested by monkeypatching,
and ``rfid`` is tested with ``pheq.fid.compute_fid`` monkeypatched so nothing
downloads Inception weights.

The rfid tests import the concurrently developed ``pheq.data`` inside the
tests (``pytest.importorskip``) so the rest of the file stays runnable before
integration.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import pheq.fid as fid_mod
from pheq.vae import ToyConvAE

SEED = 20260719


def _gen() -> torch.Generator:
    return torch.Generator().manual_seed(SEED)


# ---------------------------------------------------------------------------
# Laziness: importing pheq.fid must not import cleanfid (SPEC2 care point)
# ---------------------------------------------------------------------------


def test_import_does_not_touch_cleanfid() -> None:
    """`import pheq.fid` must leave cleanfid unimported (all imports lazy)."""
    repo_src = Path(__file__).resolve().parent.parent / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_src) + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import sys\n"
        "import pheq.fid\n"
        "assert 'cleanfid' not in sys.modules, 'cleanfid imported at module import time'\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# fid_available guard logic
# ---------------------------------------------------------------------------


def test_fid_available_returns_bool_without_downloading() -> None:
    assert isinstance(fid_mod.fid_available(), bool)


def test_fid_available_false_when_cleanfid_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # sys.modules[name] = None makes `import name` raise ImportError.
    monkeypatch.setitem(sys.modules, "cleanfid", None)
    assert fid_mod.fid_available() is False


def test_fid_available_false_when_weights_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("cleanfid")
    monkeypatch.setattr(
        fid_mod, "_inception_weight_path", lambda: tmp_path / "absent.pt"
    )
    assert fid_mod.fid_available() is False


def test_fid_available_true_when_import_ok_and_weights_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("cleanfid")
    weight = tmp_path / "inception-2015-12-05.pt"
    weight.write_bytes(b"stub")  # presence check only; never loaded
    monkeypatch.setattr(fid_mod, "_inception_weight_path", lambda: weight)
    assert fid_mod.fid_available() is True


def test_inception_weight_path_matches_cleanfid_convention() -> None:
    """The mirrored path must track the installed cleanfid's weight filename."""
    dh = pytest.importorskip("cleanfid.downloads_helper")
    path = fid_mod._inception_weight_path()
    assert path.name == os.path.basename(dh.inception_url)
    if sys.platform != "win32":
        assert path.parent == Path("/tmp")


# ---------------------------------------------------------------------------
# PNG round-trip: save/load preserves values to 1/255 (SPEC2 test requirement)
# ---------------------------------------------------------------------------


def test_png_roundtrip_preserves_values_to_1_over_255(tmp_path: Path) -> None:
    img = torch.rand((3, 17, 23), generator=_gen())
    path = tmp_path / "rt.png"
    fid_mod._save_png(img, path)
    loaded = fid_mod._load_png(path)
    assert loaded.shape == img.shape
    assert loaded.dtype == torch.float32
    assert (loaded - img).abs().max().item() <= 1.0 / 255.0 + 1e-6
    # Quantized values round-trip exactly: save(load(png)) is idempotent.
    fid_mod._save_png(loaded, path)
    assert torch.equal(fid_mod._load_png(path), loaded)


def test_png_save_clamps_out_of_range(tmp_path: Path) -> None:
    img = torch.full((3, 4, 4), 0.5)
    img[0, 0, 0] = -0.5
    img[1, 0, 0] = 1.5
    path = tmp_path / "clamp.png"
    fid_mod._save_png(img, path)
    loaded = fid_mod._load_png(path)
    assert loaded[0, 0, 0].item() == 0.0
    assert loaded[1, 0, 0].item() == 1.0
    assert loaded.min().item() >= 0.0 and loaded.max().item() <= 1.0


# ---------------------------------------------------------------------------
# rfid: ToyConvAE round-trip WITHOUT cleanfid (compute_fid monkeypatched)
# ---------------------------------------------------------------------------


def _write_source_images(src: Path, n: int = 5) -> None:
    """Write n deterministic RGB PNGs of varied sizes (exercise resize+crop)."""
    src.mkdir(parents=True, exist_ok=True)
    gen = _gen()
    sizes = [(40, 32), (32, 48), (36, 36), (52, 40), (32, 32)]
    for i in range(n):
        h, w = sizes[i % len(sizes)]
        fid_mod._save_png(torch.rand((3, h, w), generator=gen), src / f"img_{i}.png")


def _patch_compute_fid(monkeypatch: pytest.MonkeyPatch, calls: list) -> None:
    def fake_compute_fid(dir_a: str, dir_b: str, mode: str = "clean") -> float:
        calls.append((dir_a, dir_b, mode))
        return 12.5

    monkeypatch.setattr(fid_mod, "compute_fid", fake_compute_fid)


def test_rfid_dirs_and_image_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = pytest.importorskip(
        "pheq.data", reason="pheq.data not present yet (sibling v2 lane)"
    )
    src = tmp_path / "src"
    _write_source_images(src, n=5)
    calls: list = []
    _patch_compute_fid(monkeypatch, calls)

    ae = ToyConvAE(seed=0)
    out = tmp_path / "out"
    score = fid_mod.rfid(ae, str(src), str(out), n=4, device="cpu", batch=3, size=16)

    # Monkeypatched FID reached with the two output dirs.
    assert score == 12.5
    assert len(calls) == 1
    assert calls[0][0] == str(out / "real")
    assert calls[0][1] == str(out / "recon")

    real = sorted((out / "real").glob("*.png"))
    recon = sorted((out / "recon").glob("*.png"))
    assert len(real) == 4 and len(recon) == 4

    # "real" PNGs == IDENTICAL preprocessing as data.ImageFolderDataset,
    # preserved to 1/255 by the uint8 PNG round-trip (SPEC2 care point).
    dataset = data.ImageFolderDataset(str(src), size=16)
    with torch.no_grad():
        for i, (real_p, recon_p) in enumerate(zip(real, recon)):
            img, _label = dataset[i]
            assert img.shape == (3, 16, 16)
            got_real = fid_mod._load_png(real_p)
            assert (got_real - img.clamp(0, 1)).abs().max().item() <= 1 / 255 + 1e-6

            # "recon" PNGs == decode(posterior mean), clamped at PNG save.
            mu, _sigma = ae.encode_moments(img.unsqueeze(0))
            want = ae.decode_latents(mu)[0].clamp(0, 1)
            got_recon = fid_mod._load_png(recon_p)
            assert (got_recon - want).abs().max().item() <= 1 / 255 + 1e-6


def test_rfid_caps_n_at_dataset_size_and_clears_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip(
        "pheq.data", reason="pheq.data not present yet (sibling v2 lane)"
    )
    src = tmp_path / "src"
    _write_source_images(src, n=3)
    calls: list = []
    _patch_compute_fid(monkeypatch, calls)
    out = tmp_path / "out"

    # Stale PNG from a "previous run" must not survive into the FID dirs.
    stale = out / "recon" / "zzz_stale.png"
    stale.parent.mkdir(parents=True)
    fid_mod._save_png(torch.zeros(3, 4, 4), stale)

    ae = ToyConvAE(seed=1)
    fid_mod.rfid(ae, str(src), str(out), n=50, device="cpu", batch=2, size=16)
    assert not stale.exists()
    assert len(list((out / "real").glob("*.png"))) == 3
    assert len(list((out / "recon").glob("*.png"))) == 3


def test_rfid_from_run_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """rfid accepts a run-checkpoint path (SPEC2 v2 checkpoint format)."""
    pytest.importorskip(
        "pheq.data", reason="pheq.data not present yet (sibling v2 lane)"
    )
    src = tmp_path / "src"
    _write_source_images(src, n=3)
    calls: list = []
    _patch_compute_fid(monkeypatch, calls)

    ae = ToyConvAE(channels=4, hidden=8, seed=3)
    ckpt = {
        "vae": ae.state_dict(),
        "operator": None,
        "operator_kind": "none",
        "condition": "b1",
        "step": 0,
        "wfit": {"W": torch.zeros(3, 4), "c": torch.zeros(3)},
        "config": {"vae": "toy"},
    }
    ckpt_path = tmp_path / "ckpt_latest.pt"
    torch.save(ckpt, ckpt_path)

    out = tmp_path / "out"
    score = fid_mod.rfid(
        str(ckpt_path), str(src), str(out), n=3, device="cpu", batch=2, size=16
    )
    assert score == 12.5

    # Reconstructions must come from the checkpoint's weights (== ae's).
    import pheq.data as data

    dataset = data.ImageFolderDataset(str(src), size=16)
    recon = sorted((out / "recon").glob("*.png"))
    assert len(recon) == 3
    with torch.no_grad():
        img, _ = dataset[0]
        mu, _sigma = ae.encode_moments(img.unsqueeze(0))
        want = ae.decode_latents(mu)[0].clamp(0, 1)
    got = fid_mod._load_png(recon[0])
    assert (got - want).abs().max().item() <= 1 / 255 + 1e-6


def test_resolve_vae_module_passthrough_and_bad_ckpt(tmp_path: Path) -> None:
    ae = ToyConvAE(seed=0)
    resolved = fid_mod._resolve_vae(ae, "cpu")
    assert resolved is ae
    assert not resolved.training  # eval mode enforced
    with pytest.raises(KeyError):
        fid_mod._resolve_vae({"config": {"vae": "toy"}}, "cpu")  # no 'vae' key


def test_patched_frechet_distance_matches_closed_form() -> None:
    # Regression: scipy 1.18 removed sqrtm's `disp` kwarg; cleanfid 0.1.35
    # still passes it and every FID call raised TypeError. Our patched
    # frechet_distance must match the closed form for diagonal Gaussians:
    # d^2 = |mu1-mu2|^2 + sum(s1 + s2 - 2*sqrt(s1*s2)).
    import numpy as np

    from pheq.fid import _patched_frechet_distance

    rng = np.random.default_rng(0)
    mu1, mu2 = rng.normal(size=4), rng.normal(size=4)
    s1, s2 = rng.uniform(0.5, 2.0, size=4), rng.uniform(0.5, 2.0, size=4)
    got = _patched_frechet_distance(mu1, np.diag(s1), mu2, np.diag(s2))
    want = float(((mu1 - mu2) ** 2).sum() + (s1 + s2 - 2 * np.sqrt(s1 * s2)).sum())
    assert abs(got - want) < 1e-8


def test_ensure_cleanfid_patched_idempotent() -> None:
    cleanfid = pytest.importorskip("cleanfid")
    import cleanfid.fid as cf

    from pheq.fid import _ensure_cleanfid_patched

    _ensure_cleanfid_patched()
    first = cf.frechet_distance
    _ensure_cleanfid_patched()
    assert cf.frechet_distance is first
    assert first.__name__ == "_patched_frechet_distance"
