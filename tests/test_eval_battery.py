"""Tests for pheq.eval_battery (SPEC2 "eval_battery.py").

Fully offline: ToyConvAE run checkpoints, tiny synthetic images, LPIPS
disabled (``use_lpips=False``), cleanfid monkeypatched for the rFID hook.
Deterministic seeds throughout (SPEC2 ground rules).
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

import pheq.eval_battery as eb
import pheq.fid as fid_mod
from pheq.analytic import fit_w
from pheq.conditions import DEFAULT_PHOTO_RANGES, TrainConfig
from pheq.probes._common import DEFAULT_GRIDS, FACTORS, ORACLE_GRIDS, synthetic_images
from pheq.vae import ToyConvAE

IMG_SIZE = 32
N_IMAGES = 6


def _write_images(directory: Path, n: int, size: int = IMG_SIZE, seed: int = 0) -> None:
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    imgs = synthetic_images(n, size, seed=seed)
    for i in range(n):
        arr = (imgs[i].clamp(0, 1) * 255.0).round().to(torch.uint8)
        Image.fromarray(arr.permute(1, 2, 0).numpy(), mode="RGB").save(
            directory / f"img{i:03d}.png"
        )


def _make_ckpt(path: Path, operator_kind: str = "none", condition: str = "b1") -> dict:
    """A SPEC2 run checkpoint around a ToyConvAE, with a real fitted wfit."""
    vae = ToyConvAE(channels=4, hidden=8, seed=1)
    imgs = synthetic_images(8, IMG_SIZE, seed=0)
    with torch.no_grad():
        z = vae.encode(imgs)
    fit = fit_w(z, imgs)

    operator = None
    if operator_kind == "lie":
        from pheq.lie_operator import LieAffineOperator

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            operator = LieAffineOperator(channels=4, hidden=16, n_freq=4).state_dict()
    elif operator_kind == "conv":
        from pheq.conv_operator import ConvResidualOperator

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            operator = ConvResidualOperator(fit, hidden=16, n_freq=4).state_dict()

    ckpt = {
        "vae": vae.state_dict(),
        "operator": operator,
        "operator_kind": operator_kind,
        "condition": condition,
        "step": 123,
        "wfit": {"W": fit.W, "c": fit.c},
        "config": {
            "vae": "toy",
            "condition": condition,
            "image_size": IMG_SIZE,
            "photo_ranges": {k: tuple(v) for k, v in DEFAULT_PHOTO_RANGES.items()},
        },
        # ckpt_latest.pt carries extra keys in real runs — must be tolerated:
        "optimizer": {"state": {}, "param_groups": []},
    }
    torch.save(ckpt, path)
    return ckpt


@pytest.fixture(scope="module")
def battery(tmp_path_factory: pytest.TempPathFactory):
    """One full offline battery run on a ToyConvAE checkpoint (reused across tests)."""
    root = tmp_path_factory.mktemp("battery")
    img_dir = root / "images"
    _write_images(img_dir, N_IMAGES)
    ckpt_path = root / "ckpt_latest.pt"
    _make_ckpt(ckpt_path)
    out_json = root / "eval.json"
    result = eb.evaluate(
        str(ckpt_path),
        str(img_dir),
        str(out_json),
        device="cpu",
        n_images=N_IMAGES,
        rfid_n=0,
        batch=4,
        seed=0,
        use_lpips=False,
    )
    return {"root": root, "img_dir": img_dir, "ckpt_path": ckpt_path,
            "out_json": out_json, "result": result}


# ---------------------------------------------------------------------------
# Full battery on a toy checkpoint + stable JSON schema (SPEC2 tests)
# ---------------------------------------------------------------------------


def test_full_battery_schema(battery) -> None:
    result = battery["result"]

    # Top-level schema (gap-closed block absent without refs — SPEC2; the
    # refit-W blocks absent by default — tier0-log W1 caveat 1 addition;
    # ee_spatial ALWAYS present — caveat 2 addition).
    assert set(result) == {
        "ckpt", "eval", "reconstruction", "ee", "ee_spatial", "latent", "collapse",
    }
    assert "ee_refit_w" not in result and "refit_w" not in result  # --refit-w off
    assert set(result["ckpt"]) == {"path", "condition", "operator_kind", "step"}
    assert set(result["eval"]) == {"n_images", "image_size", "batch", "device", "seed"}
    assert set(result["reconstruction"]) == {"psnr", "l1", "lpips", "rfid"}
    assert set(result["ee"]) == {"grid", "held_out", "compositions"}
    assert set(result["latent"]) == {
        "mu_std_per_channel", "effective_rank", "spectral_slope", "high_freq_fraction",
    }
    assert set(result["collapse"]) == {"cross_operator", "swapped_decoder"}

    assert result["ckpt"]["condition"] == "b1"
    assert result["ckpt"]["step"] == 123
    assert result["reconstruction"]["lpips"] is None  # use_lpips=False
    assert result["reconstruction"]["rfid"] is None  # rfid_n=0
    assert math.isfinite(result["reconstruction"]["psnr"])
    assert result["reconstruction"]["l1"] >= 0.0

    # Grid = union of the imported probe grids, per factor (SPEC2: reuse).
    expected_rows = sum(
        len(set(DEFAULT_GRIDS[f]) | set(ORACLE_GRIDS[f])) for f in FACTORS
    )
    grid = result["ee"]["grid"]
    assert len(grid) == expected_rows
    row_keys = {"factor", "magnitude", "ee_l2", "ee_ciede2000", "clip_frac"}
    for row in grid + result["ee"]["held_out"]:
        assert set(row) == row_keys
        assert row["factor"] in FACTORS
        assert row["ee_l2"] >= 0.0 and math.isfinite(row["ee_l2"])
        assert row["ee_ciede2000"] >= 0.0 and math.isfinite(row["ee_ciede2000"])
        assert 0.0 <= row["clip_frac"] <= 1.0

    # Held-out magnitudes: two midpoints (identity↔range-end) per factor,
    # e.g. beta ∈ [0.7, 1.3] → 0.85 and 1.15 (spec resolution in module doc).
    held = result["ee"]["held_out"]
    assert len(held) == 8
    bri = sorted(r["magnitude"] for r in held if r["factor"] == "brightness")
    assert bri == pytest.approx([0.85, 1.15])

    comps = result["ee"]["compositions"]
    assert set(comps) == {"n", "seed", "ee_l2", "ee_ciede2000", "clip_frac"}
    assert comps["n"] == 8

    # Spatial-EE block: the four exact ops + their mean (tier0-log W1 caveat 2).
    sp = result["ee_spatial"]
    assert set(sp) == {"ops", "mean"}
    assert [r["op"] for r in sp["ops"]] == ["rot90_k1", "rot90_k2", "scale_0.5", "scale_0.25"]
    for r in sp["ops"]:
        assert set(r) == {"op", "rot90", "scale", "ee_l2", "ee_ciede2000"}
        assert r["ee_l2"] >= 0.0 and math.isfinite(r["ee_l2"])
        assert r["ee_ciede2000"] >= 0.0 and math.isfinite(r["ee_ciede2000"])
    for key in ("ee_l2", "ee_ciede2000"):
        assert sp["mean"][key] == pytest.approx(
            sum(r[key] for r in sp["ops"]) / len(sp["ops"])
        )

    lat = result["latent"]
    assert len(lat["mu_std_per_channel"]) == 4
    assert 1.0 <= lat["effective_rank"] <= 4.0 + 1e-6
    assert math.isfinite(lat["spectral_slope"])
    assert 0.0 <= lat["high_freq_fraction"] <= 1.0

    # operator_kind 'none' → the ckpt operator IS the frozen analytic one.
    cross = result["collapse"]["cross_operator"]
    assert cross["identical_to_main"] is True
    assert set(cross["per_factor"]) == set(FACTORS)

    # Swapped-decoder test is skipped for toy checkpoints (SPEC2).
    sw = result["collapse"]["swapped_decoder"]
    assert sw["skipped"] is True
    assert "toy" in sw["reason"]
    assert sw["per_factor"] is None


def test_json_written_matches_return(battery) -> None:
    on_disk = json.loads(battery["out_json"].read_text())
    assert on_disk == battery["result"]


# ---------------------------------------------------------------------------
# Gap-closed math on synthetic numbers (SPEC2 test)
# ---------------------------------------------------------------------------


def test_gap_closed_math_synthetic() -> None:
    ck = {"brightness": {0.6: 4.0, 1.4: 6.0}, "hue": {0.5: 3.0}}
    b1 = {"brightness": {0.6: 10.0, 1.4: 10.0}, "hue": {0.5: 3.0}}
    orc = {"brightness": {0.6: 2.0, 1.4: 2.0}, "hue": {0.5: 3.0}}

    res = eb.gap_closed_fraction(ck, b1, orc)
    # brightness: means ck 5, b1 10, oracle 2 → (10−5)/(10−2) = 0.625
    assert res["brightness"]["gap_closed"] == pytest.approx(0.625)
    assert res["brightness"]["n_magnitudes"] == 2
    # hue: b1 == oracle → degenerate gap → None
    assert res["hue"]["gap_closed"] is None

    # 100% closed (ckpt reaches the oracle) and 0% closed (ckpt == b1).
    res2 = eb.gap_closed_fraction(
        {"contrast": {1.0: 2.0}}, {"contrast": {1.0: 8.0}}, {"contrast": {1.0: 2.0}}
    )
    assert res2["contrast"]["gap_closed"] == pytest.approx(1.0)
    res3 = eb.gap_closed_fraction(
        {"contrast": {1.0: 8.0}}, {"contrast": {1.0: 8.0}}, {"contrast": {1.0: 2.0}}
    )
    assert res3["contrast"]["gap_closed"] == pytest.approx(0.0)

    # Factors with no overlap (missing map or disjoint magnitudes) are omitted.
    res4 = eb.gap_closed_fraction(
        {"saturation": {0.5: 1.0}, "hue": {0.1: 1.0}},
        {"saturation": {0.7: 1.0}},
        {"saturation": {0.5: 1.0}},
    )
    assert res4 == {}


# ---------------------------------------------------------------------------
# The gap-closed block appears ONLY with BOTH refs (SPEC2 care point)
# ---------------------------------------------------------------------------


def _write_oracle_csv(path: Path, de00: float = 0.0) -> None:
    header = ("factor", "magnitude", "oracle_loss", "oracle_ciede2000",
              "affine_r2", "identity_l2")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for factor, mags in ORACLE_GRIDS.items():
            for mag in mags:
                writer.writerow([factor, mag, 0.001, de00, 0.95, 0.05])


def test_gap_block_only_with_both_refs(battery, tmp_path: Path) -> None:
    result = battery["result"]
    assert "oracle_gap_closed" not in result  # no refs

    # b1 reference: this battery's own grid with +1.0 ΔE00 everywhere.
    b1 = json.loads(battery["out_json"].read_text())
    for row in b1["ee"]["grid"]:
        row["ee_ciede2000"] += 1.0
    b1_path = tmp_path / "b1.json"
    b1_path.write_text(json.dumps(b1))
    oracle_path = tmp_path / "oracle.csv"
    _write_oracle_csv(oracle_path, de00=0.0)

    # Only one ref → block still omitted.
    partial = eb.evaluate(
        str(battery["ckpt_path"]), str(battery["img_dir"]),
        str(tmp_path / "partial.json"), n_images=N_IMAGES, batch=4,
        b1_json=str(b1_path), oracle_csv=None, use_lpips=False,
    )
    assert "oracle_gap_closed" not in partial

    full = eb.evaluate(
        str(battery["ckpt_path"]), str(battery["img_dir"]),
        str(tmp_path / "full.json"), n_images=N_IMAGES, batch=4,
        b1_json=str(b1_path), oracle_csv=str(oracle_path), use_lpips=False,
    )
    block = full["oracle_gap_closed"]
    assert block["metric"] == "ciede2000"
    assert set(block["per_factor"]) == set(FACTORS)

    # Expected per factor: shared magnitudes are the ORACLE_GRIDS endpoints;
    # b1 = ckpt + 1, oracle = 0 → gap = 1 / (mean_ckpt + 1).
    grid_map = eb._de00_map_from_rows(full["ee"]["grid"])
    for factor in FACTORS:
        shared = [round(m, 6) for m in ORACLE_GRIDS[factor]]
        mean_ck = sum(grid_map[factor][m] for m in shared) / len(shared)
        g = block["per_factor"][factor]
        assert g["n_magnitudes"] == len(shared)
        assert g["ee_ckpt"] == pytest.approx(mean_ck)
        assert g["ee_b1"] == pytest.approx(mean_ck + 1.0)
        assert g["ee_oracle"] == pytest.approx(0.0)
        assert g["gap_closed"] == pytest.approx(1.0 / (mean_ck + 1.0))


# ---------------------------------------------------------------------------
# Spatial-EE block: exact-op sanity (tier0-log W1 caveat 2)
# ---------------------------------------------------------------------------


def test_spatial_ee_pointwise_decoder_rot90_equals_recon() -> None:
    """For a POINTWISE decoder, rot90 commutes with decoding exactly, so
    spatial EE at k=1/k=2 equals the plain reconstruction error bit-close
    (both metrics are invariant under a permutation of pixel sites)."""
    gen = torch.Generator().manual_seed(0)
    z = torch.randn(3, 4, 16, 16, generator=gen)
    w = 0.15 * torch.randn(3, 4, generator=gen)
    c = torch.full((3,), 0.5)

    def decode_fn(zz: torch.Tensor) -> torch.Tensor:
        return torch.einsum("dc,bchw->bdhw", w, zz) + c[None, :, None, None]

    # Images = decoded latents + a fixed perturbation, so the "recon error"
    # is a known nonzero floor shared by every exactly-commuting op.
    images = decode_fn(z) + 0.05 * torch.randn(3, 3, 16, 16, generator=gen)
    recon_l2, recon_de = eb._ee_pair(decode_fn, z, images, batch=2)
    assert recon_l2 > 0.0

    block = eb._spatial_ee_block(images, z, decode_fn, batch=2, f_img=1)
    by_op = {r["op"]: r for r in block["ops"]}
    for op in ("rot90_k1", "rot90_k2"):
        assert by_op[op]["ee_l2"] == pytest.approx(recon_l2, rel=1e-6)
        assert by_op[op]["ee_ciede2000"] == pytest.approx(recon_de, rel=1e-5)
    # Scale ops: bilinear interpolation commutes with the pointwise affine
    # decoder in real arithmetic (convex weights), so the EE is the resampled
    # perturbation — same scale as (and no larger than) the recon floor.
    for op in ("scale_0.5", "scale_0.25"):
        assert 0.05 * recon_l2 < by_op[op]["ee_l2"] < 1.05 * recon_l2


def test_spatial_ee_frozen_toy_recon_scale(battery) -> None:
    """Frozen-toy anchor: the exact spatial ops introduce no NEW error source,
    so each spatial EE stays on the plain-reconstruction-error scale."""
    result = battery["result"]
    vae = fid_mod._resolve_vae(str(battery["ckpt_path"]), "cpu")
    from pheq.data import ImageFolderDataset

    ds = ImageFolderDataset(str(battery["img_dir"]), size=IMG_SIZE)
    x = torch.stack([ds[i][0] for i in range(N_IMAGES)])
    with torch.no_grad():
        mu, _sigma = vae.encode_moments(x)
    recon_l2, recon_de = eb._ee_pair(vae.decode_latents, mu, x, batch=4)

    for row in result["ee_spatial"]["ops"]:
        assert 0.2 * recon_l2 < row["ee_l2"] < 5.0 * recon_l2
        assert 0.2 * recon_de < row["ee_ciede2000"] < 5.0 * recon_de


# ---------------------------------------------------------------------------
# Refit-W reference on a deliberately drifted toy AE (tier0-log W1 caveat 1)
# ---------------------------------------------------------------------------


def _linearized_toy() -> ToyConvAE:
    """A ToyConvAE with PLANTED weights making it ≈ exactly linear.

    Both SiLU layers are driven into their linear region with a large bias
    (SiLU(15 + v) = 15 + v up to ~4e-5 for |v| ≤ 2), which downstream layers
    subtract again. The result: encode(x) ≈ (r̄, ḡ, b̄, 0) per 2×2 block and
    decode(z) ≈ broadcast of z's first three channels — a pointwise-affine
    decoder with planted W ≈ [I₃ | 0], c ≈ 0 (the exact-linear-decoder
    setting of plan §3.2, in which the analytic operator with the CORRECT W
    is exactly equivariant, so a stale-vs-refit EE gap is attributable to
    the W alone)."""
    vae = ToyConvAE(channels=4, hidden=8, seed=1)
    lift = 15.0
    with torch.no_grad():
        conv1 = vae.encoder[0]  # Conv2d(3, 8, 3, pad 1)
        conv1.weight.zero_()
        conv1.bias.zero_()
        for i in range(3):
            conv1.weight[i, i, 1, 1] = 1.0  # center tap only: padding-safe
            conv1.bias[i] = lift
        conv2 = vae.encoder[2]  # Conv2d(8, 4, 2, stride 2)
        conv2.weight.zero_()
        conv2.bias.zero_()
        for c in range(3):
            conv2.weight[c, c] = 0.25  # 2x2 block mean
            conv2.bias[c] = -lift
        deconv = vae.decoder[0]  # ConvTranspose2d(4, 8, 2, 2): weight (C, hidden, 2, 2)
        deconv.weight.zero_()
        deconv.bias.zero_()
        for i in range(3):
            deconv.weight[i, i] = 1.0  # replicate z_i into the block
            deconv.bias[i] = lift
        conv3 = vae.decoder[2]  # Conv2d(8, 3, 3, pad 1)
        conv3.weight.zero_()
        conv3.bias.fill_(-lift)
        for c in range(3):
            conv3.weight[c, c, 1, 1] = 1.0
    return vae


def _drifted_ckpt(path: Path) -> None:
    """Checkpoint whose latent basis drifted AFTER the stored wfit was fit.

    Constructs the b1 phenomenon of docs/tier0-log.md WAVE-1 caveat 1
    deliberately: fit (W, c) on the linearized ToyConvAE's latents, then
    apply an invertible channel mix R to the encoder output and R^{-1} to the
    decoder input. Reconstruction is UNCHANGED (decode ∘ encode identical),
    but the latents are R z — the stored wfit is stale by exactly R, while a
    refit on the drifted latents recovers W R^{-1}.
    """
    vae = _linearized_toy()
    imgs = synthetic_images(8, IMG_SIZE, seed=0)
    with torch.no_grad():
        z = vae.encode(imgs)
    fit = fit_w(z, imgs)  # the "frozen-VAE-era" wfit — stale after the drift

    gen = torch.Generator().manual_seed(3)
    r_mat = torch.eye(4) + 0.7 * torch.randn(4, 4, generator=gen)
    r_inv = torch.linalg.inv(r_mat)
    with torch.no_grad():
        enc = vae.encoder[2]  # Conv2d(hidden, C, 2, stride 2): weight (C, hidden, 2, 2)
        enc.weight.copy_(torch.einsum("dc,cikw->dikw", r_mat, enc.weight))
        enc.bias.copy_(r_mat @ enc.bias)
        dec = vae.decoder[0]  # ConvTranspose2d(C, hidden, 2, 2): weight (C, hidden, 2, 2)
        dec.weight.copy_(torch.einsum("cd,cokw->dokw", r_inv, dec.weight))

    torch.save(
        {
            "vae": vae.state_dict(),
            "operator": None,
            "operator_kind": "none",
            "condition": "b1",
            "step": 123,
            "wfit": {"W": fit.W, "c": fit.c},
            "config": {
                "vae": "toy",
                "condition": "b1",
                "image_size": IMG_SIZE,
                "photo_ranges": {k: tuple(v) for k, v in DEFAULT_PHOTO_RANGES.items()},
            },
        },
        path,
    )


def test_refit_w_beats_stale_on_drifted_toy(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    _write_images(img_dir, N_IMAGES)
    ckpt_path = tmp_path / "ckpt.pt"
    _drifted_ckpt(ckpt_path)

    result = eb.evaluate(
        str(ckpt_path), str(img_dir), str(tmp_path / "eval.json"),
        n_images=N_IMAGES, batch=4, use_lpips=False, refit_w=True,
    )

    # Schema: the refit blocks mirror the "ee" block + record the refit R².
    assert set(result["ee_refit_w"]) == {"grid", "held_out", "compositions"}
    assert set(result["refit_w"]) == {"r2", "r2_per_channel"}
    assert len(result["ee_refit_w"]["grid"]) == len(result["ee"]["grid"])
    assert len(result["ee_refit_w"]["held_out"]) == len(result["ee"]["held_out"])
    assert len(result["refit_w"]["r2_per_channel"]) == 3
    # The linearized toy is exactly linear latent→RGB: the refit is perfect.
    assert result["refit_w"]["r2"] == pytest.approx(1.0, abs=1e-4)

    def mean_de00(rows: list[dict]) -> float:
        return sum(r["ee_ciede2000"] for r in rows) / len(rows)

    stale = mean_de00(result["ee"]["grid"])
    refit = mean_de00(result["ee_refit_w"]["grid"])
    # The refit analytic operator must CLEARLY beat the stale-wfit one on the
    # drifted latents (the whole point of the fair b1 reference; measured
    # margin ~30% on the grid, ~40% on held-out for this construction).
    assert refit < 0.85 * stale
    assert mean_de00(result["ee_refit_w"]["held_out"]) < 0.85 * mean_de00(
        result["ee"]["held_out"]
    )
    assert (
        result["ee_refit_w"]["compositions"]["ee_ciede2000"]
        < 0.85 * result["ee"]["compositions"]["ee_ciede2000"]
    )

    on_disk = json.loads((tmp_path / "eval.json").read_text())
    assert on_disk == result


def test_cli_refit_w_flag(battery, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    out_json = tmp_path / "cli_refit.json"
    eb.main([
        "--ckpt", str(battery["ckpt_path"]),
        "--images", str(battery["img_dir"]),
        "--out", str(out_json),
        "--n-images", "4",
        "--batch", "4",
        "--no-lpips",
        "--refit-w",
    ])
    data = json.loads(out_json.read_text())
    assert "ee_refit_w" in data and "refit_w" in data
    out = capsys.readouterr().out
    assert "refit-w: R2" in out


# ---------------------------------------------------------------------------
# Frozen-b0 mode (tier0-log W1 caveat 3)
# ---------------------------------------------------------------------------


def _write_frozen_wfit(path: Path, img_dir: Path) -> None:
    """wfit.pt for the frozen ToyConvAE (default construction: seed 0)."""
    from pheq.data import ImageFolderDataset
    from pheq.vae import ToyConvAE as _Toy

    vae = _Toy()
    ds = ImageFolderDataset(str(img_dir), size=IMG_SIZE)
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    with torch.no_grad():
        z = vae.encode(x)
    fit = fit_w(z, x)
    torch.save({"W": fit.W, "c": fit.c, "r2": fit.r2}, path)


def test_frozen_vae_mode_full_schema(battery, tmp_path: Path) -> None:
    wfit_path = tmp_path / "wfit.pt"
    _write_frozen_wfit(wfit_path, battery["img_dir"])

    ckpt = eb.make_frozen_checkpoint("toy", str(wfit_path), image_size=IMG_SIZE)
    assert ckpt["condition"] == "b0"
    assert ckpt["operator_kind"] == "none"
    assert ckpt["operator"] is None
    assert ckpt["step"] == 0
    assert ckpt["config"]["vae"] == "toy"

    # The config sub-dict is schema-identical to train_ae's writer
    # (asdict(TrainConfig) + "vae"): a future consumer reading e.g.
    # config["condition"] or config["seed"] must not KeyError in b0 mode.
    expected_config_keys = set(
        asdict(
            TrainConfig(
                condition="b0", operator_kind="none",
                tau_photo=False, tau_spatial=False,
            )
        )
    ) | {"vae"}
    assert set(ckpt["config"]) == expected_config_keys
    assert ckpt["config"]["condition"] == "b0"
    assert ckpt["config"]["operator_kind"] == "none"
    assert ckpt["config"]["steps"] == 0
    assert ckpt["config"]["image_size"] == IMG_SIZE
    assert ckpt["config"]["photo_ranges"] == {
        k: tuple(v) for k, v in DEFAULT_PHOTO_RANGES.items()
    }

    out_json = tmp_path / "b0.json"
    result = eb.evaluate(
        ckpt, str(battery["img_dir"]), str(out_json),
        n_images=4, batch=4, use_lpips=False,
    )
    # Full schema, unchanged downstream (tier0-log W1 caveat 3).
    assert set(result) == {
        "ckpt", "eval", "reconstruction", "ee", "ee_spatial", "latent", "collapse",
    }
    assert result["ckpt"]["condition"] == "b0"
    assert result["ckpt"]["operator_kind"] == "none"
    assert result["ckpt"]["step"] == 0
    assert result["ckpt"]["path"] == "<in-memory:b0>"
    assert result["eval"]["image_size"] == IMG_SIZE
    assert math.isfinite(result["reconstruction"]["psnr"])
    assert len(result["ee"]["grid"]) > 0
    assert json.loads(out_json.read_text()) == result


def test_frozen_vae_default_image_size() -> None:
    # Probe conventions: 256 for sd, 64 for toy (only 'toy' is offline-safe).
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        wfit_path = Path(d) / "wfit.pt"
        torch.save({"W": torch.zeros(3, 4), "c": torch.zeros(3)}, wfit_path)
        ckpt = eb.make_frozen_checkpoint("toy", str(wfit_path))
        assert ckpt["config"]["image_size"] == 64
        with pytest.raises(ValueError, match="'sd' or 'toy'"):
            eb.make_frozen_checkpoint("nope", str(wfit_path))


def test_frozen_vae_cli(battery, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    wfit_path = tmp_path / "wfit.pt"
    _write_frozen_wfit(wfit_path, battery["img_dir"])
    out_json = tmp_path / "b0_cli.json"
    eb.main([
        "--frozen-vae", "toy",
        "--wfit", str(wfit_path),
        "--image-size", str(IMG_SIZE),
        "--images", str(battery["img_dir"]),
        "--out", str(out_json),
        "--n-images", "4",
        "--batch", "4",
        "--no-lpips",
    ])
    data = json.loads(out_json.read_text())
    assert data["ckpt"]["condition"] == "b0"
    assert "ee_spatial" in data
    assert "b0" in capsys.readouterr().out


def test_frozen_vae_cli_mutual_exclusion(battery, tmp_path: Path) -> None:
    common = ["--images", str(battery["img_dir"]), "--out", str(tmp_path / "x.json")]
    # --ckpt and --frozen-vae are mutually exclusive (argparse exits 2).
    with pytest.raises(SystemExit):
        eb.main(["--ckpt", str(battery["ckpt_path"]), "--frozen-vae", "toy",
                 "--wfit", "w.pt", *common])
    # --frozen-vae requires --wfit.
    with pytest.raises(SystemExit):
        eb.main(["--frozen-vae", "toy", *common])
    # --wfit / --image-size are frozen-mode-only.
    with pytest.raises(SystemExit):
        eb.main(["--ckpt", str(battery["ckpt_path"]), "--wfit", "w.pt", *common])
    with pytest.raises(SystemExit):
        eb.main(["--ckpt", str(battery["ckpt_path"]), "--image-size", "32", *common])
    # Neither source given.
    with pytest.raises(SystemExit):
        eb.main(common)


# ---------------------------------------------------------------------------
# Gap-closed prefers the b1 ee_refit_w block; stale fallback is marked
# ---------------------------------------------------------------------------


def test_gap_closed_prefers_refit_w_and_marks_stale(battery, tmp_path: Path) -> None:
    import copy

    base = json.loads(battery["out_json"].read_text())
    oracle_path = tmp_path / "oracle.csv"
    _write_oracle_csv(oracle_path, de00=0.0)

    # b1 with BOTH blocks: stored-wfit "ee" at +5.0 (drift-inflated) and
    # "ee_refit_w" at +1.0 (the fair reference).
    b1 = copy.deepcopy(base)
    for row in b1["ee"]["grid"]:
        row["ee_ciede2000"] += 5.0
    b1["ee_refit_w"] = copy.deepcopy(base["ee"])
    for row in b1["ee_refit_w"]["grid"]:
        row["ee_ciede2000"] += 1.0
    b1_path = tmp_path / "b1_refit.json"
    b1_path.write_text(json.dumps(b1))

    result = eb.evaluate(
        str(battery["ckpt_path"]), str(battery["img_dir"]),
        str(tmp_path / "eval_refit_ref.json"), n_images=N_IMAGES, batch=4,
        b1_json=str(b1_path), oracle_csv=str(oracle_path), use_lpips=False,
    )
    block = result["oracle_gap_closed"]
    assert block["stale_w_reference"] is False
    # refit_w=False → the ckpt has no ee_refit_w grid to feed the numerator;
    # the fallback to the stored-wfit "ee" grid is recorded explicitly.
    assert block["ckpt_ee_block"] == "ee"
    grid_map = eb._de00_map_from_rows(result["ee"]["grid"])
    for factor in FACTORS:
        shared = [round(m, 6) for m in ORACLE_GRIDS[factor]]
        mean_ck = sum(grid_map[factor][m] for m in shared) / len(shared)
        # ee_b1 comes from the +1.0 refit block, NOT the +5.0 stored-wfit one.
        assert block["per_factor"][factor]["ee_b1"] == pytest.approx(mean_ck + 1.0)

    # Same b1 without the refit block → falls back to "ee", marked stale.
    del b1["ee_refit_w"]
    b1_path.write_text(json.dumps(b1))
    result2 = eb.evaluate(
        str(battery["ckpt_path"]), str(battery["img_dir"]),
        str(tmp_path / "eval_stale_ref.json"), n_images=N_IMAGES, batch=4,
        b1_json=str(b1_path), oracle_csv=str(oracle_path), use_lpips=False,
    )
    block2 = result2["oracle_gap_closed"]
    assert block2["stale_w_reference"] is True
    assert block2["ckpt_ee_block"] == "ee"
    for factor in FACTORS:
        shared = [round(m, 6) for m in ORACLE_GRIDS[factor]]
        mean_ck = sum(grid_map[factor][m] for m in shared) / len(shared)
        assert block2["per_factor"][factor]["ee_b1"] == pytest.approx(mean_ck + 5.0)


# ---------------------------------------------------------------------------
# Gap-closed numerator shares the b1 reference's W convention
# ---------------------------------------------------------------------------


def test_gap_closed_numerator_refit_self_consistency(tmp_path: Path) -> None:
    """A drifted analytic ('none') b1 checkpoint evaluated against ITS OWN
    --refit-w JSON must close exactly 0% of its own gap: the numerator uses
    the ckpt's ee_refit_w grid when the reference is refit-W, so the ckpt's
    own latent drift is not silently counted against it."""
    img_dir = tmp_path / "images"
    _write_images(img_dir, N_IMAGES)
    ckpt_path = tmp_path / "ckpt.pt"
    _drifted_ckpt(ckpt_path)
    oracle_path = tmp_path / "oracle.csv"
    _write_oracle_csv(oracle_path, de00=0.0)

    b1_json = tmp_path / "b1.json"
    eb.evaluate(
        str(ckpt_path), str(img_dir), str(b1_json),
        n_images=N_IMAGES, batch=4, use_lpips=False, refit_w=True,
    )
    result = eb.evaluate(
        str(ckpt_path), str(img_dir), str(tmp_path / "self.json"),
        n_images=N_IMAGES, batch=4, use_lpips=False, refit_w=True,
        b1_json=str(b1_json), oracle_csv=str(oracle_path),
    )
    block = result["oracle_gap_closed"]
    assert block["stale_w_reference"] is False
    assert block["ckpt_ee_block"] == "ee_refit_w"
    assert set(block["per_factor"]) == set(FACTORS)
    for g in block["per_factor"].values():
        assert g["ee_ckpt"] == pytest.approx(g["ee_b1"])
        assert g["gap_closed"] == pytest.approx(0.0, abs=1e-9)


def test_gap_closed_mixed_convention_warns_and_refit_w_resolves(
    battery, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Analytic ckpt + refit-W b1 reference WITHOUT --refit-w: the numerator
    falls back to the stored-wfit grid (recorded) and the CLI warns about the
    mixed W conventions; adding --refit-w switches the numerator and silences
    the warning."""
    import copy

    base = json.loads(battery["out_json"].read_text())
    b1 = copy.deepcopy(base)
    b1["ee_refit_w"] = copy.deepcopy(base["ee"])
    b1_path = tmp_path / "b1.json"
    b1_path.write_text(json.dumps(b1))
    oracle_path = tmp_path / "oracle.csv"
    _write_oracle_csv(oracle_path, de00=0.0)

    common = [
        "--ckpt", str(battery["ckpt_path"]),
        "--images", str(battery["img_dir"]),
        "--n-images", str(N_IMAGES), "--batch", "4", "--no-lpips",
        "--b1-json", str(b1_path), "--oracle-csv", str(oracle_path),
    ]
    out_mixed = tmp_path / "mixed.json"
    eb.main([*common, "--out", str(out_mixed)])
    data = json.loads(out_mixed.read_text())
    assert data["oracle_gap_closed"]["stale_w_reference"] is False
    assert data["oracle_gap_closed"]["ckpt_ee_block"] == "ee"
    assert "mixed W conventions" in capsys.readouterr().out

    out_refit = tmp_path / "refit.json"
    eb.main([*common, "--out", str(out_refit), "--refit-w"])
    data2 = json.loads(out_refit.read_text())
    assert data2["oracle_gap_closed"]["ckpt_ee_block"] == "ee_refit_w"
    assert "mixed W conventions" not in capsys.readouterr().out


def test_gap_closed_lie_ckpt_keeps_main_ee_numerator(tmp_path: Path) -> None:
    """'lie'/'conv' checkpoints: the co-trained operator IS the object under
    evaluation, so the numerator stays the main "ee" grid even when both
    refit_w=True and the b1 reference is the refit-W block."""
    img_dir = tmp_path / "images"
    _write_images(img_dir, 4)
    ckpt_path = tmp_path / "ckpt.pt"
    _make_ckpt(ckpt_path, operator_kind="lie", condition="p2_lie")
    oracle_path = tmp_path / "oracle.csv"
    _write_oracle_csv(oracle_path, de00=0.0)

    b1_json = tmp_path / "b1.json"
    eb.evaluate(
        str(ckpt_path), str(img_dir), str(b1_json),
        n_images=4, batch=4, use_lpips=False, refit_w=True,
    )
    result = eb.evaluate(
        str(ckpt_path), str(img_dir), str(tmp_path / "eval.json"),
        n_images=4, batch=4, use_lpips=False, refit_w=True,
        b1_json=str(b1_json), oracle_csv=str(oracle_path),
    )
    block = result["oracle_gap_closed"]
    assert block["stale_w_reference"] is False
    assert block["ckpt_ee_block"] == "ee"
    grid_map = eb._de00_map_from_rows(result["ee"]["grid"])
    for factor in FACTORS:
        shared = [round(m, 6) for m in ORACLE_GRIDS[factor]]
        mean_ck = sum(grid_map[factor][m] for m in shared) / len(shared)
        assert block["per_factor"][factor]["ee_ckpt"] == pytest.approx(mean_ck)


def test_gap_closed_analytic_ckpt_keeps_stored_wfit_numerator(tmp_path: Path) -> None:
    """'analytic' checkpoints: the stored wfit IS the shipped operator (training
    optimized the AE for exactly that (M, m)), so the numerator stays the main
    "ee" grid — substituting the refit grid would score a different operator
    than the artifact under evaluation and understate analytic conditions
    (measured on wave 1: p1 refit 8.4 vs shipped 4.6 dE00 brightness)."""
    img_dir = tmp_path / "images"
    _write_images(img_dir, 4)
    ckpt_path = tmp_path / "ckpt.pt"
    _make_ckpt(ckpt_path, operator_kind="analytic", condition="p1_analytic")
    oracle_path = tmp_path / "oracle.csv"
    _write_oracle_csv(oracle_path, de00=0.0)

    b1_json = tmp_path / "b1.json"
    eb.evaluate(
        str(b1_json_ckpt := str(ckpt_path)), str(img_dir), str(b1_json),
        n_images=4, batch=4, use_lpips=False, refit_w=True,
    )
    result = eb.evaluate(
        str(ckpt_path), str(img_dir), str(tmp_path / "eval.json"),
        n_images=4, batch=4, use_lpips=False, refit_w=True,
        b1_json=str(b1_json), oracle_csv=str(oracle_path),
    )
    block = result["oracle_gap_closed"]
    assert block["stale_w_reference"] is False
    assert block["ckpt_ee_block"] == "ee"


# ---------------------------------------------------------------------------
# Learned-operator checkpoints + the cross-operator audit
# ---------------------------------------------------------------------------


def test_lie_checkpoint_cross_operator_audit(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    _write_images(img_dir, 4)
    ckpt_path = tmp_path / "ckpt.pt"
    _make_ckpt(ckpt_path, operator_kind="lie", condition="p2_lie")

    result = eb.evaluate(
        str(ckpt_path), str(img_dir), str(tmp_path / "eval.json"),
        n_images=4, batch=4, use_lpips=False,
    )
    assert result["ckpt"]["operator_kind"] == "lie"
    cross = result["collapse"]["cross_operator"]
    # Learned operator → the audit is computed with the FROZEN analytic
    # operator from the ckpt's wfit, independently of the main battery.
    assert cross["identical_to_main"] is False
    assert set(cross["per_factor"]) == set(FACTORS)
    for stats in cross["per_factor"].values():
        assert math.isfinite(stats["ee_l2"]) and math.isfinite(stats["ee_ciede2000"])
    for row in result["ee"]["grid"]:
        assert math.isfinite(row["ee_l2"]) and math.isfinite(row["ee_ciede2000"])


def test_conv_operator_state_roundtrip(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "ckpt.pt"
    _make_ckpt(ckpt_path, operator_kind="conv", condition="p3_conv")
    ckpt = eb.load_run_checkpoint(str(ckpt_path))
    op = eb._load_operator(ckpt, "cpu")

    gen = torch.Generator().manual_seed(0)
    z = torch.randn(2, 4, 8, 8, generator=gen)
    # Identity at phi = 0 survives the state-dict round trip (SPEC2 conv op).
    assert torch.equal(op(z, torch.zeros(4)), z)
    assert not any(p.requires_grad for p in op.parameters())


def test_operator_state_missing_raises(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "ckpt.pt"
    ckpt = _make_ckpt(ckpt_path, operator_kind="none")
    ckpt["operator_kind"] = "lie"  # claims a learned operator, has none
    torch.save(ckpt, ckpt_path)
    loaded = eb.load_run_checkpoint(str(ckpt_path))
    with pytest.raises(ValueError, match="operator=None"):
        eb._make_apply_op(loaded, "cpu")


def test_checkpoint_missing_keys_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pt"
    torch.save({"vae": {}, "config": {}}, bad)
    with pytest.raises(KeyError, match="run-checkpoint format"):
        eb.load_run_checkpoint(str(bad))


# ---------------------------------------------------------------------------
# rFID hook (compute_fid monkeypatched — offline) + CLI
# ---------------------------------------------------------------------------


def test_rfid_hook_monkeypatched(battery, tmp_path: Path,
                                 monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_compute_fid(dir_a: str, dir_b: str, mode: str = "clean") -> float:
        calls.append((dir_a, dir_b))
        return 7.5

    # fid.rfid resolves compute_fid at module level (documented there).
    monkeypatch.setattr(fid_mod, "compute_fid", fake_compute_fid)

    out_json = tmp_path / "eval_rfid.json"
    result = eb.evaluate(
        str(battery["ckpt_path"]), str(battery["img_dir"]), str(out_json),
        n_images=4, rfid_n=3, batch=4, use_lpips=False,
    )
    assert result["reconstruction"]["rfid"] == 7.5
    assert len(calls) == 1
    rfid_root = tmp_path / "eval_rfid_rfid"
    assert (rfid_root / "recon").is_dir() and (rfid_root / "real").is_dir()
    assert len(list((rfid_root / "recon").glob("*.png"))) == 3


def test_cli_main(battery, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    out_json = tmp_path / "cli_eval.json"
    eb.main([
        "--ckpt", str(battery["ckpt_path"]),
        "--images", str(battery["img_dir"]),
        "--out", str(out_json),
        "--n-images", "4",
        "--batch", "4",
        "--no-lpips",
    ])
    assert out_json.is_file()
    data = json.loads(out_json.read_text())
    assert data["ckpt"]["condition"] == "b1"
    out = capsys.readouterr().out
    assert "recon: PSNR" in out
    assert "swapped-decoder: skipped" in out
