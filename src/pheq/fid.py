"""FID wrapper around clean-fid (SPEC2.md "src/pheq/fid.py").

rFID (reconstruction FID) is a Tier-1 reconstruction metric of
docs/research-plan.md §5 ("rFID/PSNR/LPIPS"); the custom-stats helpers cache
reference Inception statistics so DiT eval does not re-featurize the reference
set on every FID computation (SPEC2 fid.py section).

Care points (binding, SPEC2):

- ALL cleanfid imports are lazy (inside functions), so ``pheq.fid`` imports —
  and the offline test suite runs — without cleanfid ever being touched.
- :func:`fid_available` checks BOTH that cleanfid imports AND that the
  torchscript InceptionV3 weights are already on disk, WITHOUT triggering the
  download (the path is computed by mirroring cleanfid's convention, never by
  instantiating cleanfid's feature extractor).
- :func:`rfid` preprocesses the "real" side with EXACTLY the same
  resize/center-crop pipeline as :class:`pheq.data.ImageFolderDataset`
  (imported lazily): rFID is measured against the ``size``² *processed*
  originals, not the raw files on disk.
"""

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import nn

__all__ = [
    "fid_available",
    "compute_fid",
    "make_custom_stats",
    "fid_to_stats",
    "rfid",
]

#: Filename of the torchscript InceptionV3 checkpoint used by clean-fid's
#: "clean" and "legacy_tensorflow" modes (basename of
#: ``cleanfid.downloads_helper.inception_url``).
_INCEPTION_WEIGHT_NAME = "inception-2015-12-05.pt"


def _inception_weight_path() -> Path:
    """Where clean-fid stores/looks for the torchscript InceptionV3 weights.

    Mirrors ``cleanfid.features.feature_extractor``: the containing folder is
    ``"./"`` on Windows and ``"/tmp"`` otherwise, joined with
    ``inception-2015-12-05.pt``. Computed WITHOUT importing cleanfid (pure
    path logic) so calling this can never trigger a network download; a test
    cross-checks the filename against the installed cleanfid's download URL.
    """
    folder = "./" if platform.system() == "Windows" else "/tmp"
    return Path(folder) / _INCEPTION_WEIGHT_NAME


def fid_available() -> bool:
    """True iff cleanfid imports AND its Inception weights are already on disk.

    Never downloads anything: the import touches no weights, and weight
    presence is checked against the mirrored path of
    :func:`_inception_weight_path`. Callers and tests use this to skip
    FID-dependent paths gracefully (SPEC2 test-suite ground rules).
    """
    try:
        import cleanfid  # noqa: F401  (lazy per SPEC2)
        import cleanfid.fid  # noqa: F401
    except Exception:
        return False
    try:
        return _inception_weight_path().is_file()
    except OSError:
        return False



def _patched_frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    """Frechet distance compatible with scipy >= 1.18.

    scipy 1.18 removed ``sqrtm``'s ``disp`` keyword; cleanfid 0.1.35 still
    passes ``disp=False`` (cleanfid/fid.py line 46), so every FID computation
    raises TypeError. Standard FID formula, numerically identical to
    cleanfid's, minus the removed kwarg; installed over
    ``cleanfid.fid.frechet_distance`` by :func:`_ensure_cleanfid_patched`.
    """
    import numpy as np
    from scipy import linalg

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def _ensure_cleanfid_patched() -> None:
    """Install the scipy>=1.18-compatible frechet_distance into cleanfid (idempotent)."""
    import cleanfid.fid as _cleanfid_fid  # lazy (SPEC2)

    if getattr(_cleanfid_fid.frechet_distance, "__name__", "") != "_patched_frechet_distance":
        _cleanfid_fid.frechet_distance = _patched_frechet_distance


def _cleanfid_device() -> torch.device:
    """Device for cleanfid's feature extractor.

    cleanfid defaults to ``torch.device("cuda")`` unconditionally, which
    crashes on CPU-only/MPS machines — so the wrappers always pass a device
    explicitly (CUDA when available, else CPU).
    """
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def compute_fid(dir_a: str, dir_b: str, mode: str = "clean") -> float:
    """Thin wrapper: FID between two image directories via cleanfid.

    Args:
        dir_a: first image directory.
        dir_b: second image directory.
        mode: cleanfid mode ('clean' default; also 'legacy_pytorch',
            'legacy_tensorflow').

    Returns:
        The FID score as a python float.

    Notes:
        cleanfid import is lazy (SPEC2). Device is chosen by
        :func:`_cleanfid_device`; dataloader workers are 12 on CUDA and 0 on
        CPU (worker processes buy nothing there and slow small folders down).
    """
    from cleanfid import fid as _fid  # lazy per SPEC2
    _ensure_cleanfid_patched()

    device = _cleanfid_device()
    score = _fid.compute_fid(
        str(dir_a),
        str(dir_b),
        mode=mode,
        device=device,
        # num_workers=0: Python 3.14 switched POSIX multiprocessing to
        # 'forkserver', which requires picklable worker args — cleanfid passes
        # a local closure (make_resizer.<locals>.func) and every FID call died
        # with "Can't pickle local object" (observed: all in-training FID
        # hooks on the H200 cluster, py3.14). Featurization is GPU-bound, so
        # single-process loading costs little.
        num_workers=0,
    )
    return float(score)


def make_custom_stats(name: str, image_dir: str) -> None:
    """Precompute cleanfid custom reference statistics for ``image_dir``.

    Featurizes the directory once and stores Inception statistics under
    cleanfid's stats folder keyed by ``name`` (mode 'clean', split 'custom'),
    so subsequent :func:`fid_to_stats` calls never re-featurize the reference
    set (the point of this helper per SPEC2 fid.py).

    Idempotent: if stats named ``name`` already exist, returns without
    recomputing (cleanfid itself would raise; caching-friendly behavior is
    deliberate here — use ``cleanfid.fid.remove_custom_stats`` to force a
    rebuild).
    """
    from cleanfid import fid as _fid  # lazy per SPEC2
    _ensure_cleanfid_patched()

    if _fid.test_stats_exists(name, mode="clean"):
        return
    device = _cleanfid_device()
    _fid.make_custom_stats(name, str(image_dir), mode="clean", device=device)


def fid_to_stats(gen_dir: str, name: str) -> float:
    """FID of an image directory against precomputed custom stats ``name``.

    ``name`` must have been registered with :func:`make_custom_stats`
    (mode 'clean', split 'custom').

    Returns:
        The FID score as a python float.
    """
    from cleanfid import fid as _fid  # lazy per SPEC2
    _ensure_cleanfid_patched()

    device = _cleanfid_device()
    score = _fid.compute_fid(
        str(gen_dir),
        dataset_name=name,
        mode="clean",
        dataset_split="custom",
        device=device,
        # num_workers=0: Python 3.14 switched POSIX multiprocessing to
        # 'forkserver', which requires picklable worker args — cleanfid passes
        # a local closure (make_resizer.<locals>.func) and every FID call died
        # with "Can't pickle local object" (observed: all in-training FID
        # hooks on the H200 cluster, py3.14). Featurization is GPU-bound, so
        # single-process loading costs little.
        num_workers=0,
    )
    return float(score)


# ---------------------------------------------------------------------------
# rFID: decode-reconstruct a folder and FID recon vs processed originals
# ---------------------------------------------------------------------------


def _save_png(img: torch.Tensor, path: Path) -> None:
    """Save a ``(3, H, W)`` float tensor as an 8-bit RGB PNG.

    Values are clamped to [0, 1] then quantized with round-to-nearest to
    uint8, so a subsequent :func:`_load_png` recovers each in-range value to
    within 1/(2*255) < 1/255 (the PNG round-trip guarantee tested in
    tests/test_fid.py per SPEC2).
    """
    from PIL import Image  # lazy: keep pheq.fid import light

    arr = (
        img.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )
    Image.fromarray(arr, mode="RGB").save(path)


def _load_png(path: Path) -> torch.Tensor:
    """Load an RGB PNG as a ``(3, H, W)`` float32 tensor in [0, 1]."""
    import numpy as np
    from PIL import Image  # lazy: keep pheq.fid import light

    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _resolve_vae(vae_or_ckpt: Any, device: str) -> nn.Module:
    """Turn ``vae_or_ckpt`` into an eval-mode autoencoder module on ``device``.

    Accepts:

    - an autoencoder module (SD-VAE wrapper from :func:`pheq.vae.load_sd_vae`,
      or a toy AE) — moved to ``device`` and set to eval;
    - a ``str``/``Path`` to a run checkpoint (SPEC2 run-checkpoint format:
      ``{"vae": state_dict, ..., "config": dict}``) — loaded with
      ``torch.load(map_location='cpu', weights_only=False)`` (trusted local
      artifact produced by pheq.train_ae);
    - an already-loaded run-checkpoint ``dict``.

    Checkpoint architecture resolution (SPEC2 leaves this implicit; resolved
    here): ``config["vae"]`` selects 'toy' (:class:`pheq.vae.ToyConvAE`, with
    ``channels``/``hidden`` inferred from the state-dict shapes) vs anything
    else → the pretrained SD-VAE via :func:`pheq.vae.load_sd_vae` with the
    checkpoint's fine-tuned weights loaded on top.
    """
    if isinstance(vae_or_ckpt, nn.Module):
        return vae_or_ckpt.to(device).eval()

    if isinstance(vae_or_ckpt, (str, Path)):
        ckpt = torch.load(str(vae_or_ckpt), map_location="cpu", weights_only=False)
    elif isinstance(vae_or_ckpt, dict):
        ckpt = vae_or_ckpt
    else:  # pragma: no cover - defensive
        raise TypeError(
            f"vae_or_ckpt must be an nn.Module, path, or run-checkpoint dict; "
            f"got {type(vae_or_ckpt).__name__}"
        )

    if "vae" not in ckpt:
        raise KeyError(
            "run checkpoint missing 'vae' state_dict (SPEC2 run-checkpoint format)"
        )
    state = ckpt["vae"]
    kind = str(ckpt.get("config", {}).get("vae", "sd"))
    if kind == "toy":
        from pheq.vae import ToyConvAE

        # Infer sizes from the state dict so non-default toys round-trip:
        # encoder.0: Conv2d(3, hidden, 3); encoder.2: Conv2d(hidden, C, 2, s2).
        hidden = int(state["encoder.0.weight"].shape[0])
        channels = int(state["encoder.2.weight"].shape[0])
        vae: nn.Module = ToyConvAE(channels=channels, hidden=hidden)
    else:
        from pheq.vae import load_sd_vae

        vae = load_sd_vae(device)
    vae.load_state_dict(state)
    return vae.to(device).eval()


def rfid(
    vae_or_ckpt: Any,
    image_dir: str,
    out_dir: str,
    n: int = 1024,
    device: str = "cpu",
    batch: int = 8,
    size: int = 256,
) -> float:
    """Reconstruction FID: decode-reconstruct ``n`` images and FID vs originals.

    Pipeline (SPEC2 fid.py; plan §5 Tier-1 "rFID"):

    1. Load up to ``n`` images through :class:`pheq.data.ImageFolderDataset`
       (deterministic sorted order; resize shorter side to ``size``, center
       crop ``size``², float32 [0,1]) — the "real" side is resaved from these
       PROCESSED tensors, so rFID here is vs the ``size``² processed
       originals, not the raw files (documented per SPEC2).
    2. In chunks of ``batch``: encode to posterior moments, decode the
       posterior MEAN (deterministic reconstruction — no sampling noise, so
       the metric isolates the reconstruction path), save 8-bit PNGs to
       ``out_dir/recon`` and the processed originals to ``out_dir/real``
       (values clamped to [0,1] at save time; decode is pre-clip per plan
       §3.1, PNG is where quantization/clipping happens).
    3. Return :func:`compute_fid` of the two directories (module-level
       indirection so tests can monkeypatch ``pheq.fid.compute_fid`` and stay
       offline).

    Args:
        vae_or_ckpt: autoencoder module, run-checkpoint path, or loaded
            run-checkpoint dict (see :func:`_resolve_vae`).
        image_dir: source image directory (recursed by ImageFolderDataset).
        out_dir: output root; ``out_dir/recon`` and ``out_dir/real`` are
            recreated from scratch (stale PNGs from a previous run would
            silently pollute the FID).
        n: number of images (capped at the dataset size).
        device: torch device string for encode/decode.
        batch: encode/decode chunk size.
        size: ImageFolderDataset image size (default 256, the sprint eval
            resolution; keyword added beyond the SPEC2 arg list so toy tests
            can run tiny — callers using SPEC2's positional args are
            unaffected).

    Returns:
        The FID between ``out_dir/real`` and ``out_dir/recon`` as a float.
    """
    # Lazy import: pheq.data is a sibling v2 module; importing it here (not at
    # module top) keeps pheq.fid usable/testable independently of it.
    from pheq.data import ImageFolderDataset

    dataset = ImageFolderDataset(image_dir, size=size)
    n_eff = min(int(n), len(dataset))
    if n_eff == 0:
        raise ValueError(f"no images found under {image_dir!r}")

    vae = _resolve_vae(vae_or_ckpt, device)

    out = Path(out_dir)
    recon_dir = out / "recon"
    real_dir = out / "real"
    for d in (recon_dir, real_dir):
        if d.exists():
            shutil.rmtree(d)  # stale PNGs would silently pollute the FID
        d.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for start in range(0, n_eff, batch):
            idx = range(start, min(start + batch, n_eff))
            x = torch.stack([dataset[i][0] for i in idx]).to(device)
            mu, _sigma = vae.encode_moments(x)
            recon = vae.decode_latents(mu)
            for j, i in enumerate(idx):
                _save_png(x[j], real_dir / f"{i:06d}.png")
                _save_png(recon[j], recon_dir / f"{i:06d}.png")

    return compute_fid(str(real_dir), str(recon_dir))
