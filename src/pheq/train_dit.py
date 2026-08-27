"""DiT proxy trainer on cached latent shards (SPEC2 train_dit.py).

Trains a DiT eps-predictor (pheq.dit) with the DDPM loss (pheq.diffusion) on
:class:`pheq.cache_latents.LatentShardDataset` — which REQUIRES the cache's
``stats.json``: per-checkpoint latent normalization is mandatory (plan §3.4 /
SPEC2 gap-review; :func:`train` refuses to start without it). Recipe per SPEC2:
AdamW (weight decay 0), NO warmup (constant lr — the DiT recipe), EMA
(decay 0.9999), no gradient clipping.

Engineering contract (mirrors SPEC2 train_ae, coded to that contract):

- ``out_dir/ckpt_latest.pt`` every ``save_every`` (500) steps AND on SIGTERM
  (SLURM preemption: a handler sets a stop flag; the loop saves at the next
  step boundary and returns cleanly, so the CLI exits 0). ``resume=True``
  auto-loads ckpt_latest. ``max_hours`` -> clean save + return before the
  wall clock.
- ``out_dir/monitor.jsonl``: loss every ``monitor_every`` (200) steps
  (level "train"), sampling-hook records (level "sample"), FID records
  (level "fid"), non-fatal problems (level "alert").
- Sampling hook every ``sample_every`` steps: EMA weights -> ddpm_sample a
  small grid, UN-normalize the latents with stats.json (``z*std + mean`` —
  the inverse of the cache normalization) BEFORE decoding with the run
  checkpoint's VAE (``vae_ckpt``, SPEC2 run-checkpoint format), save a PNG
  grid to ``out_dir/samples``. Skipped (with an alert line) if ``vae_ckpt``
  is not provided.
- FID hook every ``fid_every`` steps: sample ``n_fid`` latents (batched),
  un-normalize, decode, save PNGs to ``out_dir/fid/step_*``, score against
  ``fid_ref_stats`` — a directory path (-> ``pheq.fid.compute_fid``) or a
  cleanfid custom-stats name (-> ``pheq.fid.fid_to_stats``); SPEC2 says
  "compute_fid vs fid_ref_stats", resolved here to support both. FID errors
  are logged as alerts, never fatal (a broken FID must not kill a 24 h run).

DiT-checkpoint format (train_dit's own — distinct from the AE run
checkpoint): ``{"model", "ema", "opt", "step", "arch", "config",
"gen_state", "torch_rng"}``. Resume restores model/EMA/optimizer/step and the
training-noise generator; loader shuffle order is re-seeded from the resume
step (bit-exact RNG restore not required per the train_ae contract — step
count and loss continuity are).

CLI::

    uv run python -m pheq.train_dit --latents DIR --out DIR [--arch dit_s]
        [--steps N] [--batch N] [--lr F] [--device D] [--seed N]
        [--vae-ckpt PATH] [--sample-every N] [--fid-every N]
        [--fid-ref-stats NAME_OR_DIR] [--n-fid N] [--num-classes N]
        [--ema-decay F] [--max-hours H] [--workers N] [--no-resume]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader

from pheq import fid as fid_mod  # module ref so tests can monkeypatch pheq.fid.*
from pheq.cache_latents import LatentShardDataset, load_stats, resolve_vae
from pheq.data import make_loader
from pheq.diffusion import EMA, GaussianDiffusion
from pheq.dit import dit_b, dit_s, dit_tiny

# Private but stable within this repo: reusing rfid's PNG writer keeps the
# clamp/round-to-uint8 quantization convention identical across rFID, the
# sampling hook, and the FID hook (tests/test_fid.py pins it).
from pheq.fid import _save_png

__all__ = ["train", "main"]

_ARCHS = {"dit_s": dit_s, "dit_b": dit_b, "dit_tiny": dit_tiny}


def _append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON record per line (the monitor.jsonl convention)."""
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _cycle(loader: DataLoader) -> Iterator:
    """Endless epoch-cycling iterator over a DataLoader."""
    while True:
        yield from loader


def _save_ckpt(path: Path, payload: dict) -> None:
    """Atomic checkpoint write (tmp + rename): preemption-safe."""
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def train(
    latent_dir: str,
    out_dir: str,
    arch: str = "dit_s",
    steps: int = 100_000,
    batch: int = 256,
    lr: float = 1e-4,
    device: str = "cpu",
    resume: bool = True,
    ema_decay: float = 0.9999,
    sample_every: int = 10_000,
    fid_every: int | None = None,
    fid_ref_stats: str | None = None,
    n_fid: int = 5000,
    num_classes: int = 0,
    max_hours: float | None = None,
    *,
    vae_ckpt: str | None = None,
    seed: int = 0,
    workers: int = 0,
    timesteps: int = 1000,
    sample_steps: int = 250,
    n_sample: int = 16,
    monitor_every: int = 200,
    save_every: int = 500,
) -> Path:
    """Train a DiT on cached latents; return the final checkpoint path.

    Args (SPEC2 train_dit.py signature; keyword-only additions documented):
        latent_dir: cache dir from pheq.cache_latents — ``stats.json``
            REQUIRED (per-checkpoint normalization, plan §3.4; hard error).
        out_dir: run dir (ckpt_latest.pt, monitor.jsonl, samples/, fid/).
        arch: 'dit_s' | 'dit_tiny' (pheq.dit factories).
        steps: total optimizer steps (resume continues to this total).
        batch: batch size (loader drops last; dataset must hold >= batch).
        lr: constant AdamW lr (no warmup, wd 0 — DiT recipe).
        device: torch device string.
        resume: auto-load ``out_dir/ckpt_latest.pt`` if present.
        ema_decay: EMA decay (0.9999, the DiT recipe).
        sample_every: sampling-hook period in steps.
        fid_every: FID-hook period (None disables).
        fid_ref_stats: reference for the FID hook — an image directory OR a
            cleanfid custom-stats name (see module docstring).
        n_fid: latents sampled per FID evaluation.
        num_classes: label classes (0 -> unconditional; labels ignored).
        max_hours: clean save + return after this wall-clock budget.
        vae_ckpt: run-checkpoint path (SPEC2 format) whose VAE decodes the
            sampling/FID hooks; hooks are skipped (alert-logged) if None.
        seed: base seed for the training-noise generator, loader shuffling,
            and hook sampling.
        workers: DataLoader workers (default 0: cached-latent loading is
            randn + one affine — worker processes buy nothing and would
            duplicate shard caches).
        timesteps: DDPM T (default 1000 per SPEC2 diffusion.py).
        sample_steps: ancestral-sampler subsequence length (default 250).
        n_sample: images in the sampling-hook grid.
        monitor_every / save_every: monitor (200) and checkpoint (500)
            periods — SPEC2 defaults, overridable for tiny tests.

    Returns:
        Path to ``out_dir/ckpt_latest.pt`` (saved at exit in all cases:
        completion, SIGTERM, max_hours).
    """
    if arch not in _ARCHS:
        raise ValueError(f"unknown arch {arch!r}; choose from {sorted(_ARCHS)}")

    # Wall-clock budget starts NOW — before dataset scanning, model build and
    # the resume torch.load — so max_hours bounds the whole call, not just
    # the loop (setup can be minutes on a cold cluster node).
    t0 = time.monotonic()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "ckpt_latest.pt"
    monitor_path = out / "monitor.jsonl"

    # MANDATORY normalization stats (plan §3.4): refuse to run without them.
    # load_stats raises FileNotFoundError with the re-cache instruction;
    # LatentShardDataset re-reads the same file for its mean/std.
    stats = load_stats(latent_dir)
    dataset = LatentShardDataset(latent_dir)
    if len(dataset) < batch:
        raise ValueError(
            f"dataset has {len(dataset)} samples < batch {batch} "
            "(drop_last loader would yield no batches)"
        )
    c, h, w = dataset.latent_shape
    if h != w:
        raise ValueError(f"DiT requires square latents, got {(c, h, w)}")

    # Label/num_classes consistency (fail at startup, not mid-run): the
    # embedding table has num_classes + 1 rows with row num_classes reserved
    # for the learned null class, so a stray label == num_classes would
    # silently train the null row and labels beyond it would IndexError
    # mid-run. A label set that is a strict subset of range(num_classes)
    # (e.g. an all-zero cache built without --class-from-dir) leaves the
    # missing classes' embeddings at init noise — sampled anyway by the
    # hooks' y = arange % num_classes — so it is alert-logged, not fatal.
    if num_classes > 0:
        seen_labels: set[int] = set()
        for shard_path in dataset.shard_paths:
            labels = torch.load(shard_path, map_location="cpu")["label"]
            seen_labels.update(int(v) for v in labels.unique())
        if seen_labels and max(seen_labels) >= num_classes:
            raise ValueError(
                f"cached labels reach {max(seen_labels)} but num_classes="
                f"{num_classes} (valid labels: 0..{num_classes - 1}; row "
                f"{num_classes} is the reserved null class). Re-cache with "
                "--class-from-dir or fix --num-classes."
            )
        missing = set(range(num_classes)) - seen_labels
        if missing:
            _append_jsonl(monitor_path, {
                "level": "alert", "step": 0,
                "msg": (
                    f"cached labels cover only {sorted(seen_labels)} of "
                    f"range({num_classes}); classes {sorted(missing)} will "
                    "train no embedding but ARE sampled by the hooks "
                    "(cache built without --class-from-dir?)"
                ),
            })

    model = _ARCHS[arch](num_classes=num_classes, input_size=h, in_channels=c)
    model = model.to(device).train()
    diffusion = GaussianDiffusion(timesteps=timesteps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    ema = EMA(model, decay=ema_decay)
    gen = torch.Generator().manual_seed(seed)  # CPU gen: t/eps for the loss

    start_step = 0
    if resume and ckpt_path.is_file():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ck.get("arch") != arch:
            raise ValueError(
                f"resume arch mismatch: checkpoint {ck.get('arch')!r} != {arch!r}"
            )
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        if "gen_state" in ck:
            gen.set_state(ck["gen_state"])
        if "torch_rng" in ck:
            torch.set_rng_state(ck["torch_rng"])
        start_step = int(ck["step"])

    config = {
        "latent_dir": str(latent_dir), "out_dir": str(out_dir), "arch": arch,
        "steps": steps, "batch": batch, "lr": lr, "device": str(device),
        "ema_decay": ema_decay, "sample_every": sample_every,
        "fid_every": fid_every, "fid_ref_stats": fid_ref_stats, "n_fid": n_fid,
        "num_classes": num_classes, "vae_ckpt": vae_ckpt, "seed": seed,
        "timesteps": timesteps, "sample_steps": sample_steps,
        "latent_shape": [c, h, w], "stats_condition": stats.get("condition"),
        "stats_step": stats.get("step"),
    }

    def save(step: int) -> None:
        _save_ckpt(ckpt_path, {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "step": step,
            "arch": arch,
            "config": config,
            "gen_state": gen.get_state(),
            "torch_rng": torch.get_rng_state(),
        })

    # Un-normalization constants (inverse of the cache normalization): the
    # DiT models NORMALIZED latents; decoding must map back to VAE latent
    # space first (SPEC2 care point; plan §3.4).
    unnorm_mean = dataset.mean.to(device)  # (C, 1, 1)
    unnorm_std = dataset.std.to(device)
    _vae_cache: list = []  # lazily resolved decode VAE

    def _get_vae():
        if not _vae_cache:
            vae, _ = resolve_vae(vae_ckpt, device)
            _vae_cache.append(vae)
        return _vae_cache[0]

    def _sample_latents(n: int, hook_seed: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """EMA-weight ancestral sampling of n latents (+ labels if class-cond)."""
        sgen = torch.Generator().manual_seed(hook_seed)
        y = (
            torch.arange(n, device=device) % num_classes
            if num_classes > 0
            else None
        )
        with ema.swap(model):
            z = diffusion.ddpm_sample(
                model, (n, c, h, w), y=y, device=device,
                steps=min(sample_steps, timesteps), gen=sgen,
            )
        return z, y

    def _decode(z_norm: torch.Tensor) -> torch.Tensor:
        """Un-normalize (stats.json) then decode with the vae_ckpt's VAE."""
        z = z_norm * unnorm_std + unnorm_mean
        with torch.no_grad():
            return _get_vae().decode_latents(z)

    def sample_hook(step: int) -> None:
        if vae_ckpt is None:
            _append_jsonl(monitor_path, {
                "level": "alert", "step": step,
                "msg": "sampling hook skipped: no --vae-ckpt provided",
            })
            return
        from torchvision.utils import make_grid  # lazy: keep import light

        z, _y = _sample_latents(n_sample, seed + 7919 * step)
        imgs = _decode(z).clamp(0.0, 1.0)
        grid = make_grid(imgs, nrow=max(1, math.ceil(math.sqrt(n_sample))))
        sample_dir = out / "samples"
        sample_dir.mkdir(exist_ok=True)
        png = sample_dir / f"step_{step:07d}.png"
        _save_png(grid, png)
        _append_jsonl(monitor_path, {
            "level": "sample", "step": step, "n": n_sample, "path": str(png),
        })

    def _hook_should_abort() -> bool:
        """SIGTERM/max_hours check for long-running hooks (a full FID pass
        can outlast SLURM's TERM grace; abort so the loop saves promptly)."""
        return stop_flag["stop"] or (
            max_hours is not None
            and (time.monotonic() - t0) > max_hours * 3600.0
        )

    def fid_hook(step: int) -> None:
        if vae_ckpt is None or fid_ref_stats is None:
            _append_jsonl(monitor_path, {
                "level": "alert", "step": step,
                "msg": "fid hook skipped: needs --vae-ckpt and --fid-ref-stats",
            })
            return
        # The WHOLE hook (sampling/decode/PNG writing included, not just the
        # scoring) is guarded: a transient decode failure (e.g. CUDA OOM)
        # must never kill a 24 h run — the module contract is "FID errors
        # are logged as alerts, never fatal".
        try:
            gen_dir = out / "fid" / f"step_{step:07d}"
            gen_dir.mkdir(parents=True, exist_ok=True)
            chunk = max(1, min(batch, 64))
            written = 0
            while written < n_fid:
                if _hook_should_abort():
                    _append_jsonl(monitor_path, {
                        "level": "alert", "step": step,
                        "msg": (
                            "fid hook aborted (SIGTERM/max_hours) after "
                            f"{written}/{n_fid} samples"
                        ),
                    })
                    return
                nb = min(chunk, n_fid - written)
                z, _y = _sample_latents(nb, seed + 104_729 * step + written)
                imgs = _decode(z).clamp(0.0, 1.0)
                for j in range(nb):
                    _save_png(imgs[j], gen_dir / f"{written + j:06d}.png")
                written += nb
            if Path(fid_ref_stats).is_dir():
                score = fid_mod.compute_fid(str(gen_dir), str(fid_ref_stats))
            else:
                score = fid_mod.fid_to_stats(str(gen_dir), fid_ref_stats)
            _append_jsonl(monitor_path, {
                "level": "fid", "step": step, "fid": float(score), "n": n_fid,
            })
        except Exception as exc:  # never kill a 24h run over a broken FID
            _append_jsonl(monitor_path, {
                "level": "alert", "step": step, "msg": f"fid failed: {exc}",
            })

    # SIGTERM -> save at the next step boundary, return cleanly (CLI exits 0).
    # Signal handlers only work in the main thread; elsewhere (tests driving
    # train() from a worker thread) the flag simply never fires.
    stop_flag = {"stop": False}
    handler_installed = False
    prev_handler: object = None
    if threading.current_thread() is threading.main_thread():
        def _on_term(signum, frame):  # noqa: ANN001 - signal API
            stop_flag["stop"] = True

        prev_handler = signal.signal(signal.SIGTERM, _on_term)
        handler_installed = True

    # Loader shuffle re-seeded from the resume step (bit-exact sampler-state
    # restore not required per the train_ae contract).
    loader_gen = torch.Generator().manual_seed(seed + 1 + start_step)
    loader = make_loader(dataset, batch, loader_gen, workers=workers, shuffle=True)
    batches = _cycle(loader)

    if start_step > 0:
        # Segment marker: after a HARD crash monitor.jsonl may hold lines
        # from the dead run with steps beyond this checkpoint; consumers
        # should dedupe keeping the last occurrence after the final marker.
        _append_jsonl(monitor_path, {"level": "resume", "step": start_step})

    window: list[float] = []
    step = start_step
    try:
        while step < steps:
            if stop_flag["stop"]:
                _append_jsonl(monitor_path, {
                    "level": "alert", "step": step,
                    "msg": "SIGTERM received: checkpoint saved, exiting",
                })
                break
            if max_hours is not None and (time.monotonic() - t0) > max_hours * 3600.0:
                _append_jsonl(monitor_path, {
                    "level": "alert", "step": step,
                    "msg": f"max_hours={max_hours} reached: checkpoint saved, exiting",
                })
                break
            z, y = next(batches)
            z = z.to(device)
            y = y.to(device) if num_classes > 0 else None
            # bf16 autocast (same pattern as train_ae): fp32 DiT-S activations
            # at batch 256 need ~23 GB and OOM 24 GB cards (Oscar jobs
            # 4188703/4). SDPA + bf16 halves activation memory; MSE reduces
            # in fp32 under autocast's op policy.
            dev_type = torch.device(device).type
            with torch.autocast(device_type=dev_type, dtype=torch.bfloat16,
                                enabled=dev_type == "cuda"):
                loss = diffusion.training_loss(model, z, y, gen)
            if not torch.isfinite(loss):
                save(step)
                raise RuntimeError(f"non-finite loss at step {step + 1}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ema.update(model)
            step += 1
            window.append(float(loss.detach()))

            if step % monitor_every == 0:
                _append_jsonl(monitor_path, {
                    "level": "train", "step": step,
                    "loss": window[-1],
                    "loss_avg": sum(window) / len(window),
                    "elapsed_s": time.monotonic() - t0,
                })
                window.clear()
            if step % save_every == 0:
                save(step)
            if sample_every and step % sample_every == 0:
                sample_hook(step)
            if fid_every and step % fid_every == 0:
                fid_hook(step)
        save(step)
    finally:
        if handler_installed:
            # prev_handler is None when the old handler wasn't set from
            # Python; SIG_DFL is the correct restoration then.
            signal.signal(
                signal.SIGTERM,
                prev_handler if prev_handler is not None else signal.SIG_DFL,
            )
    return ckpt_path


def main(argv: list[str] | None = None) -> None:
    """CLI mirroring pheq.train_ae conventions (SPEC2)."""
    parser = argparse.ArgumentParser(
        description="Train a DiT proxy on cached latents (SPEC2 train_dit.py)."
    )
    parser.add_argument("--latents", type=str, required=True,
                        help="latent cache dir (stats.json REQUIRED, plan §3.4)")
    parser.add_argument("--out", type=str, required=True, help="run output dir")
    parser.add_argument("--arch", type=str, default="dit_s",
                        choices=sorted(_ARCHS))
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--sample-every", type=int, default=10_000)
    parser.add_argument("--fid-every", type=int, default=None)
    parser.add_argument("--fid-ref-stats", type=str, default=None,
                        help="reference image dir OR cleanfid custom-stats name")
    parser.add_argument("--n-fid", type=int, default=5000)
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument("--vae-ckpt", type=str, default=None,
                        help="run checkpoint whose VAE decodes sample/FID hooks")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sample-steps", type=int, default=250)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    final = train(
        args.latents,
        args.out,
        arch=args.arch,
        steps=args.steps,
        batch=args.batch,
        lr=args.lr,
        device=args.device,
        resume=not args.no_resume,
        ema_decay=args.ema_decay,
        sample_every=args.sample_every,
        fid_every=args.fid_every,
        fid_ref_stats=args.fid_ref_stats,
        n_fid=args.n_fid,
        num_classes=args.num_classes,
        max_hours=args.max_hours,
        vae_ckpt=args.vae_ckpt,
        seed=args.seed,
        workers=args.workers,
        timesteps=args.timesteps,
        sample_steps=args.sample_steps,
    )
    print(f"final checkpoint -> {final}")
    sys.exit(0)


if __name__ == "__main__":
    main()
