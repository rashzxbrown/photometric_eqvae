"""Generate the quantitative paper figures from the committed logs.

Produces, into this directory (paper/figures/):

- fig_convergence.pdf — DiT-S/2 and DiT-B/2 FID-5K convergence curves
  (b1 baseline vs p1_analytic tokenizer) from logs/fid_*.jsonl.
- fig_grids.pdf — the seed-paired 100K-step DiT-B sample grids
  (logs/samples_b1_ditb_100k.png vs samples_p1_ditb_100k.png).

Usage:
    cd <repo root> && uv run python paper/figures/make_figures.py

Data notes (mirror these in the captions):
- FID-5K, unconditional, vs the 41K-image OpenImages val proxy; single seed
  per (tokenizer, scale) pair.
- DiT-S logs contain retro-scored 10-40K lines appended AFTER the 50-100K
  lines ("retro": true); curves are merged by sorting on step (last entry
  wins on duplicates).
- Points below 30K steps are drawn at reduced alpha: early-training FID at
  10-20K is high-variance/uninformative and the caption should say so.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
LOGS = REPO / "logs"

COLOR_B1 = "#2a78d6"  # baseline (blue) — fixed assignment in every figure
COLOR_P1 = "#eb6834"  # photometric-equivariant (orange)
EARLY_CUTOFF = 30_000  # steps below this drawn at reduced alpha
EARLY_ALPHA = 0.40

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
    }
)


def load_fid(path: Path) -> tuple[list[int], list[float]]:
    """Read a fid_*.jsonl file -> (steps, fids), merged and sorted by step.

    Duplicate steps are merged with the LAST line winning (the DiT-S logs
    have retro-scored 10-40K entries appended after the 50-100K entries).
    """
    by_step: dict[int, float] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("level") != "fid":
                continue
            by_step[int(rec["step"])] = float(rec["fid"])
    steps = sorted(by_step)
    return steps, [by_step[s] for s in steps]


def plot_series(ax, steps, fids, color, label):
    """One curve, with the <30K segment at reduced alpha (early-FID caveat)."""
    xs = [s / 1000.0 for s in steps]
    # Split at the cutoff; include the first >=cutoff point in the early
    # segment so the line is visually continuous.
    n_early = sum(1 for s in steps if s < EARLY_CUTOFF)
    kw = dict(color=color, lw=2, marker="o", ms=3, solid_capstyle="round")
    if n_early:
        ax.plot(xs[: n_early + 1], fids[: n_early + 1], alpha=EARLY_ALPHA, **kw)
    ax.plot(xs[n_early:], fids[n_early:], **kw)
    return xs, fids, label


def end_label(ax, series, dy_frac):
    """Direct label at the right end of a curve, nudged by dy_frac (log units)."""
    xs, fids, label = series
    x, y = xs[-1], fids[-1]
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(5, dy_frac),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=8.5,
        color=series_color(label),
        clip_on=False,
        annotation_clip=False,
    )


def series_color(label: str) -> str:
    return COLOR_P1 if label.startswith("p1") else COLOR_B1


def style_axis(ax, title):
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, which="major", ls=":", lw=0.5, color="0.75")
    ax.grid(False, which="minor")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("training step ($\\times 10^3$)")
    ticks = [50, 100, 200, 400]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="minor", length=0)


def fig_convergence(out: Path) -> None:
    data = {
        name: load_fid(LOGS / f"{name}.jsonl")
        for name in ("fid_b1", "fid_p1", "fid_b1_ditb", "fid_p1_ditb",
                     "fid_b1_ditb_s1", "fid_p1_ditb_s1")
    }
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)

    ax = axes[0]
    s_b1 = plot_series(ax, *data["fid_b1"], COLOR_B1, "b1 (baseline)")
    s_p1 = plot_series(ax, *data["fid_p1"], COLOR_P1, "p1 (equivariant)")
    style_axis(ax, "DiT-S/2 (33M)")
    # Endpoints are 77.9 vs 77.6 — separate the direct labels vertically.
    end_label(ax, s_b1, 6)
    end_label(ax, s_p1, -6)

    ax = axes[1]
    s_b1 = plot_series(ax, *data["fid_b1_ditb"], COLOR_B1, "b1 (baseline)")
    s_p1 = plot_series(ax, *data["fid_p1_ditb"], COLOR_P1, "p1 (equivariant)")
    # Seed-1 repeats: same hue per arm (color follows the entity), dashed and
    # lighter so seed variability reads as texture, not as new series.
    for key, color in (("fid_b1_ditb_s1", COLOR_B1), ("fid_p1_ditb_s1", COLOR_P1)):
        steps, fids = data[key]
        ax.plot([s / 1000 for s in steps], fids, color=color, linewidth=1.2,
                linestyle="--", alpha=0.45, zorder=1)
    ax.text(0.97, 0.86, "dashed: seed 1", transform=ax.transAxes, fontsize=7,
            ha="right", color="#666666")
    style_axis(ax, "DiT-B/2 (130M), 2 seeds")
    end_label(ax, s_b1, 6)
    end_label(ax, s_p1, -6)

    for ax in axes:
        ax.set_xlim(5, 128)  # room on the right for the direct end-labels
        ax.set_xticks([25, 50, 75, 100])
        ax.set_ylim(42, 500)  # keep the 50 tick and the end-labels in frame

    fig.supylabel("FID-5K (vs OpenImages val)", fontsize=9, x=0.015)
    fig.tight_layout(rect=(0.03, 0, 1, 1))
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


def fig_grids(out: Path) -> None:
    imgs = [
        (mpimg.imread(LOGS / "samples_b1_ditb_100k.png"), "baseline (b1)"),
        (
            mpimg.imread(LOGS / "samples_p1_ditb_100k.png"),
            "photometric-equivariant (p1)",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.25))
    for ax, (img, title) in zip(axes, imgs):
        ax.imshow(img, interpolation="lanczos")
        ax.set_title(title)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    fig_convergence(HERE / "fig_convergence.pdf")
    fig_grids(HERE / "fig_grids.pdf")
