# paper/figures — provenance notes for captions

Regenerate with: `cd <repo root> && uv run python paper/figures/make_figures.py`

## fig_convergence.pdf
- FID-5K (clean-fid), unconditional sampling, scored against the 41K-image
  OpenImages validation proxy; single seed per (tokenizer, scale) pair.
- Source data: `logs/fid_b1.jsonl`, `logs/fid_p1.jsonl` (DiT-S/2),
  `logs/fid_b1_ditb.jsonl`, `logs/fid_p1_ditb.jsonl` (DiT-B/2). DiT-S 10-40K
  points are retro-scored (`"retro": true` lines); curves merge duplicates by
  step (last line wins).
- Points below 30K steps are drawn at reduced alpha: 10-20K FID is
  high-variance/uninformative early-training noise. The caption should carry
  this caveat rather than the figure.

## fig_grids.pdf
- `logs/samples_b1_ditb_100k.png` vs `logs/samples_p1_ditb_100k.png`:
  DiT-B/2 samples at 100K steps. The two grids are SEED-PAIRED — the same
  sampling noise per cell for both tokenizers — so cellwise content matches
  and differences reflect the tokenizer, not the sampler draw. Note this in
  the caption.

## fig1_editing (generated on the cluster)
- `scripts/make_fig1_editing.py` — qualitative closed-form latent
  color-editing grid; needs a real AE run checkpoint + real images, so it is
  run on the cluster and its PNG/PDF outputs are copied here.
