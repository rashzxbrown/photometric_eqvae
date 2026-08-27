# photometric-eqvae

**photometric equivariance for diffusion tokenizers** — extending
EQ-VAE-style latent equivariance beyond spatial transforms via analytic and learned latent
color operators.

## Layout

```
src/pheq/
  color.py         # canonicalized photometric family (Aff(3), fixed-anchor contrast)
  analytic.py      # latent→RGB W-fit + closed-form operator M_a = W⁺A_aW + (I−W⁺W)K
  lie_operator.py  # learned Lie-affine operator g_ψ (exact identity/composition)
  metrics.py       # EE battery: CIEDE2000, L2, hue histograms; ee_pix / ee_lat
  vae.py           # SD-VAE loader + toy AEs for offline testing
  oracle.py        # decoder-inversion oracle (RQ0)
  probes/          # Tier-0 CLI probes: fit_w, analytic_probe, oracle_probe
tests/             # algebra + reference-value tests (run offline, no downloads)
```

## Setup

```bash
uv sync
uv run pytest            # full offline test suite
```

## Tier-0 probes

```bash
# offline smoke run on the toy autoencoder:
uv run python -m pheq.probes.fit_w --vae toy
uv run python -m pheq.probes.analytic_probe --vae toy
uv run python -m pheq.probes.oracle_probe --vae toy

# real run (downloads stabilityai/sd-vae-ft-mse, ~335 MB):
uv run python -m pheq.probes.fit_w --vae sd --images <dir> --device mps
```
