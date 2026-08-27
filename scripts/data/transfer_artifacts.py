"""Move run artifacts between clusters via a private Hugging Face Hub repo.

Why not git: GitHub rejects files >100 MB (the AE checkpoints are ~1 GB with
optimizer state), and committing binaries would permanently bloat the repo.
HF Hub has no practical size limit, both clusters already reach it, and a
private artifact repo is a head start on the eventual checkpoint release.

One-time: create a WRITE token at https://huggingface.co/settings/tokens and
export it on BOTH clusters:  export HF_TOKEN=hf_...

Upload (source cluster):
    .venv/bin/python scripts/data/transfer_artifacts.py upload --repo <user>/pheq-artifacts
Download (target cluster):
    .venv/bin/python scripts/data/transfer_artifacts.py download --repo <user>/pheq-artifacts

Default file set is the generation-phase kit; add more with --files.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

DEFAULT_FILES = [
    "outputs/wfit_sd_cluster.pt",
    "outputs/oracle_probe_sd_cluster.csv",
    "outputs/spectral_stats.json",
    "outputs/runs/b1/ckpt_latest.pt",
    "outputs/runs/b1/eval.json",
    "outputs/runs/p1_analytic/ckpt_latest.pt",
    "outputs/runs/p1_analytic/eval.json",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["upload", "download"])
    p.add_argument("--repo", required=True, help="HF repo id, e.g. <user>/pheq-artifacts")
    p.add_argument("--files", nargs="*", default=DEFAULT_FILES,
                   help="repo-relative paths (default: the generation-phase kit)")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("export HF_TOKEN=hf_... first (write token from "
                         "https://huggingface.co/settings/tokens)")

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    if args.mode == "upload":
        repo_id = api.create_repo(args.repo, private=True, exist_ok=True).repo_id
        for f in args.files:
            if not Path(f).exists():
                print(f"SKIP (missing): {f}")
                continue
            api.upload_file(path_or_fileobj=f, path_in_repo=f, repo_id=repo_id)
            print(f"uploaded: {f} ({Path(f).stat().st_size / 1e6:.0f} MB)")
    else:
        for f in args.files:
            try:
                cached = hf_hub_download(args.repo, f, token=token)
            except Exception as exc:  # missing on the hub: report, keep going
                print(f"SKIP ({type(exc).__name__}): {f}")
                continue
            Path(f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, f)
            print(f"downloaded: {f}")


if __name__ == "__main__":
    main()
