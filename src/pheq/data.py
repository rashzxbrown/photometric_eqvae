"""Image pipeline: folder dataset, seeded loaders, deterministic splits.

Implements the ``data.py`` section of SPEC2.md. Global conventions
(SPEC.md): images are ``(3, H, W)`` float32 in ``[0, 1]`` per sample,
``(B, 3, H, W)`` after collation.

Reproducibility contract (SPEC2 "data.py"): file order is a deterministic
sort of the recursive glob, and all shuffling is driven by an explicit
``torch.Generator`` — two loaders built with identically seeded generators
yield identical batches. This is what makes train_ae runs (and their
resumes) comparable across conditions (docs/plan-3month.md M2).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from pheq.probes._common import IMAGE_EXTENSIONS


class ImageFolderDataset(Dataset):
    """Recursive image-folder dataset with deterministic ordering.

    SPEC2 "data.py": recursive glob of :data:`IMAGE_EXTENSIONS` (the v1
    list from ``pheq.probes._common``), files in deterministic sorted order
    (sorted by full path string). Per image: load as RGB, resize the
    SHORTER side to ``size`` (bilinear, antialiased), center-crop to
    ``size x size``, convert to float32 ``[0, 1]`` CHW.

    Args:
        root: directory to search recursively for images.
        size: output resolution (square, ``size x size``).
        class_from_dir: if True, the label is the index of the image's
            immediate parent directory name within the sorted set of all
            parent names present (the ImageNet-100 layout later in the
            sprint); if False every label is 0.

    ``__getitem__`` returns ``(img, label)`` with ``img`` a float32
    ``(3, size, size)`` tensor in ``[0, 1]`` and ``label`` an int.
    """

    def __init__(self, root: str, size: int = 256, class_from_dir: bool = False) -> None:
        self.root = Path(root)
        self.size = int(size)
        self.class_from_dir = bool(class_from_dir)
        # Deterministic sorted order (SPEC2 reproducibility contract).
        # os.walk instead of Path.rglob: pathlib's glob SWALLOWS scandir
        # errors (PermissionError/OSError) and silently yields nothing —
        # on networked filesystems (Lustre/NFS) that turns a transient
        # mount/ACL hiccup into a bogus "no images" result. os.walk lets us
        # collect and report the real errors, and follows dir symlinks.
        walk_errors: list[OSError] = []
        found: list[Path] = []
        for dirpath, _dirnames, filenames in os.walk(
            self.root, onerror=walk_errors.append, followlinks=True
        ):
            for name in filenames:
                if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                    found.append(Path(dirpath) / name)
        self.paths: list[Path] = sorted(found, key=str)
        if not self.paths:
            try:
                listing = sorted(os.listdir(self.root))
            except OSError as e:
                listing = [f"<listdir failed: {e}>"]
            raise FileNotFoundError(
                f"no images with extensions {IMAGE_EXTENSIONS} under {str(self.root)!r}; "
                f"walk errors: {[str(e) for e in walk_errors] or 'none'}; "
                f"root has {len(listing)} entries, first 5: {listing[:5]}"
            )
        #: Sorted parent-directory names -> class indices (used when
        #: ``class_from_dir``; exposed for downstream label decoding).
        self.classes: list[str] = sorted({p.parent.name for p in self.paths})
        self._class_to_idx = {name: i for i, name in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.paths)

    def _load(self, path: Path) -> torch.Tensor:
        """Load one image: RGB -> shorter-side resize -> center crop -> CHW [0,1]."""
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        img = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
        _, h, w = img.shape
        s = self.size
        # Resize the SHORTER side to `size`, preserving aspect ratio
        # (bilinear, antialias=True — SPEC2 "data.py").
        if h <= w:
            new_h, new_w = s, max(s, round(w * s / h))
        else:
            new_h, new_w = max(s, round(h * s / w)), s
        if (new_h, new_w) != (h, w):
            img = F.interpolate(
                img[None],
                size=(new_h, new_w),
                mode="bilinear",
                antialias=True,
                align_corners=False,
            )[0]
            # The antialiased kernel is a convex combination, but float
            # rounding can overshoot [0, 1] by ~1e-7; clamp to keep the
            # SPEC [0, 1] image convention exact. (This clamps loaded
            # files only — nothing to do with pre-clip targets.)
            img = img.clamp_(0.0, 1.0)
        # Center crop AFTER the resize (SPEC2 care point).
        top = (new_h - s) // 2
        left = (new_w - s) // 2
        return img[:, top : top + s, left : left + s].contiguous()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path = self.paths[index]
        img = self._load(path)
        label = self._class_to_idx[path.parent.name] if self.class_from_dir else 0
        return img, label


def make_loader(
    dataset: Dataset,
    batch_size: int,
    gen: torch.Generator,
    workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """Seeded DataLoader (SPEC2 "data.py").

    Shuffle order is drawn from ``gen`` (the DataLoader ``generator``
    argument seeds the RandomSampler), so two loaders whose generators were
    seeded identically produce identical batch sequences — the
    reproducibility contract. ``drop_last=True`` keeps every step's batch
    size constant; ``pin_memory`` is enabled iff CUDA is available.

    Args:
        dataset: any map-style dataset (typically :class:`ImageFolderDataset`).
        batch_size: samples per batch.
        gen: explicit ``torch.Generator`` driving the shuffle (and worker
            base seeds).
        workers: DataLoader worker processes (0 = load in the main process).
        shuffle: pass False for evaluation loaders (order = dataset order).
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=gen,
        num_workers=workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )


def train_val_split(
    dataset: Dataset, val_n: int, gen: torch.Generator
) -> tuple[Subset, Subset]:
    """Deterministic train/val split (SPEC2 "data.py").

    The validation set is the FIRST ``val_n`` indices of a permutation drawn
    from ``gen``; the training set is the remainder. Same seed -> same split,
    so the fixed val images used by train_ae's monitors/eval hooks are stable
    across runs and resumes.

    Args:
        dataset: dataset to split.
        val_n: number of validation samples (``0 <= val_n <= len(dataset)``).
        gen: seeded generator defining the permutation.

    Returns:
        ``(train_subset, val_subset)``.
    """
    n = len(dataset)  # type: ignore[arg-type]
    if not 0 <= val_n <= n:
        raise ValueError(f"val_n={val_n} out of range for dataset of length {n}")
    perm = torch.randperm(n, generator=gen).tolist()
    val = Subset(dataset, perm[:val_n])
    train = Subset(dataset, perm[val_n:])
    return train, val
