"""Tests for pheq.data per SPEC2.md "data.py".

Covers: deterministic sorted file order, the seeded-shuffle reproducibility
contract (two loaders, same seed -> identical batches), shorter-side resize
followed by center crop, labels (class_from_dir and default), drop_last, and
the deterministic train/val split.

All images are tiny synthetic PNGs written to tmp_path — fully offline.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from pheq.data import ImageFolderDataset, make_loader, train_val_split


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path)


def _const_image(h: int, w: int, rgb: tuple[int, int, int]) -> np.ndarray:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:] = rgb
    return arr


@pytest.fixture()
def flat_dir(tmp_path: Path) -> Path:
    """12 distinct constant-color 40x64 images in one directory."""
    root = tmp_path / "imgs"
    for i in range(12):
        _write_png(root / f"img_{i:02d}.png", _const_image(40, 64, (20 * i, 10, 255 - 20 * i)))
    return root


# ---------------------------------------------------------------------------
# file order / loading / crop
# ---------------------------------------------------------------------------


def test_sorted_deterministic_order(flat_dir: Path) -> None:
    ds1 = ImageFolderDataset(str(flat_dir), size=32)
    ds2 = ImageFolderDataset(str(flat_dir), size=32)
    assert ds1.paths == ds2.paths
    assert ds1.paths == sorted(ds1.paths, key=str)
    assert len(ds1) == 12


def test_recursive_glob_and_extension_filter(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    _write_png(root / "a" / "x.png", _const_image(16, 16, (1, 2, 3)))
    _write_png(root / "a" / "deep" / "y.jpg", _const_image(16, 16, (4, 5, 6)))
    (root / "notes.txt").parent.mkdir(parents=True, exist_ok=True)
    (root / "notes.txt").write_text("not an image")
    ds = ImageFolderDataset(str(root), size=8)
    assert len(ds) == 2  # recursive, non-image files ignored


def test_empty_dir_raises(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        ImageFolderDataset(str(tmp_path / "empty"), size=8)


def test_shape_dtype_range_and_constant_preserved(flat_dir: Path) -> None:
    ds = ImageFolderDataset(str(flat_dir), size=32)
    img, label = ds[0]
    assert img.shape == (3, 32, 32)
    assert img.dtype == torch.float32
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0
    assert label == 0
    # Constant image survives antialiased resize + crop exactly (bilinear
    # weights sum to 1), pinning down the value convention [0, 1] = /255.
    expected = torch.tensor([0.0, 10 / 255, 1.0]).view(3, 1, 1).expand(3, 32, 32)
    assert torch.allclose(img, expected, atol=1e-5)


def test_center_crop_after_shorter_side_resize(tmp_path: Path) -> None:
    # H=32 == size, W=64: shorter-side resize is the identity, so the crop
    # window is exactly columns [16, 48). White there, black elsewhere.
    arr = np.zeros((32, 64, 3), dtype=np.uint8)
    arr[:, 16:48, :] = 255
    _write_png(tmp_path / "wide" / "w.png", arr)
    ds = ImageFolderDataset(str(tmp_path / "wide"), size=32)
    img, _ = ds[0]
    assert torch.all(img == 1.0)


def test_tall_image_resizes_width(tmp_path: Path) -> None:
    # H=80 > W=40: the WIDTH is the shorter side -> width becomes `size`.
    _write_png(tmp_path / "tall" / "t.png", _const_image(80, 40, (100, 100, 100)))
    ds = ImageFolderDataset(str(tmp_path / "tall"), size=20)
    img, _ = ds[0]
    assert img.shape == (3, 20, 20)


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------


def test_labels_class_from_dir(tmp_path: Path) -> None:
    root = tmp_path / "classes"
    _write_png(root / "wolf" / "a.png", _const_image(16, 16, (1, 1, 1)))
    _write_png(root / "ant" / "b.png", _const_image(16, 16, (2, 2, 2)))
    _write_png(root / "ant" / "c.png", _const_image(16, 16, (3, 3, 3)))
    ds = ImageFolderDataset(str(root), size=8, class_from_dir=True)
    assert ds.classes == ["ant", "wolf"]  # sorted parent names
    labels = [ds[i][1] for i in range(len(ds))]
    by_parent = {str(p): lab for p, lab in zip(ds.paths, labels)}
    for path_str, lab in by_parent.items():
        assert lab == (0 if "/ant/" in path_str else 1)


def test_labels_default_zero(flat_dir: Path) -> None:
    ds = ImageFolderDataset(str(flat_dir), size=8, class_from_dir=False)
    assert all(ds[i][1] == 0 for i in range(len(ds)))


# ---------------------------------------------------------------------------
# loader: seeded shuffle reproducibility contract, drop_last
# ---------------------------------------------------------------------------


def test_loader_same_seed_identical_batches(flat_dir: Path) -> None:
    ds = ImageFolderDataset(str(flat_dir), size=16)
    batches = []
    for _ in range(2):
        gen = torch.Generator().manual_seed(123)
        loader = make_loader(ds, batch_size=4, gen=gen, workers=0, shuffle=True)
        batches.append([(x.clone(), y.clone()) for x, y in loader])
    assert len(batches[0]) == len(batches[1]) == 3
    for (x1, y1), (x2, y2) in zip(*batches):
        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)


def test_loader_shuffles_and_seeds_differ(flat_dir: Path) -> None:
    ds = ImageFolderDataset(str(flat_dir), size=16)

    def first_batch(seed: int, shuffle: bool) -> torch.Tensor:
        gen = torch.Generator().manual_seed(seed)
        loader = make_loader(ds, batch_size=12, gen=gen, workers=0, shuffle=shuffle)
        return next(iter(loader))[0]

    unshuffled = first_batch(0, shuffle=False)
    in_order = torch.stack([ds[i][0] for i in range(12)])
    assert torch.equal(unshuffled, in_order)  # shuffle=False keeps sorted order
    # A seeded shuffle of 12 distinct images: identity permutation has
    # probability 1/12! — treat a match as failure.
    assert not torch.equal(first_batch(0, shuffle=True), in_order)


def test_loader_drop_last(flat_dir: Path) -> None:
    ds = ImageFolderDataset(str(flat_dir), size=16)  # 12 images
    gen = torch.Generator().manual_seed(0)
    loader = make_loader(ds, batch_size=5, gen=gen, workers=0)
    sizes = [x.shape[0] for x, _ in loader]
    assert sizes == [5, 5]  # trailing 2 dropped


# ---------------------------------------------------------------------------
# train/val split
# ---------------------------------------------------------------------------


def test_train_val_split_deterministic_and_disjoint(flat_dir: Path) -> None:
    ds = ImageFolderDataset(str(flat_dir), size=16)
    train1, val1 = train_val_split(ds, val_n=3, gen=torch.Generator().manual_seed(7))
    train2, val2 = train_val_split(ds, val_n=3, gen=torch.Generator().manual_seed(7))
    assert len(val1) == 3 and len(train1) == 9
    assert val1.indices == val2.indices and train1.indices == train2.indices
    assert set(val1.indices) | set(train1.indices) == set(range(12))
    assert set(val1.indices) & set(train1.indices) == set()


def test_train_val_split_bad_val_n(flat_dir: Path) -> None:
    ds = ImageFolderDataset(str(flat_dir), size=16)
    with pytest.raises(ValueError):
        train_val_split(ds, val_n=13, gen=torch.Generator().manual_seed(0))
