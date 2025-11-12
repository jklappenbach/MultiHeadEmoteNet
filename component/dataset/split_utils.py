from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from pathlib import Path

import h5py  # type: ignore
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

try:
    from torchvision import transforms as T  # type: ignore
except Exception:  # pragma: no cover - optional import at runtime
    T = None  # type: ignore

from .dataset import H5LabeledImageDataset


# -----------------------------------------------------------------------------
# Seeding helpers
# -----------------------------------------------------------------------------

def seed_all(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    """Worker initializer to make DataLoader workers deterministically seeded."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# -----------------------------------------------------------------------------
# Stratified splitting
# -----------------------------------------------------------------------------

@dataclass
class SplitIndices:
    train: List[int]
    val: List[int]
    test: List[int]

    def to_dict(self) -> Dict[str, List[int]]:
        return {"train": self.train, "val": self.val, "test": self.test}

    @staticmethod
    def from_dict(d: Dict[str, Iterable[int]]) -> "SplitIndices":
        return SplitIndices(list(d["train"]), list(d["val"]), list(d["test"]))


def compute_stratified_indices(
    labels: np.ndarray,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    ensure_min_per_class: bool = True,
) -> SplitIndices:
    """Compute stratified train/val/test indices without external deps.

    - Preserves class proportions by splitting within each class bucket.
    - If `ensure_min_per_class`, guarantees at least one sample goes to each
      split for classes with >= 3 items (best effort for tiny classes).
    """
    assert len(ratios) == 3, "ratios must be a 3-tuple (train, val, test)"
    tr, vr, te = ratios
    if not np.isclose(tr + vr + te, 1.0):
        raise ValueError("ratios must sum to 1.0")

    rng = np.random.RandomState(seed)
    indices = np.arange(len(labels))

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    classes = np.unique(labels)
    for c in classes:
        cls_idx = indices[labels == c]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        if n == 0:
            continue

        # Base allocation via rounding
        n_train = int(round(n * tr))
        n_val = int(round(n * vr))
        n_test = n - n_train - n_val

        # Fix rounding edge cases
        if n_test < 0:
            n_test = 0
            n_val = n - n_train
        if n_val < 0:
            n_val = 0
            n_test = n - n_train
        if n_train < 0:
            n_train = 0

        if ensure_min_per_class and n >= 3:
            # Try to ensure at least 1 per split if possible
            if n_train == 0:
                n_train, n_val = 1, max(0, n_val - 1)
            if n_val == 0 and (n - n_train) >= 2:
                n_val, n_test = 1, max(0, n_test - 1)
            if n_test == 0 and (n - n_train - n_val) >= 1:
                n_test = 1
                # Reduce the larger of train/val if necessary
                if n_val > n_train and n_val > 1:
                    n_val -= 1
                elif n_train > 1:
                    n_train -= 1

        # Final sanity: do not exceed n
        total = n_train + n_val + n_test
        if total > n:
            # Reduce the largest bucket
            for _ in range(total - n):
                if n_train >= n_val and n_train >= n_test and n_train > 0:
                    n_train -= 1
                elif n_val >= n_test and n_val > 0:
                    n_val -= 1
                elif n_test > 0:
                    n_test -= 1

        train_idx.extend(cls_idx[:n_train].tolist())
        val_idx.extend(cls_idx[n_train:n_train + n_val].tolist())
        test_idx.extend(cls_idx[n_train + n_val:n_train + n_val + n_test].tolist())

    # Shuffle within buckets for randomness
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return SplitIndices(train=train_idx, val=val_idx, test=test_idx)


# -----------------------------------------------------------------------------
# Persistence helpers
# -----------------------------------------------------------------------------

def split_filename_for(h5_path: str | Path, split_id: Optional[str] = None, seed: int = 42) -> str:
    base = Path(h5_path).stem
    split_id = split_id or f"seed_{seed}"
    splits_dir = Path("splits")
    splits_dir.mkdir(parents=True, exist_ok=True)
    return str(splits_dir / f"{base}_stratified_{split_id}.json")


def save_split(path: str, split: SplitIndices) -> None:
    with open(path, "w") as f:
        json.dump(split.to_dict(), f)


def load_split(path: str) -> SplitIndices:
    with open(path, "r") as f:
        data = json.load(f)
    return SplitIndices.from_dict(data)


# -----------------------------------------------------------------------------
# Main entry: build DataLoaders from H5 with stratified, persisted split
# -----------------------------------------------------------------------------

def default_transform() -> Optional["T.Compose"]:
    if T is None:
        return None
    return T.Compose([
        T.Resize((64, 64)),
        T.Grayscale(num_output_channels=3),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def make_h5_loaders_stratified(
    h5_path: str | Path,
    batch_size: int = 64,
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    split_id: Optional[str] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    transform: Optional[object] = None,
):
    """Create stratified train/val/test DataLoaders from a single H5 dataset.

    Accepts `h5_path` as either a string or `pathlib.Path`.

    - Loads labels directly from the H5 file (no image decoding).
    - Computes per-class stratified indices with the given `seed` and `ratios`.
    - Saves/loads the exact split indices in `splits/` for reproducibility.

    Returns: (train_loader, val_loader, test_loader, split_file)
    """
    seed_all(seed)

    h5_path = Path(h5_path)

    # Load labels (fast)
    with h5py.File(str(h5_path), "r") as f:
        labels = np.asarray(f["labels"])  # shape (N,)

    split_path = split_filename_for(h5_path, split_id=split_id, seed=seed)

    if os.path.exists(split_path):
        split = load_split(split_path)
    else:
        split = compute_stratified_indices(labels=labels, ratios=ratios, seed=seed)
        save_split(split_path, split)
        print(f"Saved stratified split to {split_path}")

    # Build datasets
    tfm = transform if transform is not None else default_transform()
    full_ds = H5LabeledImageDataset(h5_path, transform=tfm)
    train_ds = Subset(full_ds, split.train)
    val_ds = Subset(full_ds, split.val)
    test_ds = Subset(full_ds, split.test)

    # Seeded generator for deterministic per-epoch shuffle
    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader, test_loader, split_path
