from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import h5py  # type: ignore
import numpy as np
from PIL import Image  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from torch.utils.data import Dataset


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _bytes_to_image(img_bytes: bytes) -> "Image.Image":
    if Image is None:
        raise RuntimeError("Pillow (PIL) is required to decode images. Please install pillow.")
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def _read_vlen_uint8(ds: Any, idx: int) -> bytes:
    """Read a variable-length uint8 array element from HDF5 dataset as raw bytes."""
    arr = ds[idx]
    # arr may be numpy array of dtype uint8 or a Python bytes-like already
    if isinstance(arr, (bytes, bytearray)):
        return bytes(arr)
    if hasattr(arr, "dtype") and getattr(arr, "dtype", None) == np.uint8:
        return bytes(bytearray(np.asarray(arr).tolist()))
    # Fallbacks
    return bytes(bytearray(np.asarray(arr, dtype=np.uint8).tolist()))


def _to_int_label(x: Any) -> Optional[int]:
    try:
        if isinstance(x, (np.integer, int)):
            return int(x)
        if isinstance(x, (bytes, bytearray)):
            try:
                s = bytes(x).decode("utf-8").strip()
                if s.isdigit():
                    return int(s)
            except Exception:
                pass
            return int.from_bytes(bytes(x), byteorder="little", signed=False) if len(x) > 0 else None
        if hasattr(x, "dtype"):
            arr = np.asarray(x)
            if arr.size == 0:
                return None
            if np.issubdtype(arr.dtype, np.integer):
                if arr.shape == ():
                    return int(arr.item())
                if arr.dtype == np.uint8:
                    b = bytes(arr.tolist())
                    return int.from_bytes(b, byteorder="little", signed=False) if len(b) > 0 else None
                return int(arr.flatten()[0])
    except Exception:
        return None
    return None


# -----------------------------------------------------------------------------
# Stats container
# -----------------------------------------------------------------------------

@dataclass
class EDAStats:
    num_items: int
    label_counts: Dict[int, int]


# -----------------------------------------------------------------------------
# Torch Dataset backed by HDF5 (images: vlen uint8, labels: int-like)
# -----------------------------------------------------------------------------

class H5LabeledImageDataset(Dataset):  # type: ignore[misc]
    """Minimal PyTorch-compatible dataset for H5 files with
    top-level datasets: 'images' (vlen uint8) and 'labels' (int-like).

    It opens the file lazily in worker processes to be DataLoader-safe.
    """

    def __init__(
        self,
        h5_path: str | Path,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        return_index: bool = False,
    ) -> None:
        self.h5_path = str(h5_path)
        self.transform = transform
        self.target_transform = target_transform
        self.return_index = return_index

        # File handle/datasets are set in _ensure_open() lazily to be safe with workers
        self._file: Optional[h5py.File] = None
        self._images = None
        self._labels = None

        # Probe length without keeping file open permanently
        with h5py.File(self.h5_path, "r") as f:
            if "images" not in f or "labels" not in f:
                raise ValueError("HDF5 must contain top-level datasets 'images' and 'labels'.")
            n_images = len(f["images"])  # type: ignore[arg-type]
            try:
                n_labels = len(f["labels"])  # type: ignore[arg-type]
            except TypeError:
                n_labels = f["labels"].shape[0]  # type: ignore[index]
            self._length = min(n_images, n_labels)

    def _ensure_open(self) -> None:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
            self._images = self._file["images"]
            self._labels = self._file["labels"]

    def __len__(self) -> int:  # type: ignore[override]
        return self._length  # type: ignore[attr-defined]

    def __getitem__(self, index: int):  # type: ignore[override]
        self._ensure_open()
        assert self._images is not None and self._labels is not None

        raw_bytes = _read_vlen_uint8(self._images, index)
        try:
            img = _bytes_to_image(raw_bytes)
        except Exception as e:
            # Raise a clear error so callers can decide how to handle bad items.
            raise RuntimeError(f"Failed to decode image at index {index}: {e}") from e

        target = _to_int_label(self._labels[index])
        if target is None:
            target = 0

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_index:
            return img, target, index
        return img, target

    def __del__(self):  # pragma: no cover - best effort cleanup
        try:
            if getattr(self, "_file", None) is not None:
                self._file.close()  # type: ignore[union-attr]
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def h5_to_dataset(
    h5_path: str | Path,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    return_index: bool = False,
) -> H5LabeledImageDataset:
    """Create a PyTorch-compatible dataset from an HDF5 file.

    The file is expected to contain two top-level datasets:
    - 'images': variable-length arrays of dtype uint8 (encoded image bytes)
    - 'labels': integer label indices (or encodings convertible to int)

    Returns a lazily-opening dataset safe to use with multi-worker DataLoaders.
    """
    return H5LabeledImageDataset(
        h5_path=h5_path,
        transform=transform,
        target_transform=target_transform,
        return_index=return_index,
    )


def verify_h5_images(
    h5_path: str | Path,
    limit: Optional[int] = None,
    stop_on_error: bool = False,
    examples_per_label: int = 5,
    show_examples: bool = True,
    save_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Verify that each element in 'images' can be decoded as an image.

    Attempts to open each image with Pillow. Any failures are captured and
    reported. Returns a dict summary with counts and a list of failures.

    Additionally:
    - Collects and reports the set of unique labels found in the file.
    - Displays (and/or saves) the first N images for each label value.

    Args:
        h5_path: path to HDF5 file
        limit: if provided, verify at most this many items for decoding check
        stop_on_error: if True, raises after first failure; otherwise continues
        examples_per_label: how many example images to show per label (default 5)
        show_examples: if True and matplotlib is available, display per-label grids
        save_dir: if provided, also save per-label example grids as PNGs to this folder
    """
    h5_path = str(h5_path)
    failures: List[Dict[str, Any]] = []
    total = 0
    unique_labels: Optional[List[int]] = None
    example_indices_per_label: Dict[int, List[int]] = {}
    saved_paths_per_label: Dict[int, Optional[str]] = {}

    with h5py.File(h5_path, "r") as f:
        if "images" not in f:
            raise ValueError("HDF5 does not contain dataset 'images'.")
        images = f["images"]
        labels = f.get("labels")

        # 1) Verify images (optionally limited by 'limit')
        n_imgs_total = len(images)
        n = n_imgs_total
        if limit is not None:
            n = min(n, int(limit))
        for i in range(n):
            total += 1
            try:
                buf = _read_vlen_uint8(images, i)
                # Quick sanity: many image formats have a magic header; trying to decode is ultimate test
                _ = _bytes_to_image(buf)
            except Exception as e:
                lab = None
                try:
                    if labels is not None and i < len(labels):
                        lab = _to_int_label(labels[i])
                except Exception:
                    lab = None
                info = {
                    "index": i,
                    "label": lab,
                    "bytes": len(buf) if 'buf' in locals() else None,
                    "error": repr(e),
                    "head_hex": " ".join(f"{b:02x}" for b in (buf[:16] if 'buf' in locals() and buf is not None else b"")),
                }
                failures.append(info)
                if stop_on_error:
                    raise RuntimeError(f"Verification failed for image index {i}: {e}") from e

        # 2) Collect unique labels and the first K indices per label over the entire dataset
        if labels is not None:
            try:
                # Determine the paired length we can safely index for both images and labels
                try:
                    n_labels_total = len(labels)
                except TypeError:
                    n_labels_total = labels.shape[0]
                paired_n = min(n_imgs_total, n_labels_total)

                # Iterate once and collect unique labels + first indices per label
                u_set = set()
                example_indices_per_label = {}
                for i in range(paired_n):
                    il = _to_int_label(labels[i])
                    if il is None:
                        continue
                    u_set.add(il)
                    bucket = example_indices_per_label.setdefault(int(il), [])
                    if len(bucket) < examples_per_label:
                        bucket.append(i)
                unique_labels = sorted(u_set)
            except Exception:
                # Fallback: best effort unique labels only
                try:
                    u_set = set()
                    for i in range(len(labels)):
                        il = _to_int_label(labels[i])
                        if il is not None:
                            u_set.add(il)
                    unique_labels = sorted(u_set)
                except Exception:
                    unique_labels = None

            # 3) Decode and display/save examples per label (still inside the open file)
            if unique_labels is not None and examples_per_label > 0:
                # Prepare save directory if requested
                out_dir: Optional[Path] = None
                if save_dir is not None:
                    out_dir = Path(save_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                else:
                    saved_paths_per_label = {lbl: None for lbl in unique_labels}

                for lbl in unique_labels:
                    idxs = example_indices_per_label.get(lbl, [])
                    imgs = []
                    for idx in idxs:
                        try:
                            buf = _read_vlen_uint8(images, idx)
                            im = _bytes_to_image(buf)
                            imgs.append(im)
                        except Exception:
                            # Skip bad example silently; verification already recorded failures above
                            continue
                    if not imgs:
                        continue

                    # Display via matplotlib if requested and available
                    if show_examples and plt is not None:
                        cols = min(len(imgs), examples_per_label)
                        rows = 1
                        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, 2.2))
                        if cols == 1:
                            axes = [axes]  # type: ignore[list-item]
                        for ax, im in zip(axes, imgs[:cols]):
                            ax.imshow(im)
                            ax.axis('off')
                        fig.suptitle(f"Label {lbl} — first {len(imgs)} examples")
                        plt.tight_layout()
                        plt.show()

                    # Optionally save a grid image using PIL (works even without matplotlib)
                    if out_dir is not None:
                        try:
                            # Build a simple horizontal strip grid
                            thumb_w, thumb_h = 128, 128
                            thumbs = []
                            for im in imgs[:examples_per_label]:
                                t = im.copy()
                                t = t.resize((thumb_w, thumb_h))
                                thumbs.append(t)
                            grid_w = thumb_w * len(thumbs)
                            grid_h = thumb_h
                            grid = Image.new('RGB', (grid_w, grid_h), color=(0, 0, 0))
                            x = 0
                            for t in thumbs:
                                grid.paste(t, (x, 0))
                                x += thumb_w
                            out_path = out_dir / f"{Path(h5_path).stem}_label_{lbl}_examples.png"
                            grid.save(out_path)
                            saved_paths_per_label[lbl] = str(out_path)
                        except Exception:
                            # If saving fails, just ignore for now
                            saved_paths_per_label[lbl] = None

    summary = {
        "file": h5_path,
        "checked": total,
        "failures": failures,
        "num_failures": len(failures),
        "unique_labels": unique_labels,
        "num_unique_labels": (len(unique_labels) if unique_labels is not None else 0),
        "example_indices_per_label": example_indices_per_label,
        "saved_paths_per_label": saved_paths_per_label,
    }

    # Console output for quick visibility
    print(f"Verification summary for: {h5_path}")
    print(f"Checked images: {total}")
    print(f"Failures: {len(failures)}")
    if unique_labels is not None:
        print(f"Unique labels ({len(unique_labels)}): {unique_labels}")
    else:
        print("Unique labels: <unavailable>")
    if unique_labels is not None and example_indices_per_label:
        print(f"Per-label examples (up to {examples_per_label} each):")
        for lbl in unique_labels:
            idxs = example_indices_per_label.get(lbl, [])
            extra = ""
            sp = None if 'saved_paths_per_label' not in locals() else saved_paths_per_label.get(lbl)
            if sp:
                extra = f" saved-> {sp}"
            print(f"  label {lbl}: indices {idxs}{extra}")
    if failures:
        print("\nFailed items (up to first 20 shown):")
        for item in failures[:20]:
            print(
                f"- idx={item['index']}, label={item['label']}, bytes={item['bytes']}, "
                f"head={item['head_hex']}, error={item['error']}"
            )
    return summary

def _copy_item(src_h5, output_h5, name):
    obj = src_h5[name]
    if isinstance(obj, h5py.Group):
        print("group determined")
        # Ensure group exists in destination
        if name not in output_h5:
            dst_grp = output_h5.create_group(name)
            # copy group attributes
            for k, v in obj.attrs.items():
                dst_grp.attrs[k] = v
        else:
            dst_grp = output_h5[name]
        # Recurse over children
        for child_name in obj.keys():
            _copy_item(obj, dst_grp, child_name)
    elif isinstance(obj, h5py.Dataset):
        # Preserve existing data by creating once and appending thereafter.
        src_data = obj[()]
        # Handle variable-length uint8 datasets (common for encoded images)
        is_vlen_uint8 = False
        try:
            dt = obj.dtype
            is_vlen_uint8 = (dt.metadata or {}).get("vlen", None) == np.dtype("uint8")
        except Exception:
            is_vlen_uint8 = False

        if name not in output_h5:
            # Create destination dataset
            if is_vlen_uint8:
                vlen_dt = h5py.vlen_dtype(np.uint8)
                # store as 1D vlen array along axis 0
                output_ds = output_h5.create_dataset(
                    name,
                    shape=(len(src_data),),
                    maxshape=(None,),
                    dtype=vlen_dt,
                )
                output_ds[:] = src_data
            else:
                src_arr = np.asarray(src_data)
                # Ensure first dimension exists for appending semantics
                if src_arr.ndim == 0:
                    src_arr = src_arr[None]
                maxshape = (None,) + src_arr.shape[1:]
                output_ds = output_h5.create_dataset(
                    name,
                    shape=src_arr.shape,
                    maxshape=maxshape,
                    dtype=src_arr.dtype,
                    chunks=True,
                )
                output_ds[...] = src_arr

            # Copy dataset attributes on creation
            for k, v in obj.attrs.items():
                output_ds.attrs[k] = v
        else:
            output_ds = output_h5[name]
            if is_vlen_uint8:
                # Append vlen items one-by-one (h5py supports resizing 1D vlen)
                cur_len = len(output_ds)
                add_len = len(src_data)
                output_ds.resize((cur_len + add_len,))
                output_ds[cur_len : cur_len + add_len] = src_data
            else:
                src_arr = np.asarray(src_data)
                if src_arr.ndim == 0:
                    src_arr = src_arr[None]
                # Validate shape compatibility (all dims except axis 0)
                if output_ds.ndim != src_arr.ndim:
                    raise ValueError(f"Incompatible ndim when appending to '{name}': {output_ds.shape} vs {src_arr.shape}")
                if output_ds.shape[1:] != tuple(src_arr.shape[1:]):
                    raise ValueError(f"Incompatible shapes when appending to '{name}': {output_ds.shape} vs {src_arr.shape}")
                # Resize and append
                cur_len = output_ds.shape[0]
                add_len = src_arr.shape[0]
                output_ds.resize((cur_len + add_len,) + output_ds.shape[1:])
                output_ds[cur_len : cur_len + add_len, ...] = src_arr

            # Preserve existing attributes; only set missing ones
            for k, v in obj.attrs.items():
                if k not in output_ds.attrs:
                    output_ds.attrs[k] = v

    else:
        # Unknown HDF5 type; skip or handle as needed
        pass

def combine_h5(src_files, output_file):
    with h5py.File(output_file, 'w') as output_h5:
        for filename in src_files:
            print("Processing", filename)
            with h5py.File(filename, 'r') as src_h5:
                # Report total size (number of elements) for each dataset under this source file
                def _report_sizes(h5obj, prefix=""):
                    for name, item in h5obj.items():
                        full_name = f"{prefix}/{name}" if prefix else name
                        if isinstance(item, h5py.Dataset):
                            # total size as number of elements
                            print(f"[SRC SIZE] {full_name}: {item.size}")
                        elif isinstance(item, h5py.Group):
                            _report_sizes(item, full_name)
                _report_sizes(src_h5)

                # Iterate over top-level items (groups or datasets)
                for name in src_h5.keys():
                    print("Copying", name)
                    _copy_item(src_h5, output_h5, name)

        # After copying is done, report sizes in the output file
        print("Final sizes in output file:")
        def _report_output_sizes(h5obj, prefix=""):
            for name, item in h5obj.items():
                full_name = f"{prefix}/{name}" if prefix else name
                if isinstance(item, h5py.Dataset):
                    print(f"[OUT SIZE] {full_name}: {item.size}")
                elif isinstance(item, h5py.Group):
                    _report_output_sizes(item, full_name)
        _report_output_sizes(output_h5)

__all__ = [
    "H5LabeledImageDataset",
    "h5_to_dataset",
    "verify_h5_images",
    "EDAStats",
    "combine_h5"
]