from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set

import numpy as np

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None

from component.ingestors.label_normalizer import normalize_label, load_or_init_registry, build_reverse_map, save_registry
from component.ingestors.adaptor_base import BaseAdaptor


@dataclass
class AggregateOptions:
    stage_dir: Optional[Path] = None
    h5_out_dir: Optional[Path] = None  # directory for output H5 files
    limit_per_dataset: int = 0  # 0 = no limit
    dry_run: bool = False
    # New fields for label SSOT
    registry_path: Optional[Path] = None  # defaults to project_root/data/h5_datasets/label_map.json
    dataset_key: Optional[str] = None
    allow_extend: bool = False  # if True, extend registry for unknown labels (default False)

def _encode_image_bytes(img_path: Path) -> bytes:
    """
    Encode image to bytes, preserving original format (PNG or JPEG).
    Falls back to raw bytes if PIL is not available or conversion fails.
    """
    if Image is None:
        # If PIL not available, store raw bytes
        return img_path.read_bytes()

    try:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            buf = io.BytesIO()

            # Determine format based on file extension
            suffix = img_path.suffix.lower()
            if suffix in {'.jpg', '.jpeg'}:
                # Use JPEG encoding with high quality
                im.save(buf, format="JPEG", quality=95, optimize=True)
            elif suffix in {'.png'}:
                # Use PNG encoding
                im.save(buf, format="PNG", optimize=True)
            else:
                # For other formats, try to detect from the file or default to PNG
                # Get format from the original image if available
                original_format = im.format
                if original_format in {'JPEG', 'JPG'}:
                    im.save(buf, format="JPEG", quality=95, optimize=True)
                elif original_format == 'PNG':
                    im.save(buf, format="PNG", optimize=True)
                else:
                    # Default to PNG for unknown formats
                    im.save(buf, format="PNG", optimize=True)

            return buf.getvalue()
    except Exception:
        # Fallback to raw bytes if conversion fails
        return img_path.read_bytes()


def aggregate_dataset(dataset_name: str, root: Path, adaptor: BaseAdaptor, options: AggregateOptions) -> Dict[str, int]:
    """
    Traverse a dataset with given loader, map labels to global canonical IDs using the
    central registry, and write an HDF5 file per dataset with consistent label IDs.

    Returns the registry mapping (canonical_name -> id) in use.
    """
    # Determine registry path default if not provided
    project_root = Path(__file__).resolve().parents[2]
    registry_path = options.registry_path or (project_root / "data" / "h5_datasets" / "label_map.json")
    id_map = load_or_init_registry(registry_path)
    id_to_name = build_reverse_map(id_map)

    # Accumulators
    images: List[bytes] = []
    labels: List[int] = []
    raw_to_id: Dict[str, int] = {}
    alias_warnings: Dict[str, Tuple[str, str]] = {}  # raw -> (normalized, canonical)
    unknowns: Set[str] = set()

    count = 0
    for img_path, raw_label in adaptor.iterate(root):
        if options.limit_per_dataset and count >= options.limit_per_dataset:
            break
        raw_str = str(raw_label)
        norm = normalize_label(raw_str)

        # If normalization changed the token, record an alias mapping notice
        if norm and norm != raw_str.lower():
            # final canonical key is norm (normalize_label already applies synonyms)
            alias_warnings[raw_str] = (norm, id_to_name.get(id_map.get(norm, -1), norm))

        if norm in id_map:
            label_idx = int(id_map[norm])
        else:
            if options.allow_extend:
                # Extend registry with new canonical name
                next_id = (max(id_map.values()) + 1) if id_map else 0
                id_map[norm] = next_id
                id_to_name[next_id] = norm
                label_idx = next_id
            else:
                # Skip unknown labels (not recognized via synonyms)
                unknowns.add(raw_str)
                continue

        raw_to_id.setdefault(raw_str, label_idx)

        try:
            img_bytes = _encode_image_bytes(Path(img_path))
            images.append(img_bytes)
            labels.append(label_idx)
        except Exception:
            continue

        count += 1

    # Early exit if dry-run
    out_dir = options.h5_out_dir
    if options.dry_run:
        # Optionally persist updated registry in dry-run if extended
        if options.allow_extend:
            save_registry(registry_path, id_map)
        return id_map

    assert out_dir is not None, "h5_out_dir must be provided when not in dry-run mode"
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_file_path = out_dir / f"{dataset_name}.h5"

    vlen_dt = h5py.vlen_dtype(np.dtype('uint8'))

    with h5py.File(str(h5_file_path), "w") as f:
        # Write primary datasets
        buffers = [np.frombuffer(b, dtype=np.uint8) for b in images]
        n_images = len(buffers)

        img_ds = f.create_dataset(
            "images",
            shape=(n_images,),
            maxshape=(None,),
            dtype=vlen_dt,
            compression="gzip",
        )
        labels_ds = f.create_dataset(
            "labels",
            shape=(n_images,),
            maxshape=(None,),
            dtype=np.int64,
            compression="gzip",
        )
        img_ds[:] = buffers
        labels_ds[:] = labels
        img_ds.attrs["count"] = n_images

        # Root attributes with registry information
        f.attrs["dataset_key"] = options.dataset_key or dataset_name
        # Store registry path relative to H5 location if possible
        try:
            rel = Path(os.path.relpath(registry_path, start=h5_file_path.parent))
            f.attrs["registry_relpath"] = str(rel)
        except Exception:
            f.attrs["registry_relpath"] = str(registry_path)
        f.attrs["id_to_emotion_json"] = json.dumps(id_to_name)

        # Meta group with per-dataset mapping
        meta = f.create_group("meta")
        meta.attrs["raw_to_id_json"] = json.dumps(raw_to_id)
        if alias_warnings:
            meta.attrs["alias_notes_json"] = json.dumps(alias_warnings)
        if unknowns:
            meta.attrs["unknown_labels_json"] = json.dumps(sorted(list(unknowns)))

    # Persist updated registry if extended
    if options.allow_extend:
        save_registry(registry_path, id_map)

    # Print warnings for visibility
    if alias_warnings:
        print(f"[alias] {dataset_name}: mapped {len(alias_warnings)} raw labels via aliases")
        for raw, (norm, canon) in list(alias_warnings.items())[:10]:
            print(f"  '{raw}' -> '{norm}' (id={id_map.get(norm)})")
    if unknowns:
        print(f"[warn] {dataset_name}: skipped {len(unknowns)} unknown labels:")
        print("  ", sorted(list(unknowns))[:20])

    return id_map
