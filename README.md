# MultiHeadEmoteNet
Exploring a multi-head, squeeze-excitation, residual convolutional network for emotion classification


## Dataset aggregation to unified HDF5

This project now includes a lightweight pipeline to fetch common emotion datasets (via Kaggle), traverse their unique folder structures, normalize labels to a canonical set, optionally materialize a unified directory structure, and build a combined HDF5 for training.

Canonical labels used: angry, disgust, fear, happy, neutral, sad, surprise. If additional labels are discovered, they are added automatically and recorded in data/label_map.json.

### 1) Fetch/record dataset paths

Use the registry manager to download datasets via Kaggle Hub and store their local paths:

```
python -m component.dataset_registry_manager
```

This writes data/data_sets.csv with three columns: dataset_name, file_path_str, loader_class.

### 2) Build unified directory and/or HDF5

```
python -m component.build_h5_dataset --registry data/data_sets.csv \
  --stage-dir data/aggregated_images \
  --h5-out data/aggregated.h5
```

Flags:
- --stage-dir: optional on-disk unified structure with top-level folders per label (disabled by default). Images from all datasets merge into these folders.
- --h5-out: output HDF5 storing PNG-compressed bytes per image and integer labels.
- --limit-per-dataset: integer limit for quick tests (default 0 = no limit).
- --skip-missing: skip registry entries with missing paths.
- --dry-run: just traverse and report counts without writing files.

### Loaders

Implemented loaders under tool/loaders/:
- GenericImageFolderLoader: recursively scans and uses nearest parent folder (ignoring split dirs like train/val/test) as label.
- FERLoader: flattens split folders with class subfolders.
- FilenameLabelLoader: derives label from the filename (e.g., happy_001.png -> happy).
- YOLOEmotionLoader: reads data.yaml to map class IDs; falls back to folder names if missing.
- FER2013Loader: stub – recognizes fer2013.csv but does not convert CSV rows to images yet.

You can adjust the loader class names per dataset in tool/dataset_registry_manager.py. The build script resolves them dynamically.

# MultiHeadEmoteNet
Exploring a multi-head, squeeze-excitation, residual convolutional network for emotion classification


## Dataset aggregation to unified HDF5

This project now includes a lightweight pipeline to fetch common emotion datasets (via Kaggle), traverse their unique folder structures, normalize labels to a canonical set, optionally materialize a unified directory structure, and build a combined HDF5 for training.

Canonical labels used: angry, disgust, fear, happy, neutral, sad, surprise. If additional labels are discovered, they are added automatically and recorded in data/label_map.json.

### 1) Fetch/record dataset paths

Use the registry manager to download datasets via Kaggle Hub and store their local paths:

```
python -m component.dataset_registry_manager
```

This writes data/data_sets.csv with three columns: dataset_name, file_path_str, loader_class.

### 2) Build unified directory and/or HDF5

```
python -m component.build_h5_dataset --registry data/data_sets.csv \
  --stage-dir data/aggregated_images \
  --h5-out data/aggregated.h5
```

Flags:
- --stage-dir: optional on-disk unified structure with top-level folders per label (disabled by default). Images from all datasets merge into these folders.
- --h5-out: output HDF5 storing PNG-compressed bytes per image and integer labels. Default: <project_root>/data/dataset.h5
- --limit-per-dataset: integer limit for quick tests (default 0 = no limit).
- --skip-missing: skip registry entries with missing paths.
- --dry-run: just traverse and report counts without writing files.

Note:
- The build script can be invoked as `python -m component.build_h5_dataset`. If you omit `--h5-out`, it writes to `<project_root>/data/dataset.h5`.

### Loaders

Implemented loaders under tool/loaders/:
- GenericImageFolderLoader: recursively scans and uses nearest parent folder (ignoring split dirs like train/val/test) as label.
- FERLoader: flattens split folders with class subfolders.
- FilenameLabelLoader: derives label from the filename (e.g., happy_001.png -> happy).
- YOLOEmotionLoader: reads data.yaml to map class IDs; falls back to folder names if missing.
- FER2013Loader: stub – recognizes fer2013.csv but does not convert CSV rows to images yet.

You can adjust the loader class names per dataset in tool/dataset_registry_manager.py. The build script resolves them dynamically.


## HDF5 Dataset + EDA utilities

Use the PyTorch-compatible dataset with rich EDA and duplicate detection:

```
python -m component.dataset.dataset --h5 data\dataset.h5 --out data\reports --report-csv data\reports\eda_summary.csv \
  --use-opencv --haar evaluation\ResEmoteNet\haarcascade_frontalface_default.xml \
  --perceptual-dedup --hash-type phash --hash-distance 0
```

Outputs include per-label plots for:
- Image size, color balance, brightness, sharpness
- HSV saturation/value (when OpenCV is available)
- Entropy, contrast (RMS of luminance), and aspect ratio
- Face-detection coverage rates per label (OpenCV Haar)

Also supports duplicate detection:
- Exact byte duplicates via MD5 (non-destructive removal from dataset instance)
- Perceptual duplicates via ImageHash (phash/ahash/dhash/whash)



## Use in Jupyter (quick start)

In a notebook cell:

```python
from component.dataset import H5LabeledImageDataset

# Load the default dataset at <project_root>/data/dataset.h5
ds = H5LabeledImageDataset()

# Run EDA and get a pandas DataFrame summary plus saved plot paths
by_label, paths, df = ds.run_eda(return_dataframe=True)
print('Saved plots:')
for p in paths:
    print('-', p)
df
```

Optional extras:

```python
# Face detection coverage (requires OpenCV and a Haar cascade; uses bundled default when present)
try:
    face_stats = ds.compute_face_detection_stats()
    p = ds.plot_face_detection_rates(face_stats)
    print('Wrote:', p)
except Exception as e:
    print('[INFO] Skipping face detection:', e)

# Duplicate detection (non-destructive to HDF5)
md5_map, dups = ds.build_md5_map_and_find_duplicates()
print(f'Found {len(dups)} duplicates')
removed = ds.remove_duplicates_in_place()
print(f'Removed {len(removed)} from this dataset instance. New len: {len(ds)}')
```

You can also see a complete, copy‑pastable example in:
- evaluation/H5Dataset_EDA_Quickstart.py
