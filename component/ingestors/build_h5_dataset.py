from __future__ import annotations

import argparse
import json
import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import List, Tuple

from component.ingestors.data_aggregator import AggregateOptions, aggregate_dataset
from component.ingestors import resolve_adaptor
from component.ingestors.dataset_registry_manager import generate_registry

# Read the CACHE_PATH from environment variable KAGGLE_DATASET_HOME, default to /home/julian/.cache/kagglehub/datasets
KAGGLE_DATASET_HOME = Path(os.environ.get("KAGGLE_DATASET_HOME", "/home/julian/.cache/kagglehub/datasets"))

def read_registry(json_path: Path) -> List[Tuple[str, str, Path, str]]:
    rows: List[Tuple[str, str, Path, str]] = []
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        for ds in data.get("datasets", []):
            ds_name = ds["dataset_name"]
            file_path_str = ds["file_path_str"]
            kaggle_key = ds["kaggle_key"]
            adaptor_cls = ds["loader_class"]
            rows.append((ds_name, kaggle_key, Path(file_path_str) if file_path_str else Path(""), adaptor_cls))
    return rows


def setup_logging(project_root: Path) -> None:
    """Configure logging with file rotation (24 hours)."""
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "build_h5_dataset.log"
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    
    # File handler with 24-hour rotation
    file_handler = TimedRotatingFileHandler(
        log_file,
        when="midnight",  # Rotate at midnight
        interval=1,       # Every 1 day
        backupCount=30,   # Keep 30 days of logs
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

def main() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent

    ap = argparse.ArgumentParser(description="Build HDF5 files (one per dataset) from heterogeneous emotion datasets")
    ap.add_argument("--registry", type=str, default=project_root / "data" / "datasets.json", help="JSON registry with dataset information")
    # Label SSOT configuration
    ap.add_argument("--label-registry", type=str, default=str(project_root / "data" / "h5_datasets" / "label_map.json"), help="Path to the central label registry JSON (SSOT)")
    ap.add_argument("--dataset-key", type=str, default=None, help="Optional dataset key to embed in H5 metadata (defaults to dataset_name)")
    ap.add_argument("--allow-extend", action="store_true", help="Allow extending the label registry with new classes (default: False)")
    # General options
    ap.add_argument("--stage-dir", type=str, default=None, help="Optional output directory to stage unified label folders (disabled by default)")
    ap.add_argument("--h5-out-dir", type=str, default=project_root / "data" / "h5_datasets", help="Output directory for HDF5 files (default: <project_root>/data/h5_datasets)")
    ap.add_argument("--limit-per-dataset", type=int, default=0, help="Limit number of images per dataset (0 = no limit)")
    ap.add_argument("--skip-missing", action="store_true", help="Skip registry rows with empty or non-existent paths")
    ap.add_argument("--dry-run", action="store_true", help="Traverse and report without writing outputs")
    args = ap.parse_args()

    # Configure logging with file rotation
    setup_logging(project_root)
    logger = logging.getLogger(__name__)

    registry_path = Path(args.registry)
    if not registry_path.exists():
        logger.info(f"Registry not found at {registry_path}, generating...")
        generate_registry()

    options = AggregateOptions(
        stage_dir=Path(args.stage_dir) if args.stage_dir else None,
        h5_out_dir=Path(args.h5_out_dir) if args.h5_out_dir else None,
        limit_per_dataset=args.limit_per_dataset,
        dry_run=bool(args.dry_run),
        registry_path=Path(args.label_registry) if args.label_registry else None,
        dataset_key=args.dataset_key,
        allow_extend=bool(args.allow_extend),
    )

    rows = read_registry(registry_path)
    logger.info(f"Loaded {len(rows)} datasets from registry")

    total = 0
    for ds_name, kaggle_key, path, adaptor_name in rows:
        full_path = KAGGLE_DATASET_HOME / path
        if args.skip_missing and (not path or not str(path) or not Path(path).exists()):
            logger.warning(f"Skipping missing dataset: {ds_name}")
            print(f"Skipping missing dataset: {ds_name}")
            continue
        if not path or not str(path):
            logger.warning(f"Dataset path missing: {ds_name} (loader={adaptor_name})")
            print(f"Dataset path missing: {ds_name} (loader={adaptor_name})")
            continue
        if not Path(full_path).exists():
            print(f"Dataset path does not exist: {full_path}")
            continue
        try:
            adaptor_class = resolve_adaptor(adaptor_name)
        except KeyError as e:
            logger.error(f"Unknown loader for {ds_name}: {e}")
            print(f"Unknown loader for {ds_name}: {e}")
            continue
        adaptor = adaptor_class()
        print(f"Processing {ds_name} with {adaptor_name} at {full_path}...")
        label_map = aggregate_dataset(ds_name, Path(full_path), adaptor, options)
        
        if not args.dry_run and options.h5_out_dir:
            h5_path = options.h5_out_dir / f"{ds_name}.h5"
            logger.info(f"Completed {ds_name} -> {h5_path}")
            print(f"Completed {ds_name} -> {h5_path}")
        
        total += 1

    logger.info(f"Done. Processed {total} datasets. Outputs:\n  Stage dir: {args.stage_dir}\n  H5 output dir: {args.h5_out_dir}")
    print(f"Done. Processed {total} datasets. Outputs:\n  Stage dir: {args.stage_dir}\n  H5 output dir: {args.h5_out_dir}")


if __name__ == "__main__":
    main()
