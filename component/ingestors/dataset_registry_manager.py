import json
from pathlib import Path
import logging
from logging.handlers import TimedRotatingFileHandler
import kagglehub

# Central registry of datasets and their intended loader class names.
# Adjust loader_class values to your actual dataset loader implementations.
DATASETS = [
    # (file_path, kaggle_key, dataset_name, loader_class)
    ("aklimarimi/8-facial-expressions-for-yolo/versions/4/9 Facial Expressions you need", "aklimarimi/8-facial-expressions-for-yolo", "8-facial-expressions-for-yolo", "YOLOEmotionAdaptor"),
    ("ananthu017/emotion-detection-fer/versions/1", "ananthu017/emotion-detection-fer", "emotion-detection-fer", "FERAdaptor"),
    ("sujaykapadnis/emotion-recognition-dataset/versions/1", "sujaykapadnis/emotion-recognition-dataset", "emotion-recognition-dataset", "GenericImageFolderAdaptor"),
    ("msambare/fer2013/versions/1", "msambare/fer2013", "fer2013", "FER2013Adaptor"),
    ("juniorbueno/rating-opencv-emotion-images/versions/1", "juniorbueno/rating-opencv-emotion-images", "rating-opencv-emotion-images", "GenericImageFolderAdaptor"),
    ("yousefmohamed20/sentiment-images-classifier/versions/3/6 Emotions for image classification", "yousefmohamed20/sentiment-images-classifier", "sentiment-images-classifier", "GenericImageFolderAdaptor"),
    ("tapakah68/facial-emotion-recognition/versions/2", "tapakah68/facial-emotion-recognition", "facial-emotion-recognition", "FilenameLabelAdaptor"),
    ("sudarshanvaidya/random-images-for-face-emotion-recognition/versions/1", "sudarshanvaidya/random-images-for-face-emotion-recognition", "random-images-for-face-emotion-recognition", "GenericImageFolderAdaptor"),
    ("tusharpaul2001/face-emotion-mood-image-dataset/versions/1", "tusharpaul2001/face-emotion-mood-image-dataset", "face-emotion-mood-image-dataset", "GenericImageFolderAdaptor"),
    ("mh0386/facial-emotion/versions/93", "mh0386/facial-emotion", "facial-emotion", "GenericImageFolderAdaptor")
]

_here = Path(__file__).resolve()
_project_root = _here.parent.parent.parent
_default_json = _project_root / "data" / "datasets.json"

_OUTPUT_JSON = Path(_default_json)

def _setup_logging() -> None:
    """Configure logging with file rotation (24 hours)."""
    log_dir = _project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "dataset_registry.log"
    
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

def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def _read_existing_datasets(json_path: Path) -> dict:
    if not json_path.exists():
        return {"datasets": []}
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _write_datasets(json_path: Path, datasets: list[dict]) -> None:
    _ensure_parent_dir(json_path)
    # De-duplicate while preserving order
    seen = set()
    deduped = []
    for ds in datasets:
        key = (ds["file_path_str"], ds["kaggle_key"], ds["dataset_name"], ds["loader_class"])
        if key not in seen:
            seen.add(key)
            deduped.append(ds)
    
    output = {"datasets": deduped}
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

def generate_registry() -> None:
    # Configure global logging at DEBUG level
    logger = logging.getLogger(__name__)
    collected: list[dict] = []
    # Merge with existing rows to keep previous eval_results and update paths if needed
    existing_data = _read_existing_datasets(_OUTPUT_JSON)
    
    # Convert existing entries to a dict for easy lookup
    existing_map = {ds["dataset_name"]: ds for ds in existing_data.get("datasets", [])}
    
    for file_path, kaggle_key, dataset_name, loader_class in DATASETS:
        # Check if we already have this dataset
        if dataset_name in existing_map:
            collected.append(existing_map[dataset_name])
        
        try:
            local_path = kagglehub.dataset_download(file_path)
            file_path_str = str(Path(local_path).resolve())
            logger.info(f"Path to dataset files ({dataset_name}): {file_path_str}")
            print(f"Path to dataset files ({dataset_name}): {file_path_str}")
            collected.append({
                "dataset_name": dataset_name,
                "kaggle_key": kaggle_key,
                "file_path_str": file_path_str,
                "loader_class": loader_class
            })
        except Exception as e:
            # Record as missing to keep track; you can filter later
            logger.error(f"Failed to download {file_path}: {e}")
            print(f"Failed to download {file_path}: {e}")
            collected.append({
                "dataset_name": dataset_name,
                "kaggle_key": kaggle_key,
                "file_path_str": "",
                "loader_class": loader_class
            })

    _write_datasets(_OUTPUT_JSON, collected)
    logger.info(f"Wrote dataset index to: {_OUTPUT_JSON.resolve()}")
    print(f"Wrote dataset index to: {_OUTPUT_JSON.resolve()}")

    default_h5 = _project_root / "data" / "dataset.h5"

if __name__ == "__main__":
    _setup_logging()
    generate_registry()
