import csv
import logging
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from eval.ResEmoteNet.approach.ResEmoteNet import ResEmoteNet
from component.dataset.dataset import H5LabeledImageDataset

# ------------------------------
# Configuration
# ------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
DATA_ROOT = (PROJECT_ROOT / "data").resolve()
DATASETS_DIR = (DATA_ROOT / "h5_datasets").resolve()
EVALUATION_DATA_ROOT = (DATA_ROOT / "evaluation").resolve()
RES_EMOTE_NET_DATA = (EVALUATION_DATA_ROOT / "ResEmoteNet").resolve()
WEIGHTS_DIR = (DATA_ROOT / "weights").resolve()
RESULTS_DIR = (RES_EMOTE_NET_DATA / "results").resolve()

# Initialize logging if not already configured: daily rotating file in ../logs at INFO level
if not logging.getLogger().handlers:
    logs_dir = (PROJECT_ROOT / "logs").resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "ResEmoteNet_eval.log"

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    file_handler = TimedRotatingFileHandler(str(log_file), when="midnight", backupCount=14, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

MODEL_NAME = "ResEmoteNet"
BATCH_SIZE = 128
NUM_WORKERS = 2
IMAGE_SIZE = (64, 64)


def _now_local_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _epoch_seconds() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp())


def _list_h5_files() -> List[Path]:
    if not DATASETS_DIR.exists():
        return []
    return sorted([p for p in DATASETS_DIR.iterdir() if p.suffix.lower() == ".h5"])


def _load_weights(model: torch.nn.Module, weights_dir: Path) -> Path:
    # Prefer the newest .pth in weights dir
    candidates = sorted(weights_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No .pth weight files found in {weights_dir}")
    weight_path = candidates[0]
    state_dict = torch.load(weight_path, map_location="cuda")

    # Allow checkpoints saved as dicts like {state_dict: ...} or raw state_dict
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        raise ValueError(f"Failed to load weights from {weight_path}: {e}")
    return weight_path


def _get_transforms():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _init_csv(outfile: Path, header: List[str]):
    # Create and write header if file doesn't exist
    if not outfile.exists():
        with outfile.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()


def _append_row(outfile: Path, row: Dict[str, object], header: List[str]):
    with outfile.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerow(row)


def _update_macro_metrics(stats: Dict[str, torch.Tensor], preds: torch.Tensor, targets: torch.Tensor, num_classes: int):
    # Build confusion components per class
    # stats will keep tp, fp, fn per class across batches
    if not stats:
        stats["tp"] = torch.zeros(num_classes, dtype=torch.long)
        stats["fp"] = torch.zeros(num_classes, dtype=torch.long)
        stats["fn"] = torch.zeros(num_classes, dtype=torch.long)
        stats["n"] = torch.zeros(1, dtype=torch.long)
        stats["correct"] = torch.zeros(1, dtype=torch.long)

    preds = preds.view(-1)
    targets = targets.view(-1)

    stats["n"] += targets.numel()
    stats["correct"] += (preds == targets).sum()

    for c in range(num_classes):
        c_preds = preds == c
        c_tgts = targets == c
        tp = (c_preds & c_tgts).sum()
        fp = (c_preds & (~c_tgts)).sum()
        fn = ((~c_preds) & c_tgts).sum()
        stats["tp"][c] += tp
        stats["fp"][c] += fp
        stats["fn"][c] += fn


def _finalize_metrics(stats: Dict[str, torch.Tensor]) -> Tuple[float, float, float, float]:
    n = int(stats["n"].item()) if "n" in stats else 0
    correct = int(stats["correct"].item()) if "correct" in stats else 0
    accuracy = (correct / n) if n else 0.0

    tp = stats["tp"].to(torch.float32)
    fp = stats["fp"].to(torch.float32)
    fn = stats["fn"].to(torch.float32)

    with torch.no_grad():
        precision_per_class = tp / torch.clamp(tp + fp, min=1.0)
        recall_per_class = tp / torch.clamp(tp + fn, min=1.0)
        f1_per_class = 2 * precision_per_class * recall_per_class / torch.clamp(precision_per_class + recall_per_class, min=1e-6)

        precision = float(precision_per_class.mean().item())
        recall = float(recall_per_class.mean().item())
        f1 = float(f1_per_class.mean().item())

    return float(accuracy), recall, precision, f1


def evaluate():
    logger.info("Starting evaluation run ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using {device} device")

    # Model & weights
    model = ResEmoteNet().to(device)
    weight_path = _load_weights(model, WEIGHTS_DIR)
    logger.info(f"Loaded weights from: {weight_path}")
    model.eval()

    # Session and CSV setup
    session_id = _epoch_seconds()
    epoch_sec = session_id
    logger.info(f"Session ID: {session_id}")

    # Per-image CSV in current working directory
    per_image_filename = f"{MODEL_NAME}_per_image_{session_id}.csv"
    per_image_csv = RESULTS_DIR / per_image_filename
    per_image_header = [
        "session_id",
        "datetime",
        "dataset",
        "image_index",
        "label",
        "predicted",
        "model",
        "correct",
    ]
    print("Per image report:", per_image_csv)
    _init_csv(per_image_csv, per_image_header)
    logger.info(f"Per-image CSV will be written to: {per_image_csv}")

    # Results directory and overall summary CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    session_summary_csv = (RESULTS_DIR / f"session_summary_{session_id}.csv").resolve()
    dataset_summary_csv = (RESULTS_DIR / f"dataset_summary_{session_id}.csv").resolve()

    overall_header = ["date", "model", "session_id", "accuracy", "recall", "precision", "f1"]
    dataset_header = ["date", "model", "dataset", "session_id", "accuracy", "recall", "precision", "f1"]
    _init_csv(session_summary_csv, dataset_header)
    _init_csv(dataset_summary_csv, dataset_header)

    logger.info(f"Session summary CSV: {session_summary_csv}")
    logger.info(f"Dataset summary CSV: {dataset_summary_csv}")

    # Iterate datasets
    h5_files = _list_h5_files()
    if not h5_files:
        raise FileNotFoundError(f"No .h5 datasets found in {DATASETS_DIR}")
    logger.info(f"Found {len(h5_files)} dataset(s) in {DATASETS_DIR}")
    for p in h5_files:
        logger.info(f"Dataset: {p}")

    all_stats: Dict[str, torch.Tensor] = {}

    transform = _get_transforms()

    for h5 in h5_files:
        logger.info(f"Evaluating dataset: {h5}")
        # Create a mirrored path under eval_results (e.g., eval_results/data/h5_datasets/<file-stem>/metrics.csv)
        mirrored_root = RESULTS_DIR / h5.parent.relative_to(PROJECT_ROOT)
        dataset_dir = mirrored_root / h5.stem
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_metrics_csv = (RESULTS_DIR / (h5.name + str(session_id) + ".csv")).resolve()
        _init_csv(dataset_metrics_csv, dataset_header)

        # Prepare dataset/dataloader
        ds = H5LabeledImageDataset(h5, transform=transform, return_index=True)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS,
                            pin_memory=torch.cuda.is_available())
        logger.info(f"DataLoader prepared: batch_size={BATCH_SIZE}, num_workers={NUM_WORKERS}")

        # Infer number of classes from the model output size (assumes fixed 7)
        num_classes = 7

        dataset_stats: Dict[str, torch.Tensor] = {}

        with torch.no_grad():
            for batch in loader:
                if len(batch) == 3:
                    images, targets, indices = batch  # type: ignore[assignment]
                else:
                    images, targets = batch  # type: ignore[misc]
                    # If indices missing, fall back to a simple range (best-effort)
                    indices = torch.arange(start=0, end=targets.shape[0])

                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                logits = model(images)
                preds = torch.argmax(logits, dim=1)

                # Update stats
                _update_macro_metrics(dataset_stats, preds.cpu(), targets.cpu(), num_classes)
                _update_macro_metrics(all_stats, preds.cpu(), targets.cpu(), num_classes)

                # Write per-image rows
                for idx, tgt, pred in zip(indices.tolist(), targets.cpu().tolist(), preds.cpu().tolist()):
                    _append_row(
                        per_image_csv,
                        {
                            "session_id": session_id,
                            "datetime": _now_local_iso(),
                            "image_index": idx,
                            "label": int(tgt),
                            "predicted": int(pred),
                            "model": MODEL_NAME,
                            "correct": bool(int(tgt) == int(pred)),
                        },
                        per_image_header,
                    )

        # Finalize dataset metrics and append row
        d_acc, d_rec, d_prec, d_f1 = _finalize_metrics(dataset_stats)
        _append_row(
            dataset_metrics_csv,
            {
                "date": _now_local_iso(),
                "model": MODEL_NAME,
                "session_id": session_id,
                "dataset": h5.name,
                "accuracy": f"{d_acc:.6f}",
                "recall": f"{d_rec:.6f}",
                "precision": f"{d_prec:.6f}",
                "f1": f"{d_f1:.6f}",
            },
            dataset_header,
        )
        _append_row(
            dataset_summary_csv,
            {
                "date": _now_local_iso(),
                "model": MODEL_NAME,
                "dataset": h5.name,
                "session_id": session_id,
                "accuracy": f"{d_acc:.6f}",
                "recall": f"{d_rec:.6f}",
                "precision": f"{d_prec:.6f}",
                "f1": f"{d_f1:.6f}",
            },
            dataset_header,
        )
        logger.info(f"Completed {h5.name}: acc={d_acc:.4f}, recall={d_rec:.4f}, precision={d_prec:.4f}, f1={d_f1:.4f}")

    # Overall metrics across all datasets
    acc, rec, prec, f1 = _finalize_metrics(all_stats)
    _append_row(
        session_summary_csv,
        {
            "date": _now_local_iso(),
            "model": MODEL_NAME,
            "session_id": session_id,
            "accuracy": f"{acc:.6f}",
            "recall": f"{rec:.6f}",
            "precision": f"{prec:.6f}",
            "f1": f"{f1:.6f}",
        },
        overall_header,
    )
    logger.info(f"Overall metrics: acc={acc:.4f}, recall={rec:.4f}, precision={prec:.4f}, f1={f1:.4f}")
    logger.info(f"Results root: {RESULTS_DIR}")

if __name__ == "__main__":
    evaluate()
