import logging
import time
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from tqdm import tqdm
import os  # Import os module
from evaluation.ResEmoteNet.approach.ResEmoteNet import ResEmoteNet
from component.dataset.split_utils import make_h5_loaders_stratified

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_ROOT = (PROJECT_ROOT / "data").resolve()
H5_ROOT = (DATA_ROOT / "h5_datasets").resolve()
TRAINING_ROOT = (DATA_ROOT / "training").resolve()
RES_EMOTE_NET_DATA = (TRAINING_ROOT / "ResEmoteNet").resolve()
WEIGHTS_ROOT = (RES_EMOTE_NET_DATA / "weights").resolve()

TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
RES_EMOTE_NET_DATA.mkdir(parents=True, exist_ok=True)
WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)

h5_paths = [(H5_ROOT / "emotion-detection-fer.h5").resolve(),
           (H5_ROOT / "8-facial-expressions-for-yolo.h5").resolve(),
           (H5_ROOT / "face-emotion-mood-image-dataset.h5").resolve()]

if not logging.getLogger().handlers:
    logs_dir = (PROJECT_ROOT / "logs").resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "ResEmoteNet_train.log"

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using {device} device")

# Transform the dataset
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load the model
model = ResEmoteNet().to(device)

# Print the number of parameters
total_params = sum(p.numel() for p in model.parameters())
logger.info(f"{total_params:,} total parameters.")

# Hyperparameters
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

# Early stopping after 5 epochs with no improvement
patience = 5
best_val_acc = 0.0
patience_counter = 0
epoch_counter = 0

num_epochs = 80

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
test_losses = []
test_accuracies = []

# Start training
start_time = time.time()
session_id = int(start_time)
logger.info(f"Executing {num_epochs} epochs...")
dataset_loaders = []

for h5_path in h5_paths:
    logger.info(f"Creating loaders for {h5_path}")
    train_loader, val_loader, test_loader, split_file = make_h5_loaders_stratified(
        h5_path=h5_path,
        batch_size=64,
        seed=123,  # change to get a different split
        ratios=(0.8, 0.1, 0.1),  # adjust if needed
        split_id="run1",  # optional label for saving/loading
        num_workers=int(os.getenv("DATALOADER_WORKERS", "0")),  # default 0 for stability
        transform=transform,  # ensure worker gets proper transform
    )
    dataset_loaders.append((train_loader, val_loader, test_loader, split_file))
    logger.info(f"Using split file: {split_file}")

    # Log basic split sizes to ensure non-empty loaders
    try:
        _train_len = len(train_loader.dataset)  # type: ignore[attr-defined]
        _val_len = len(val_loader.dataset)  # type: ignore[attr-defined]
        _test_len = len(test_loader.dataset)  # type: ignore[attr-defined]
        logger.info(f"Split sizes -> train: {_train_len}, val: {_val_len}, test: {_test_len}")
    except Exception as e:
        logger.exception(f"Failed to determine dataset sizes for {h5_path}: {e}")
        pass

    # Peek one batch from each loader to get shapes; surface decoding errors
    try:
        train_image, train_label = next(iter(train_loader))
        val_image, val_label = next(iter(val_loader))
        test_image, test_label = next(iter(test_loader))
        logger.info(f"Train batch: Image shape {tuple(train_image.shape)}, Label shape {tuple(train_label.shape)}")
        logger.info(f"Validation batch: Image shape {tuple(val_image.shape)}, Label shape {tuple(val_label.shape)}")
        logger.info(f"Test batch: Image shape {tuple(test_image.shape)}, Label shape {tuple(test_label.shape)}")
    except Exception as e:
        logger.exception(f"Failed to fetch initial batch from loaders for {h5_path}: {e}")
        raise

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for train_loader, val_loader, test_loader, split_file in dataset_loaders:
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        test_running_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_loss = test_running_loss / len(test_loader)
        test_acc = test_correct / test_total
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Step LR scheduler on validation loss plateau
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        epoch_time = int(time.time() - epoch_start_time)
        logger.info(
            f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}"
        )
        epoch_counter += 1

        # Capture epoch timestamp for unique artifacts and row key
        epoch_id = int(time.time())

        # We only want one set of weights stored per run.  This will be referenced multiple times by epoch records
        checkpoint_path = (WEIGHTS_ROOT / f"ResEmoteNetWeights_{session_id}.pth").resolve()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint_path = checkpoint_path
            torch.save(model.state_dict(), str(checkpoint_path))
            logger.info(f"Validation accuracy improved to {best_val_acc:.4f}. Saved {checkpoint_path}")
        else:
            patience_counter += 1
            logger.info(f"No improvement in validation accuracy for {patience_counter}/{patience} epochs.")

        # Append per-epoch metrics row to CSV, keyed by epoch_ts
        try:
            csv_path = (RES_EMOTE_NET_DATA / "ResEmoteNetTraining.csv").resolve()
            row = {
                "id": epoch_id,
                "epoch": epoch + 1,
                "epoch_time": float(epoch_time),
                "total_time": float(time.time() - start_time),
                "train_loss": float(train_loss),
                "test_loss": float(test_loss),
                "val_loss": float(val_loss),
                "train_acc": float(train_acc),
                "test_acc": float(test_acc),
                "val_acc": float(val_acc),
                "lr": float(current_lr),
                "split_file": str(split_file),
                "checkpoint": checkpoint_path,
            }
            if csv_path.exists():
                prev = pd.read_csv(csv_path)
                prev = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
                prev.to_csv(csv_path, index=False, mode="w")
            else:
                pd.DataFrame([row]).to_csv(csv_path, index=False, mode="w")
            logger.info(f"Appended metrics for epoch {epoch+1} with key {epoch_id} to {csv_path.name}")
        except Exception as e:
            logger.exception(f"Failed to append metrics CSV: {e}")

        if patience_counter >= patience:
            logger.warning("Early stopping: no improvement in validation accuracy.")
            break

logger.info(f"Training complete in {time.time() - start_time}s. Metrics saved to ResEmoteNetTraining.csv")