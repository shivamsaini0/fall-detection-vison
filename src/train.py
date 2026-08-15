#!/usr/bin/env python3
"""Training loop for the fall detection LSTM classifier."""

import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.fall_detector import FallDetector
from src.dataset import create_dataloaders, load_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate model. Returns (avg_loss, accuracy, macro_f1)."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item() * len(y_batch)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    total = len(all_labels)
    accuracy = sum(p == label for p, label in zip(all_preds, all_labels)) / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return total_loss / total, accuracy, macro_f1


def train(
    data_dir: str = "data/splits",
    checkpoint_dir: str = "models/checkpoints",
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    seed: int = 42,
    device_name: str | None = None,
):
    """Full training pipeline."""
    set_seed(seed)

    # Device
    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    logger.info(f"Device: {device}")

    # Data
    logger.info(f"Loading data from {data_dir}...")
    splits = load_splits(data_dir)
    loaders = create_dataloaders(splits, batch_size=batch_size, num_workers=0)

    logger.info(
        f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}"
    )

    # Compute class weights
    y_train = splits["train"].y.numpy()
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = torch.FloatTensor(weights).to(device)
    logger.info(f"Class weights: {dict(zip(classes.tolist(), [f'{w:.3f}' for w in weights]))}")

    # Model
    model = FallDetector().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    # Training setup
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # Training loop
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    history = []
    start_time = time.time()

    logger.info(f"\nStarting training for up to {epochs} epochs...")
    logger.info("-" * 80)

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, val_f1 = evaluate(model, loaders["val"], criterion, device)

        # Scheduler step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Log
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "lr": current_lr,
        })

        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
            }, checkpoint_path / "best.pth")
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            logger.info(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    training_time = time.time() - start_time

    # Save last checkpoint
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_acc": val_acc,
    }, checkpoint_path / "last.pth")

    # Final evaluation on test set
    logger.info("\n" + "=" * 80)
    logger.info("Final evaluation on test set...")

    # Load best model
    best_ckpt = torch.load(checkpoint_path / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_loss, test_acc, test_f1 = evaluate(model, loaders["test"], criterion, device)
    logger.info(f"Test Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | Macro F1: {test_f1:.4f}")

    # Per-class metrics
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loaders["test"]:
            X_batch = X_batch.to(device)
            preds = model(X_batch).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    from sklearn.metrics import classification_report, confusion_matrix

    class_names = ["normal", "falling", "fallen"]
    report = classification_report(all_labels, all_preds, target_names=class_names)
    cm = confusion_matrix(all_labels, all_preds)

    logger.info(f"\nClassification Report:\n{report}")
    logger.info(f"Confusion Matrix:\n{cm}")

    # Save training report
    report_dict = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True
    )
    training_report = {
        "training_time_seconds": round(training_time, 1),
        "total_epochs": epoch,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "test_accuracy": round(test_acc, 4),
        "test_macro_f1": round(test_f1, 4),
        "per_class_metrics": report_dict,
        "confusion_matrix": cm.tolist(),
        "history": history,
    }

    report_path = checkpoint_path / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(training_report, f, indent=2, default=str)
    logger.info(f"\nTraining report saved to {report_path}")

    logger.info(f"\nTraining complete in {training_time:.1f}s")
    logger.info(f"Best model (epoch {best_epoch}) saved to {checkpoint_path / 'best.pth'}")

    return model, training_report


if __name__ == "__main__":
    train()
