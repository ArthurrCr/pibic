"""Training utilities for cloud segmentation models."""

import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.amp
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
from tqdm.notebook import tqdm

from utils.constants import NUM_CLASSES


class EarlyStopping:
    """
    Early stopping callback for training.

    Monitors a metric and stops training if no improvement is seen
    for a specified number of epochs.

    Attributes:
        patience: Number of epochs to wait for improvement.
        mode: "max" if higher is better, "min" if lower is better.
        min_delta: Minimum change to qualify as an improvement.
        counter: Current count of epochs without improvement.
        best_metric: Best metric value seen so far.
        early_stop: Flag indicating whether to stop training.
    """

    def __init__(
        self,
        patience: int = 3,
        mode: str = "max",
        min_delta: float = 1e-4,
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping.
            mode: "max" for metrics where higher is better (e.g., accuracy),
                "min" for metrics where lower is better (e.g., loss).
            min_delta: Minimum change to count as improvement.
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric: Optional[float] = None
        self.early_stop = False

    def __call__(self, current_metric: float) -> None:
        """
        Check if training should stop.

        Args:
            current_metric: Current value of the monitored metric.
        """
        if self.best_metric is None:
            self.best_metric = current_metric
            return

        if self.mode == "max":
            improvement = current_metric - self.best_metric
        else:
            improvement = self.best_metric - current_metric

        if improvement < self.min_delta:
            self.counter += 1
        else:
            self.best_metric = current_metric
            self.counter = 0

        if self.counter >= self.patience:
            self.early_stop = True


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int = 10,
    lr: float = 1e-4,
    device: str = "cuda",
    checkpoint_dir: Optional[str] = None,
    resume_checkpoint: Optional[str] = None,
    save_best: bool = True,
    metric_to_monitor: str = "val_loss",
    mode: str = "min",
    patience: int = 3,
    min_delta: float = 1e-4,
    use_early_stopping: bool = True,
) -> Dict[str, list]:
    """
    Train a multiclass segmentation model.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        num_epochs: Maximum number of training epochs.
        lr: Initial learning rate.
        device: Device for training ("cuda" or "cpu").
        checkpoint_dir: Directory to save checkpoints.
        resume_checkpoint: Path to checkpoint to resume from.
        save_best: If True, saves the best model.
        metric_to_monitor: Metric name for scheduling and early stopping.
        mode: "min" if lower is better, "max" if higher is better.
        patience: Patience for early stopping.
        min_delta: Minimum improvement for early stopping.
        use_early_stopping: If True, enables early stopping.

    Returns:
        Dictionary containing training history with losses and metrics.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode=mode, factor=0.1, patience=3, verbose=True
    )
    scaler = torch.amp.GradScaler()

    early_stopping = (
        EarlyStopping(patience=patience, mode=mode, min_delta=min_delta)
        if use_early_stopping
        else None
    )

    best_metric = -np.inf if mode == "max" else np.inf
    best_epoch = -1
    start_epoch = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_iou": [],
        "val_iou": [],
    }

    # Resume from checkpoint if specified
    if resume_checkpoint is not None and os.path.isfile(resume_checkpoint):
        print(f"Loading checkpoint '{resume_checkpoint}'")
        checkpoint = torch.load(
            resume_checkpoint, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_metric = checkpoint.get("best_metric", best_metric)
        best_epoch = checkpoint.get("best_epoch", best_epoch)
        history = checkpoint.get("history", history)
        print(f"Resuming from epoch {start_epoch} with best_metric={best_metric:.4f}")

    # Metrics
    train_acc_metric = MulticlassAccuracy(
        num_classes=NUM_CLASSES, average="macro"
    ).to(device)
    train_iou_metric = MulticlassJaccardIndex(
        num_classes=NUM_CLASSES, average="macro"
    ).to(device)
    val_acc_metric = MulticlassAccuracy(
        num_classes=NUM_CLASSES, average="macro"
    ).to(device)
    val_iou_metric = MulticlassJaccardIndex(
        num_classes=NUM_CLASSES, average="macro"
    ).to(device)

    use_amp = device.type == "cuda"

    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        train_acc_metric.reset()
        train_iou_metric.reset()

        loop_train = tqdm(
            train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}] (Train)"
        )

        for images, masks in loop_train:
            images = images.to(device, non_blocking=True)

            if masks.dim() == 4:
                if masks.shape[1] == 1:
                    masks = masks.squeeze(1)
                elif masks.shape[-1] == 1:
                    masks = masks.squeeze(-1)

            masks = masks.long().to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss_value = criterion(outputs, masks)

            if use_amp:
                scaler.scale(loss_value).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_value.backward()
                optimizer.step()

            epoch_train_loss += loss_value.item()

            preds = torch.argmax(outputs, dim=1)
            train_acc_metric.update(preds, masks)
            train_iou_metric.update(preds, masks)

            loop_train.set_postfix(loss=f"{loss_value.item():.4f}")

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_acc = train_acc_metric.compute().item()
        train_iou = train_iou_metric.compute().item()

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        val_acc_metric.reset()
        val_iou_metric.reset()

        loop_val = tqdm(val_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}] (Val)")

        with torch.no_grad():
            for images, masks in loop_val:
                images = images.to(device, non_blocking=True)

                if masks.dim() == 4:
                    if masks.shape[1] == 1:
                        masks = masks.squeeze(1)
                    elif masks.shape[-1] == 1:
                        masks = masks.squeeze(-1)

                masks = masks.long().to(device, non_blocking=True)

                outputs = model(images)
                loss_value = criterion(outputs, masks)
                epoch_val_loss += loss_value.item()

                preds = torch.argmax(outputs, dim=1)
                val_acc_metric.update(preds, masks)
                val_iou_metric.update(preds, masks)

                loop_val.set_postfix(loss=f"{loss_value.item():.4f}")

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_acc = val_acc_metric.compute().item()
        val_iou = val_iou_metric.compute().item()

        # Update history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)

        print(
            f"\nEpoch [{epoch + 1}/{num_epochs}] Summary:\n"
            f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n"
            f"  Train Acc:  {train_acc:.4f} | Val Acc:  {val_acc:.4f}\n"
            f"  Train IoU:  {train_iou:.4f} | Val IoU:  {val_iou:.4f}\n"
        )

        # Get current metric for monitoring
        metric_map = {
            "val_loss": avg_val_loss,
            "val_iou": val_iou,
            "val_acc": val_acc,
        }
        current_metric = metric_map.get(
            metric_to_monitor, history[metric_to_monitor][-1]
        )

        scheduler.step(current_metric)

        # Check for improvement
        is_better = (
            (mode == "max" and current_metric > best_metric)
            or (mode == "min" and current_metric < best_metric)
        )

        if is_better:
            best_metric = current_metric
            best_epoch = epoch + 1

            if save_best and checkpoint_dir:
                best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "best_metric": best_metric,
                        "best_epoch": best_epoch,
                        "history": history,
                    },
                    best_model_path,
                )
                print(
                    f"[Improvement] Model saved with "
                    f"{metric_to_monitor}: {best_metric:.4f} at epoch {epoch + 1}."
                )

        # Save current checkpoint
        if checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_metric": best_metric,
                    "best_epoch": best_epoch,
                    "history": history,
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved to '{checkpoint_path}'.")

        # Early stopping check
        if use_early_stopping and early_stopping is not None:
            early_stopping(current_metric)
            if early_stopping.early_stop:
                print(
                    f"Early stopping triggered at epoch {epoch + 1}. "
                    f"Best {metric_to_monitor}: {early_stopping.best_metric:.4f}"
                )
                break

        # Clear GPU cache
        if use_amp and torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        f"Training completed. Best {metric_to_monitor}: "
        f"{best_metric:.4f} at epoch {best_epoch}."
    )

    return history