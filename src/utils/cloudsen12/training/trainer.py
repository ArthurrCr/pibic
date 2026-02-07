"""Training loop for multi-class segmentation models."""

import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.amp
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
from tqdm.notebook import tqdm

from cloudsen12.config.constants import NUM_CLASSES
from cloudsen12.training.callbacks import EarlyStopping


def _squeeze_masks(masks: torch.Tensor) -> torch.Tensor:
    """Remove singleton dimensions from mask tensors."""
    if masks.dim() == 4:
        if masks.shape[1] == 1:
            masks = masks.squeeze(1)
        elif masks.shape[-1] == 1:
            masks = masks.squeeze(-1)
    return masks


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-4,
    device: str = "cuda",
    checkpoint_dir: Optional[str] = None,
    resume_checkpoint: Optional[str] = None,
    save_best: bool = True,
    metric_to_monitor: str = "val_loss",
    mode: str = "min",
    patience: int = 5,
    min_delta: float = 1e-4,
    use_early_stopping: bool = True,
    loss_fn: Optional[nn.Module] = None,
    scheduler_factor: float = 0.1,
    scheduler_patience: int = 3,
) -> Dict[str, list]:
    """Train a multi-class segmentation model with mixed-precision support.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        num_epochs: Maximum number of training epochs.
        lr: Initial learning rate.
        device: Device string ("cuda" or "cpu").
        checkpoint_dir: Directory for saving checkpoints.
        resume_checkpoint: Path to checkpoint for resuming.
        save_best: If True, saves the best model checkpoint.
        metric_to_monitor: Metric for early stopping and checkpointing.
        mode: "min" or "max" for the monitored metric.
        patience: Early stopping patience.
        min_delta: Minimum improvement for early stopping.
        use_early_stopping: If True, enables early stopping.
        loss_fn: Loss function. Defaults to CrossEntropyLoss.
        scheduler_factor: Factor for ReduceLROnPlateau.
        scheduler_patience: Patience for ReduceLROnPlateau.

    Returns:
        Dictionary with training history (losses, accuracies, IoU, lr).
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    criterion = loss_fn.to(device) if loss_fn is not None else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode=mode, factor=scheduler_factor, patience=scheduler_patience
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

    history: Dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_iou": [], "val_iou": [],
        "lr": [],
    }

    if resume_checkpoint is not None and os.path.isfile(resume_checkpoint):
        print(f"Loading checkpoint: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_metric = ckpt.get("best_metric", best_metric)
        best_epoch = ckpt.get("best_epoch", best_epoch)
        history = ckpt.get("history", history)

        if early_stopping and "early_stopping" in ckpt:
            early_stopping.load_state_dict(ckpt["early_stopping"])
            best_str = (
                f"{early_stopping.best_metric:.4f}"
                if early_stopping.best_metric is not None
                else "N/A"
            )
            print(f"Early stopping: counter={early_stopping.counter}, best={best_str}")

        print(f"Resuming from epoch {start_epoch}, best_metric={best_metric:.4f}")

    train_acc_metric = MulticlassAccuracy(num_classes=NUM_CLASSES, average="macro").to(device)
    train_iou_metric = MulticlassJaccardIndex(num_classes=NUM_CLASSES, average="macro").to(device)
    val_acc_metric = MulticlassAccuracy(num_classes=NUM_CLASSES, average="macro").to(device)
    val_iou_metric = MulticlassJaccardIndex(num_classes=NUM_CLASSES, average="macro").to(device)

    use_amp = device.type == "cuda"

    for epoch in range(start_epoch, num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]

        # -- Training --
        model.train()
        epoch_train_loss = 0.0
        train_acc_metric.reset()
        train_iou_metric.reset()

        loop_train = tqdm(
            train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}] Train"
        )
        for images, masks in loop_train:
            images = images.to(device, non_blocking=True)
            masks = _squeeze_masks(masks).long().to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
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
            loop_train.set_postfix(loss=f"{loss_value.item():.4f}", lr=f"{current_lr:.2e}")

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_acc = train_acc_metric.compute().item()
        train_iou = train_iou_metric.compute().item()

        # -- Validation --
        model.eval()
        epoch_val_loss = 0.0
        val_acc_metric.reset()
        val_iou_metric.reset()

        loop_val = tqdm(val_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}] Val")
        with torch.no_grad():
            for images, masks in loop_val:
                images = images.to(device, non_blocking=True)
                masks = _squeeze_masks(masks).long().to(device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
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

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)
        history["lr"].append(current_lr)

        print(
            f"\nEpoch [{epoch + 1}/{num_epochs}] Summary:\n"
            f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n"
            f"  Train Acc:  {train_acc:.4f} | Val Acc:  {val_acc:.4f}\n"
            f"  Train IoU:  {train_iou:.4f} | Val IoU:  {val_iou:.4f}\n"
            f"  LR:         {current_lr:.2e}"
        )

        metric_map = {
            "val_loss": avg_val_loss,
            "val_iou": val_iou,
            "val_acc": val_acc,
        }
        current_metric = metric_map.get(metric_to_monitor, avg_val_loss)
        scheduler.step(current_metric)

        is_better = (
            (mode == "max" and current_metric > best_metric)
            or (mode == "min" and current_metric < best_metric)
        )

        if is_better:
            best_metric = current_metric
            best_epoch = epoch + 1

            if save_best and checkpoint_dir:
                _save_checkpoint(
                    os.path.join(checkpoint_dir, "best_model.pth"),
                    epoch, model, optimizer, scaler,
                    best_metric, best_epoch, history, early_stopping,
                )
                print(
                    f"[Improvement] Saved best model: "
                    f"{metric_to_monitor}={best_metric:.4f} at epoch {best_epoch}"
                )

        if checkpoint_dir:
            _save_checkpoint(
                os.path.join(checkpoint_dir, "checkpoint.pth"),
                epoch, model, optimizer, scaler,
                best_metric, best_epoch, history, early_stopping,
            )

        if use_early_stopping and early_stopping is not None:
            early_stopping(current_metric)
            print(f"Early stopping: {early_stopping.counter}/{early_stopping.patience}")
            if early_stopping.early_stop:
                print(
                    f"Early stopping triggered at epoch {epoch + 1}. "
                    f"Best {metric_to_monitor}: {best_metric:.4f} at epoch {best_epoch}"
                )
                break

        if use_amp and torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        f"Training completed. Best {metric_to_monitor}: "
        f"{best_metric:.4f} at epoch {best_epoch}"
    )
    return history


def _save_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    best_metric: float,
    best_epoch: int,
    history: Dict[str, list],
    early_stopping: Optional[EarlyStopping],
) -> None:
    """Persist training state to disk."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "history": history,
            "early_stopping": (
                early_stopping.state_dict() if early_stopping else None
            ),
        },
        path,
    )