"""Visualization utilities for cloud segmentation results."""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils.constants import CLASS_NAMES


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    normalize: bool = True,
    class_names: List[str] = CLASS_NAMES,
    title: Optional[str] = None,
    cmap: str = "Blues",
    figsize: Tuple[int, int] = (8, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot a confusion matrix (absolute or row-normalized).

    Args:
        conf_matrix: Square confusion matrix (n_classes, n_classes).
        normalize: If True, converts each row to percentages.
        class_names: Labels for X and Y axes.
        title: Plot title. If None, a default title is used.
        cmap: Colormap for seaborn heatmap.
        figsize: Figure size if ax is not provided.
        ax: Target axes. If None, a new figure is created.

    Returns:
        The axes containing the confusion matrix plot.
    """
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            cm_show = (
                conf_matrix.astype(float)
                / conf_matrix.sum(axis=1, keepdims=True)
            )
        cm_show = np.nan_to_num(cm_show) * 100
        fmt = ".1f"
        cbar_label = "Percentage (%)"
    else:
        cm_show = conf_matrix
        fmt = "d"
        cbar_label = "Count"

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm_show,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": cbar_label},
        ax=ax,
    )

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")

    if title is None:
        title = "Confusion Matrix"
        if normalize:
            title += " (normalized)"

    ax.set_title(title)
    plt.tight_layout()

    return ax


def plot_training_history(
    history: dict,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot training history curves.

    Creates three subplots showing loss, accuracy, and IoU over epochs.

    Args:
        history: Dictionary with training history containing keys:
            train_loss, val_loss, train_acc, val_acc, train_iou, val_iou.
        figsize: Figure size.
        save_path: If provided, saves the figure to this path.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, history["train_acc"], "b-", label="Train")
    axes[1].plot(epochs, history["val_acc"], "r-", label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # IoU plot
    axes[2].plot(epochs, history["train_iou"], "b-", label="Train")
    axes[2].plot(epochs, history["val_iou"], "r-", label="Validation")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("IoU")
    axes[2].set_title("Training and Validation IoU")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_threshold_curve(
    thresholds: np.ndarray,
    median_boas: np.ndarray,
    best_threshold: float,
    best_boa: float,
    experiment: str,
    model_name: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot BOA optimization curve for threshold selection.

    Args:
        thresholds: Array of threshold values tested.
        median_boas: Array of median BOA values for each threshold.
        best_threshold: Optimal threshold value.
        best_boa: BOA value at optimal threshold.
        experiment: Name of the experiment.
        model_name: Optional model name for the title.
        figsize: Figure size.
        save_path: If provided, saves the figure to this path.
    """
    plt.figure(figsize=figsize)

    plt.plot(thresholds, median_boas, linewidth=2)
    plt.scatter(
        best_threshold,
        best_boa,
        s=100,
        zorder=5,
        label=f"t* = {best_threshold:.2f}",
    )

    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("Median BOA", fontsize=12)

    title = f"Threshold Optimization Curve - {experiment}"
    if model_name:
        title += f" - {model_name}"
    plt.title(title, fontsize=14)

    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()