"""Metrics computation for cloud segmentation evaluation."""

from typing import Dict, List, Optional, Union

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from utils.constants import CLASS_NAMES, SENTINEL_BANDS
from utils.inference import get_normalization_stats, get_predictions, normalize_images


def compute_metrics(conf_matrix: np.ndarray) -> Dict:
    """
    Compute per-class metrics from a confusion matrix.

    Calculates F1-Score, Precision, Recall, Omission Error, and
    Commission Error for each class, plus overall accuracy.

    Args:
        conf_matrix: Confusion matrix with shape (n_classes, n_classes).

    Returns:
        Dictionary containing metrics for each class and overall statistics.
    """
    metrics = {}

    for i, name in enumerate(CLASS_NAMES):
        tp = conf_matrix[i, i]
        fn = conf_matrix[i, :].sum() - tp
        fp = conf_matrix[:, i].sum() - tp

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)

        metrics[name] = {
            "F1-Score": f1,
            "Precision": precision,
            "Recall": recall,
            "Omission Error": fn / (tp + fn + 1e-7),
            "Commission Error": fp / (tp + fp + 1e-7),
            "Support": conf_matrix[i, :].sum(),
        }

    metrics["Overall"] = {
        "Accuracy": np.diag(conf_matrix).sum() / conf_matrix.sum(),
        "Total Samples": conf_matrix.sum(),
    }

    return metrics


def evaluate_model(
    test_loader: torch.utils.data.DataLoader,
    models: Union[torch.nn.Module, List[torch.nn.Module]],
    device: str = "cuda",
    use_ensemble: bool = True,
    normalize_imgs: bool = True,
) -> np.ndarray:
    """
    Evaluate models and return aggregated confusion matrix.

    Args:
        test_loader: DataLoader with test data.
        models: Single model or list of models.
        device: Device for execution.
        use_ensemble: If True, uses ensemble prediction.
        normalize_imgs: If True, normalizes images before inference.

    Returns:
        Confusion matrix with shape (4, 4).
    """
    mean, std = get_normalization_stats(device, False, SENTINEL_BANDS)

    if not isinstance(models, list):
        models = [models]
    for m in models:
        m.to(device).eval()

    conf_matrix = np.zeros((4, 4), dtype=np.int64)
    print("Starting evaluation...")

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Processing batches"):
            images = images.to(device).float()
            labels = labels.to(device)

            if normalize_imgs:
                images = normalize_images(images, mean, std)

            preds = get_predictions(models, images, use_ensemble=use_ensemble)

            batch_conf = confusion_matrix(
                labels.cpu().numpy().ravel(),
                preds.cpu().numpy().ravel(),
                labels=[0, 1, 2, 3],
            )
            conf_matrix += batch_conf
            torch.cuda.empty_cache()

    return conf_matrix