"""Patch-level Balanced Overall Accuracy (BOA) evaluation."""

from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from cloudsen12.config.constants import EXPERIMENTS, SENTINEL_BANDS
from cloudsen12.inference.normalization import get_normalization_stats, normalize_images
from cloudsen12.inference.prediction import get_predictions


def safe_divide(numerator: float, denominator: float) -> float:
    """Return NaN when denominator is zero, otherwise numerator/denominator."""
    return np.nan if denominator == 0 else numerator / denominator


def compute_bucket_percentages(arr: np.ndarray) -> Tuple[float, float, float]:
    """Compute percentage of values in ranges <0.1, 0.1-0.9, >0.9.

    Args:
        arr: Array of values (NaN values are ignored).

    Returns:
        Tuple of (low_pct, mid_pct, high_pct) as percentages.
    """
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (0.0, 0.0, 0.0)
    hist, _ = np.histogram(arr, [0, 0.1, 0.9, 1.01])
    return tuple(hist / arr.size * 100)


def _prepare_models(
    models: Union[torch.nn.Module, List[torch.nn.Module]],
    device: str,
) -> List[torch.nn.Module]:
    """Ensure models is a list and move all to device in eval mode."""
    if not isinstance(models, list):
        models = [models]
    for m in models:
        m.to(device).eval()
    return models


def _compute_binary_stats(
    tp: int, fn: int, fp: int, tn: int, skip_cloud_check: bool = False
) -> Dict[str, float]:
    """Compute PA, UA, BOA, OE, CE from binary confusion counts."""
    pa = safe_divide(tp, tp + fn)
    ua = safe_divide(tp, tp + fp)

    if skip_cloud_check and (tp + fn) == 0:
        pa = np.nan
        ua = np.nan

    boa = 0.5 * (safe_divide(tp, tp + fn) + safe_divide(tn, tn + fp))
    oe = np.nan if np.isnan(pa) else (1.0 - pa)
    ce = np.nan if np.isnan(ua) else (1.0 - ua)

    return {"PA": pa, "UA": ua, "BOA": boa, "OE": oe, "CE": ce}


def _build_summary_table(exps: Dict[str, Dict]) -> pd.DataFrame:
    """Build summary DataFrame from accumulated experiment metrics."""
    rows = []
    for name, cfg in exps.items():
        pa_low, pa_mid, pa_high = compute_bucket_percentages(np.array(cfg["PA"]))
        ua_low, ua_mid, ua_high = compute_bucket_percentages(np.array(cfg["UA"]))

        rows.append({
            "Experiment": name,
            "Median BOA": f"{np.nanmedian(cfg['BOA']):.4f}",
            "PA low%": f"{pa_low:.2f}",
            "PA middle%": f"{pa_mid:.2f}",
            "PA high%": f"{pa_high:.2f}",
            "UA low%": f"{ua_low:.2f}",
            "UA middle%": f"{ua_mid:.2f}",
            "UA high%": f"{ua_high:.2f}",
            "N patches": len(cfg["BOA"]),
        })

    return pd.DataFrame(rows)


def evaluate_test_dataset(
    test_loader: torch.utils.data.DataLoader,
    models: Union[torch.nn.Module, List[torch.nn.Module]],
    device: str = "cuda",
    use_ensemble: bool = True,
    normalize_imgs: bool = True,
) -> pd.DataFrame:
    """Compute patch-level BOA using argmax predictions.

    Evaluates each binary experiment defined in EXPERIMENTS by computing
    per-patch PA, UA, BOA, OE, and CE, then summarizes with median BOA
    and PA/UA bucket distributions.

    Args:
        test_loader: DataLoader with test data.
        models: Single model or list of models.
        device: Device for execution.
        use_ensemble: If True, uses ensemble prediction.
        normalize_imgs: If True, normalizes images before inference.

    Returns:
        DataFrame with summary statistics for each experiment.
    """
    mean, std = get_normalization_stats(device, False, SENTINEL_BANDS)
    models = _prepare_models(models, device)

    exps: Dict[str, Dict] = {
        k: dict(v, PA=[], UA=[], BOA=[], OE=[], CE=[])
        for k, v in EXPERIMENTS.items()
    }

    with torch.no_grad():
        for imgs, gts in tqdm(test_loader, desc="Processing patches"):
            imgs = imgs.to(device).float()
            gts = gts.to(device)

            if normalize_imgs:
                imgs = normalize_images(imgs, mean, std)

            preds = get_predictions(models, imgs, use_ensemble=use_ensemble)

            for gt, pr in zip(gts.cpu().numpy(), preds.cpu().numpy()):
                cm = confusion_matrix(gt.ravel(), pr.ravel(), labels=[0, 1, 2, 3])

                for name, cfg in exps.items():
                    pos = cfg["pos"]
                    neg = [c for c in range(4) if c not in pos]

                    tp = cm[np.ix_(pos, pos)].sum()
                    fn = cm[np.ix_(pos, neg)].sum()
                    fp = cm[np.ix_(neg, pos)].sum()
                    tn = cm[np.ix_(neg, neg)].sum()

                    stats = _compute_binary_stats(
                        tp, fn, fp, tn,
                        skip_cloud_check=(name == "cloud/no cloud"),
                    )
                    for key in ("PA", "UA", "BOA", "OE", "CE"):
                        cfg[key].append(stats[key])

    return _build_summary_table(exps)