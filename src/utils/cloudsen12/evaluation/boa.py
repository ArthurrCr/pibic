"""Binary Overall Accuracy (BOA) metrics and threshold optimization."""

from typing import Dict, List, Optional, Tuple, Union

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


def _build_summary_table(
    exps: Dict[str, Dict], include_threshold: bool = False
) -> pd.DataFrame:
    """Build summary DataFrame from accumulated experiment metrics."""
    rows = []
    for name, cfg in exps.items():
        pa_low, pa_mid, pa_high = compute_bucket_percentages(np.array(cfg["PA"]))
        ua_low, ua_mid, ua_high = compute_bucket_percentages(np.array(cfg["UA"]))

        row = {
            "Experiment": name,
            "Median BOA": f"{np.nanmedian(cfg['BOA']):.4f}",
            "PA low%": f"{pa_low:.2f}",
            "PA middle%": f"{pa_mid:.2f}",
            "PA high%": f"{pa_high:.2f}",
            "UA low%": f"{ua_low:.2f}",
            "UA middle%": f"{ua_mid:.2f}",
            "UA high%": f"{ua_high:.2f}",
            "N patches": len(cfg["BOA"]),
        }

        if include_threshold and "threshold" in cfg:
            row["Threshold"] = f"{cfg['threshold']:.2f}"

        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_test_dataset(
    test_loader: torch.utils.data.DataLoader,
    models: Union[torch.nn.Module, List[torch.nn.Module]],
    device: str = "cuda",
    use_ensemble: bool = True,
    normalize_imgs: bool = True,
) -> pd.DataFrame:
    """Compute patch-level PA, UA, BOA, OE and CE for binary experiments.

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


def evaluate_test_dataset_with_thresholds(
    test_loader: torch.utils.data.DataLoader,
    models: Union[torch.nn.Module, List[torch.nn.Module]],
    t_star: Dict[str, float],
    device: str = "cuda",
    use_ensemble: bool = True,
    normalize_imgs: bool = True,
) -> pd.DataFrame:
    """Evaluate binary experiments using provided optimal thresholds.

    Args:
        test_loader: DataLoader with test data.
        models: Single model or list of models.
        t_star: Mapping of experiment names to optimal thresholds.
        device: Device for execution.
        use_ensemble: If True, uses ensemble prediction.
        normalize_imgs: If True, normalizes images before inference.

    Returns:
        DataFrame with summary statistics for each experiment.
    """
    mean, std = get_normalization_stats(device, False, SENTINEL_BANDS)
    models = _prepare_models(models, device)

    exps: Dict[str, Dict] = {}
    for name, cfg in EXPERIMENTS.items():
        if name in t_star:
            exps[name] = dict(
                cfg,
                threshold=t_star[name],
                PA=[], UA=[], BOA=[], OE=[], CE=[],
            )

    with torch.no_grad():
        for imgs, gts in tqdm(test_loader, desc="Processing patches"):
            imgs = imgs.to(device).float()
            gts = gts.to(device)

            if normalize_imgs:
                imgs = normalize_images(imgs, mean, std)

            probs = get_predictions(
                models, imgs, use_ensemble=use_ensemble, return_probs=True
            )

            for i in range(imgs.size(0)):
                gt = gts[i].cpu().numpy()

                for name, cfg in exps.items():
                    pos = cfg["pos"]
                    neg = [c for c in range(4) if c not in pos]
                    t = cfg["threshold"]

                    ppos = probs[i, pos].sum(0).cpu().numpy()
                    pred_pos = ppos >= t

                    gt_pos = np.isin(gt, pos)
                    gt_neg = np.isin(gt, neg)

                    tp = np.logical_and(pred_pos, gt_pos).sum()
                    fn = np.logical_and(~pred_pos, gt_pos).sum()
                    fp = np.logical_and(pred_pos, gt_neg).sum()
                    tn = np.logical_and(~pred_pos, gt_neg).sum()

                    stats = _compute_binary_stats(
                        tp, fn, fp, tn,
                        skip_cloud_check=(name == "cloud/no cloud" and not gt_pos.any()),
                    )
                    for key in ("PA", "UA", "BOA", "OE", "CE"):
                        cfg[key].append(stats[key])

    return _build_summary_table(exps, include_threshold=True)


def find_optimal_threshold(
    test_loader: torch.utils.data.DataLoader,
    models: Union[torch.nn.Module, List[torch.nn.Module]],
    experiment: str,
    device: str = "cuda",
    use_ensemble: bool = True,
    normalize_imgs: bool = True,
    thresholds: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Dict:
    """Find the threshold that maximizes median BOA per patch.

    Args:
        test_loader: DataLoader with test data.
        models: Single model or list of models.
        experiment: Experiment name from EXPERIMENTS.
        device: Device for execution.
        use_ensemble: If True, uses ensemble prediction.
        normalize_imgs: If True, normalizes images before inference.
        thresholds: Array of thresholds to test. Defaults to linspace(0, 1, 101).
        verbose: If True, shows progress bar.

    Returns:
        Dictionary with optimization results.

    Raises:
        ValueError: If experiment name is not in EXPERIMENTS.
    """
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}")

    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)

    pos_ids = EXPERIMENTS[experiment]["pos"]

    mean, std = None, None
    if normalize_imgs:
        mean, std = get_normalization_stats(device, False, SENTINEL_BANDS)

    models = _prepare_models(models, device)
    th_boas = {t: [] for t in thresholds}

    with torch.no_grad():
        for imgs, gts in tqdm(
            test_loader,
            desc=f"Finding t* for {experiment}",
            disable=not verbose,
        ):
            imgs = imgs.to(device).float()
            gts = gts.to(device)

            if normalize_imgs:
                imgs = normalize_images(imgs, mean, std)

            probs = get_predictions(
                models, imgs, use_ensemble=use_ensemble, return_probs=True
            )

            for i in range(imgs.size(0)):
                gt = gts[i].cpu().numpy()
                ppos = probs[i, pos_ids].sum(0).cpu().numpy()
                pos_mask = np.isin(gt, pos_ids)
                neg_mask = ~pos_mask

                for t in thresholds:
                    pred_pos = ppos >= t
                    tp = np.logical_and(pred_pos, pos_mask).sum()
                    fn = np.logical_and(~pred_pos, pos_mask).sum()
                    fp = np.logical_and(pred_pos, neg_mask).sum()
                    tn = np.logical_and(~pred_pos, neg_mask).sum()

                    boa = 0.5 * (
                        safe_divide(tp, tp + fn) + safe_divide(tn, tn + fp)
                    )
                    th_boas[t].append(boa)

    median_boa = np.array([np.nanmedian(th_boas[t]) for t in thresholds])
    best_idx = int(np.nanargmax(median_boa))

    return {
        "experiment": experiment,
        "thresholds": thresholds,
        "median_boas": median_boa,
        "best_idx": best_idx,
        "best_threshold": float(thresholds[best_idx]),
        "best_median_boa": float(median_boa[best_idx]),
        "n_patches": sum(len(v) for v in th_boas.values()) // len(thresholds),
    }