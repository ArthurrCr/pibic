from typing import Dict
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# Importar do módulo metrics
from .metrics import (
    SENTINEL_BANDS,
    EXPERIMENTS,
    get_normalization_stats,
    get_predictions,
    normalize_images
)


def safe_div(num: float, den: float):
    """Divisão segura que devolve NaN se o denominador for zero."""
    return np.nan if den == 0 else num / den            


def buckets(arr: np.ndarray):
    """Percentual de valores em três faixas (<0.1 | 0.1—0.9 | >0.9)."""
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (0.0, 0.0, 0.0)
    hist, _ = np.histogram(arr, [0, 0.1, 0.9, 1.01])
    return tuple(hist / arr.size * 100)


def evaluate_test_dataset(
    test_loader,
    models,
    device: str = "cuda",
    use_ensemble: bool = True,
    normalize_imgs: bool = True,
):
    """
    Compute patch‐level PA, UA, BOA, OE and CE for the three binary experiments.
    """

    mean, std = get_normalization_stats(
        device, False, SENTINEL_BANDS
    )

    # Ensure models is a list
    if not isinstance(models, list):
        models = [models]
    for m in models:
        m.to(device).eval()

    # Accumulators
    exps: Dict[str, Dict[str, list]] = {
        k: dict(v, PA=[], UA=[], BOA=[], OE=[], CE=[])
        for k, v in EXPERIMENTS.items()
    }

    n_patches = 0

    with torch.no_grad():
        for imgs, gts in tqdm(test_loader, desc="Processando patches"):
            imgs, gts = imgs.to(device).float(), gts.to(device)
            n_patches += imgs.size(0)

            if normalize_imgs:
                imgs = normalize_images(imgs, mean, std)

            preds = get_predictions(models, imgs, use_ensemble=use_ensemble)

            for gt, pr in zip(gts.cpu().numpy(), preds.cpu().numpy()):
                cm = confusion_matrix(gt.ravel(), pr.ravel(), labels=[0, 1, 2, 3])

                for name, cfg in exps.items():
                    pos = cfg["pos"]
                    neg = [c for c in range(4) if c not in pos]

                    TP = cm[np.ix_(pos, pos)].sum()
                    FN = cm[np.ix_(pos, neg)].sum()
                    FP = cm[np.ix_(neg, pos)].sum()
                    TN = cm[np.ix_(neg, neg)].sum()

                    pa = safe_div(TP, TP + FN)
                    ua = safe_div(TP, TP + FP)

                    # Pular PA/UA em patches *sem nuvem* (apenas "cloud/no cloud")
                    if name == "cloud/no cloud" and (TP + FN) == 0:
                        pa = np.nan
                        ua = np.nan

                    boa = 0.5 * (safe_div(TP, TP + FN) + safe_div(TN, TN + FP))

                    # OE/CE respeitam NaN corretamente
                    oe = (np.nan if np.isnan(pa) else (1.0 - pa))
                    ce = (np.nan if np.isnan(ua) else (1.0 - ua))

                    cfg["PA"].append(pa)
                    cfg["UA"].append(ua)
                    cfg["BOA"].append(boa)
                    cfg["OE"].append(oe)
                    cfg["CE"].append(ce)

    # Summary table
    rows = []
    for name, cfg in exps.items():
        pa_low, pa_mid, pa_high = buckets(np.array(cfg["PA"]))
        ua_low, ua_mid, ua_high = buckets(np.array(cfg["UA"]))

        rows.append(
            {
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
        )

    return pd.DataFrame(rows)


def evaluate_test_dataset_with_thresholds(
    test_loader,
    models,
    t_star: Dict[str, float],
    device: str = "cuda",
    use_ensemble: bool = True,
    normalize_imgs: bool = True,
):
    """
    Evaluate binary experiments on a test set using provided optimal thresholds (t*).
    """

    mean, std = get_normalization_stats(
        device, False, SENTINEL_BANDS
    )

    if not isinstance(models, list):
        models = [models]
    for m in models:
        m.to(device).eval()

    # Prepare accumulator per experiment for which a threshold is provided
    exps: Dict[str, Dict[str, list]] = {}
    for name, cfg in EXPERIMENTS.items():
        if name in t_star:
            exps[name] = dict(
                cfg,
                threshold=t_star[name],
                PA=[], UA=[], BOA=[], OE=[], CE=[],
            )

    n_patches = 0

    with torch.no_grad():
        for imgs, gts in tqdm(test_loader, desc="Processando patches"):
            imgs, gts = imgs.to(device).float(), gts.to(device)
            n_patches += imgs.size(0)

            if normalize_imgs:
                imgs = normalize_images(imgs, mean, std)

            # Obtain probabilities for each class
            probs = get_predictions(
                models, imgs, use_ensemble=use_ensemble, return_probs=True
            )

            for i in range(imgs.size(0)):
                gt = gts[i].cpu().numpy()

                for name, cfg in exps.items():
                    pos = cfg["pos"]
                    neg = [c for c in range(4) if c not in pos]
                    t = cfg["threshold"]

                    # Sum probabilities of positive classes
                    ppos = probs[i, pos].sum(0).cpu().numpy()
                    pred_pos = ppos >= t

                    gt_pos = np.isin(gt, pos)
                    gt_neg = np.isin(gt, neg)

                    tp = np.logical_and(pred_pos, gt_pos).sum()
                    fn = np.logical_and(~pred_pos, gt_pos).sum()
                    fp = np.logical_and(pred_pos, gt_neg).sum()
                    tn = np.logical_and(~pred_pos, gt_neg).sum()

                    pa  = safe_div(tp, tp + fn)
                    ua  = safe_div(tp, tp + fp)

                    # Pular PA/UA em patches *sem nuvem* (apenas "cloud/no cloud")
                    if name == "cloud/no cloud" and not gt_pos.any():
                        pa = np.nan
                        ua = np.nan

                    boa = 0.5 * (safe_div(tp, tp + fn) + safe_div(tn, tn + fp))

                    # OE/CE respeitam NaN corretamente
                    oe = (np.nan if np.isnan(pa) else (1.0 - pa))
                    ce = (np.nan if np.isnan(ua) else (1.0 - ua))

                    cfg["PA"].append(pa)
                    cfg["UA"].append(ua)
                    cfg["BOA"].append(boa)
                    cfg["OE"].append(oe)
                    cfg["CE"].append(ce)

    # Construct summary table
    rows = []
    for name, cfg in exps.items():
        pa_low, pa_mid, pa_high = buckets(np.array(cfg["PA"]))
        ua_low, ua_mid, ua_high = buckets(np.array(cfg["UA"]))
        threshold = cfg.get("threshold")

        rows.append(
            {
                "Experiment": name,
                "Threshold": f"{threshold:.2f}",
                "Median BOA": f"{np.nanmedian(cfg['BOA']):.4f}",
                "PA low%": f"{pa_low:.2f}",
                "PA middle%": f"{pa_mid:.2f}",
                "PA high%": f"{pa_high:.2f}",
                "UA low%": f"{ua_low:.2f}",
                "UA middle%": f"{ua_mid:.2f}",
                "UA high%": f"{ua_high:.2f}",
                "N patches": len(cfg["BOA"]),
            }
        )

    return pd.DataFrame(rows)


def find_optimal_threshold_by_patch(
        test_loader,
        models,
        experiment: str,
        device: str = 'cuda',
        use_ensemble: bool = True,
        normalize_imgs: bool = True,
        thresholds: np.ndarray = np.linspace(0, 1, 101),
        verbose: bool = True
    ):
    """
    Procura o limiar t* que maximiza a mediana do BOA por patch
    para um dos três experimentos definidos em `EXPERIMENTS`.

    Parâmetro `experiment` deve ser uma das chaves de `EXPERIMENTS`
    ('cloud/no cloud' | 'cloud shadow' | 'valid/invalid').
    """

    if experiment not in EXPERIMENTS:
        raise ValueError(f"experimento desconhecido: {experiment}")

    pos_ids = EXPERIMENTS[experiment]['pos']

    if normalize_imgs:
        mean, std = get_normalization_stats(device, False, SENTINEL_BANDS)

    if not isinstance(models, list):
        models = [models]
    for m in models:
        m.to(device).eval()

    th_boas = {t: [] for t in thresholds}

    with torch.no_grad():
        for imgs, gts in tqdm(test_loader,
                              desc=f"t* :: {experiment}",
                              disable=not verbose):
            imgs, gts = imgs.to(device).float(), gts.to(device)
            if normalize_imgs:
                imgs = normalize_images(imgs, mean, std)

            probs = get_predictions(models, imgs,
                                    use_ensemble=use_ensemble,
                                    return_probs=True)

            for i in range(imgs.size(0)):
                gt   = gts[i].cpu().numpy()
                ppos = probs[i, pos_ids].sum(0).cpu().numpy()

                # Máscaras fixas por patch
                pos_mask = np.isin(gt, pos_ids)
                neg_mask = ~pos_mask  # equivale aos ids não positivos

                for t in thresholds:
                    pred_pos = ppos >= t

                    tp = np.logical_and(pred_pos,  pos_mask).sum()
                    fn = np.logical_and(~pred_pos, pos_mask).sum()
                    fp = np.logical_and(pred_pos,  neg_mask).sum()
                    tn = np.logical_and(~pred_pos, neg_mask).sum()

                    boa = 0.5 * (safe_div(tp, tp + fn) +
                                 safe_div(tn, tn + fp))
                    th_boas[t].append(boa)

    median_boa = np.array([np.nanmedian(th_boas[t]) for t in thresholds])
    best_idx   = int(np.nanargmax(median_boa))

    return {
        'experiment'       : experiment,
        'thresholds'       : thresholds,
        'median_boas'      : median_boa,
        'best_idx'         : best_idx,
        'best_threshold'   : float(thresholds[best_idx]),
        'best_median_boa'  : float(median_boa[best_idx]),
        'n_patches'        : int(sum(len(v) for v in th_boas.values()) // len(thresholds))
    }