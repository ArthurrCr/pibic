"""
Utility functions for evaluating cloud and cloud‑shadow detection models.

These routines encapsulate several commonly used metrics in the remote sensing
literature, namely the **producer's accuracy (PA)**, **user's accuracy (UA)** and
**balanced overall accuracy (BOA)**. The functions are designed to operate on
patch‑level predictions produced by one or more PyTorch models and summarise
their performance across predefined experiments (e.g. detecting any type of
cloud vs. clear sky, detecting cloud shadows, or distinguishing valid from
invalid pixels).

The metrics implemented here follow the definitions used by Aybar et al.
(2022) and Wright et al. (2024). In particular, producer's accuracy (also
known as recall or sensitivity) is defined as

`
PA = TP / (TP + FN),
`

where *TP* and *FN* denote the number of true positive and false negative
pixels, respectively. User's accuracy (equivalent to
precision) is defined as

`
UA = TP / (TP + FP),
`

where *FP* denotes false positives. Balanced overall
accuracy takes into account both the ability to detect positive classes and to
avoid false alarms. It is computed as the average of the producer's
accuracy and the specificity (true negative rate):

`
BOA = 0.5 * (TP/(TP + FN) + TN/(TN + FP)).
`

When reporting PA and UA values over many image patches, the CloudSEN12
evaluation protocol bins values into three categories: **low** (< 0.1),
**middle** (0.1–0.9) and **high** (> 0.9). The distribution of PA/UA values
across patches can be summarised as the percentage of patches falling into
each bin.

These helper functions implement the above definitions, handle NaN values
gracefully and support model ensembling by averaging the predicted
probabilities from multiple models. They can be used to evaluate cloud and
cloud‑shadow masks in a reproducible and self‑contained manner.

References
----------
* C. Aybar et al., “CloudSEN12, a global dataset for semantic understanding
  of cloud and cloud shadow in Sentinel‑2,” *Scientific Data* **9**, 782
  (2022).
* N. Wright et al., “CloudS2Mask: A novel deep learning approach for
  improved cloud and cloud shadow masking in Sentinel‑2 imagery,” *Remote
  Sensing of Environment* **306**, 114122 (2024).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

__all__ = [
    "EXPERIMENTS",
    "buckets",
    "get_predictions",
    "safe_div",
    "evaluate_metrics",
    "evaluate_with_thresholds",
    "find_optimal_threshold_by_patch",
]

# -----------------------------------------------------------------------------
# Experiment definitions
#
# Each experiment groups one or more of the four label indices into a
# “positive” class. Remaining indices are considered “negative” for the
# purpose of computing binary metrics. The order of labels corresponds to the
# CloudSEN12 dataset: 0 – clear, 1 – thick cloud, 2 – thin cloud and
# 3 – cloud shadow. See Aybar et al. (2022) for details.

EXPERIMENTS: Mapping[str, Mapping[str, List[int]]] = {
    "cloud/no cloud": {"pos": [1, 2]},  # both thick and thin clouds vs everything else
    "cloud shadow": {"pos": [3]},       # cloud shadows vs non‑shadows
    "valid/invalid": {"pos": [0]},      # clear vs any kind of contamination
}


def buckets(arr: np.ndarray) -> Tuple[float, float, float]:
    """Compute the percentage of values falling into predefined metric ranges.

    Given an array of metric values in the interval [0, 1], this helper
    function classifies each value into one of three bins: **low** (< 0.1),
    **middle** (0.1–0.9) or **high** (> 0.9). The idea originates from the
    evaluation protocol of the CloudSEN12 dataset, where PA and UA values
    across patches are summarised by the proportion of patches in each bin.

    Parameters
    ----------
    arr : np.ndarray
        Array of floating point values in the range [0, 1]. Values outside
        this range or NaNs are ignored.

    Returns
    -------
    Tuple[float, float, float]
        A tuple (low_pct, mid_pct, high_pct) containing the percentage of
        finite values falling into the **low**, **middle** and **high** bins,
        expressed as percentages in the range [0, 100]. If the input array
        contains no finite values, all three percentages will be 0.
    """
    # Remove NaNs and values outside [0, 1]
    finite = arr[np.isfinite(arr)]
    finite = finite[(finite >= 0.0) & (finite <= 1.0)]
    if finite.size == 0:
        return 0.0, 0.0, 0.0

    # Use numpy.histogram to count the number of values in each bin. The last
    # bin edge is slightly above 1 to include values equal to 1.
    hist, _ = np.histogram(finite, bins=[0.0, 0.1, 0.9, 1.01])
    # Convert counts to percentages
    return tuple((hist / float(finite.size)) * 100.0)


def get_predictions(
    models: Union[torch.nn.Module, Sequence[torch.nn.Module]],
    images: torch.Tensor,
    *,
    return_probs: bool = False,
) -> torch.Tensor:
    """Generate predictions or class probabilities for a batch of images.

    This function accepts either a single PyTorch model or a sequence of models.
    If multiple models are provided, their softmax probabilities are averaged
    (model ensembling) to produce more robust predictions. The models should
    already be set to evaluation mode and moved to the appropriate device.

    Parameters
    ----------
    models : torch.nn.Module or Sequence[torch.nn.Module]
        A single model or a collection of models that map a batch of images to
        unnormalised logits. All models must accept the same input shape and
        output the same number of classes.
    images : torch.Tensor
        A tensor of shape (B, C, H, W) containing a batch of images. The
        tensor should already reside on the same device as the models.
    return_probs : bool, optional
        If True, return the softmax probabilities for each class. If
        False (default), return the hard class predictions obtained via
        argmax on the averaged probabilities.

    Returns
    -------
    torch.Tensor
        If return_probs is True, returns a tensor of shape
        (B, num_classes, H, W) containing averaged probabilities. Otherwise,
        returns a tensor of shape (B, H, W) with integer class labels.

    Notes
    -----
    Ensembling multiple models by averaging their probabilities can improve
    prediction accuracy, as suggested by Wright et al. (2024) for the
    CloudS2Mask model family.
    """
    # Normalise input models into a list for unified processing
    if isinstance(models, (list, tuple)):
        model_list = list(models)
    else:
        model_list = [models]

    # Accumulate probabilities from each model
    prob_list: List[torch.Tensor] = []
    with torch.no_grad():
        for model in model_list:
            # Forward pass to obtain logits
            logits = model(images)
            # Apply softmax along the class dimension (1) to obtain probabilities
            probs = torch.softmax(logits, dim=1)
            prob_list.append(probs)

    # Average probabilities across models
    if len(prob_list) == 1:
        avg_probs = prob_list[0]
    else:
        avg_probs = torch.mean(torch.stack(prob_list, dim=0), dim=0)

    return avg_probs if return_probs else torch.argmax(avg_probs, dim=1)


def safe_div(num: float, den: float) -> float:
    """Safely divide two numbers, returning NaN if the denominator is zero.

    Parameters
    ----------
    num : float
        Numerator.
    den : float
        Denominator.

    Returns
    -------
    float
        num / den if den is non‑zero; otherwise np.nan. This helper
        prevents division by zero and maintains NaN values in metric
        computations when no positive or negative examples are present.
    """
    return np.nan if den == 0 else num / den


def evaluate_metrics(
    test_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    models: Union[torch.nn.Module, Sequence[torch.nn.Module]],
    *,
    device: str = "cuda",
) -> pd.DataFrame:
    """Evaluate PA, UA and BOA metrics across predefined experiments.

    This routine iterates over a test data loader, generates predictions via
    :func:`get_predictions` and computes the producer's accuracy (PA), user's
    accuracy (UA) and balanced overall accuracy (BOA) for each image patch.
    Patches are evaluated under several binary experiments defined in
    :data:`EXPERIMENTS`, where one or more of the four semantic classes are
    treated as “positive” and the remainder as “negative”.

    The per‑patch metrics are aggregated to summarise model performance:

    * **Median BOA** — the median balanced overall accuracy across all patches.
    * **PA/UA low%** — percentage of patches with PA/UA < 0.1.
    * **PA/UA middle%** — percentage of patches with PA/UA between 0.1 and 0.9.
    * **PA/UA high%** — percentage of patches with PA/UA > 0.9.

    These summary statistics mirror the CloudSEN12 evaluation protocol.

    Parameters
    ----------
    test_loader : iterable of tuples
        An iterable yielding (images, labels) pairs. images should be
        tensors of shape (B, C, H, W) and labels should be integer
        tensors of shape (B, H, W) with values in {0, 1, 2, 3}.
    models : torch.nn.Module or sequence of torch.nn.Module
        A single model or list of models to be evaluated. Each model must be
        moved to device and set to evaluation mode prior to calling this
        function.
    device : str, optional
        The device on which inference should run (default: "cuda"). If
        CUDA is unavailable, set this to "cpu".

    Returns
    -------
    pandas.DataFrame
        A DataFrame with one row per experiment containing the median BOA and
        distribution of PA/UA values. The column "Experiment" identifies
        the experiment, and "N patches" records the number of patches
        evaluated.

    Notes
    -----
    * For the “cloud/no cloud” experiment, patches containing no cloud pixels
      (i.e. no positive examples) will have undefined PA and UA; these are
      recorded as NaN to avoid skewing the distribution.
    * BOA is computed per patch using the formula defined above.
    """
    # Normalise models into a list and set evaluation mode
    if isinstance(models, (list, tuple)):
        model_list = list(models)
    else:
        model_list = [models]
    for m in model_list:
        m.to(device)
        m.eval()

    # Initialise accumulators for each experiment
    exps: Dict[str, Dict[str, List[float]]] = {
        name: {**cfg, "PA": [], "UA": [], "BOA": []}
        for name, cfg in EXPERIMENTS.items()
    }

    with torch.no_grad():
        for imgs, gts in tqdm(test_loader, desc="Evaluating patches"):
            imgs = imgs.to(device).float()
            gts = gts.to(device)

            # Generate hard predictions for the batch
            preds = get_predictions(model_list, imgs, return_probs=False)

            # Loop over individual patches in the batch
            for gt, pr in zip(gts.cpu().numpy(), preds.cpu().numpy()):
                # Compute confusion matrix for all four classes (0–3)
                cm = confusion_matrix(gt.ravel(), pr.ravel(), labels=[0, 1, 2, 3])

                for name, cfg in exps.items():
                    pos = cfg["pos"]
                    neg = [c for c in range(4) if c not in pos]

                    # Sum confusion matrix blocks to get TP, FN, FP, TN
                    TP = cm[np.ix_(pos, pos)].sum()
                    FN = cm[np.ix_(pos, neg)].sum()
                    FP = cm[np.ix_(neg, pos)].sum()
                    TN = cm[np.ix_(neg, neg)].sum()

                    # Compute metrics; safe_div handles divisions by zero
                    pa = safe_div(TP, TP + FN)
                    ua = safe_div(TP, TP + FP)

                    # In the cloud/no cloud experiment, skip PA/UA for
                    # cloudless patches (no positive ground‑truth pixels)
                    if name == "cloud/no cloud" and (TP + FN) == 0:
                        pa = np.nan
                        ua = np.nan

                    boa = 0.5 * (
                        safe_div(TP, TP + FN) + safe_div(TN, TN + FP)
                    )

                    cfg["PA"].append(pa)
                    cfg["UA"].append(ua)
                    cfg["BOA"].append(boa)

    # Build summary table
    rows: List[Mapping[str, Union[str, float, int]]] = []
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


def evaluate_with_thresholds(
    test_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    models: Union[torch.nn.Module, Sequence[torch.nn.Module]],
    t_star: Mapping[str, float],
    *,
    device: str = "cuda",
) -> pd.DataFrame:
    """Evaluate metrics under custom probability thresholds for each experiment.

    Instead of using the argmax over class probabilities, this function sums
    probabilities over the positive classes defined in :data:`EXPERIMENTS` and
    applies a scalar threshold to decide whether each pixel belongs to the
    positive or negative superclass. Separate thresholds can be supplied for
    different experiments via the mapping t_star.

    For example, in the “cloud/no cloud” experiment the probabilities of
    thick and thin cloud classes are summed and compared to t_star['cloud/no
    cloud']. Pixels with a summed probability greater than or equal to this
    threshold are predicted as cloud; otherwise, they are predicted as clear.

    Parameters
    ----------
    test_loader : iterable of tuples
        Yields (images, labels) pairs as described in
        :func:`evaluate_metrics`.
    models : torch.nn.Module or sequence of torch.nn.Module
        Model or models used to generate class probabilities. Probabilities
        from multiple models are averaged as in :func:`get_predictions`.
    t_star : Mapping[str, float]
        A dictionary mapping experiment names to threshold values in the
        interval [0, 1]. Only experiments present in this mapping are
        evaluated; others are ignored.
    device : str, optional
        The device on which inference should run (default: "cuda").

    Returns
    -------
    pandas.DataFrame
        Summary statistics in the same format as returned by
        :func:`evaluate_metrics`.
    """
    # Normalise models into a list and set evaluation mode
    if isinstance(models, (list, tuple)):
        model_list = list(models)
    else:
        model_list = [models]
    for m in model_list:
        m.to(device)
        m.eval()

    # Initialise accumulators only for experiments that have a threshold
    exps: Dict[str, Dict[str, Union[List[float], List[int], float]]] = {}
    for name, cfg in EXPERIMENTS.items():
        if name in t_star:
            exps[name] = {**cfg, "threshold": t_star[name], "PA": [], "UA": [], "BOA": []}

    with torch.no_grad():
        for imgs, gts in tqdm(test_loader, desc="Evaluating thresholds"):
            imgs = imgs.to(device).float()
            gts = gts.to(device)
            # Obtain per‑class probabilities from the ensemble
            probs = get_predictions(model_list, imgs, return_probs=True)

            # Loop over patches within the batch
            for idx in range(imgs.size(0)):
                gt = gts[idx].cpu().numpy()

                for name, cfg in exps.items():
                    pos = cfg["pos"]
                    neg = [c for c in range(4) if c not in pos]
                    threshold = float(cfg["threshold"])

                    # Sum probabilities over the positive classes and apply the threshold
                    ppos = probs[idx, pos].sum(dim=0).cpu().numpy()
                    pred_pos = ppos >= threshold

                    gt_pos = np.isin(gt, pos)
                    gt_neg = np.isin(gt, neg)

                    tp = np.logical_and(pred_pos, gt_pos).sum()
                    fn = np.logical_and(~pred_pos, gt_pos).sum()
                    fp = np.logical_and(pred_pos, gt_neg).sum()
                    tn = np.logical_and(~pred_pos, gt_neg).sum()

                    pa = safe_div(tp, tp + fn)
                    ua = safe_div(tp, tp + fp)

                    # As before, skip PA/UA for cloudless patches in the cloud/no cloud experiment
                    if name == "cloud/no cloud" and not gt_pos.any():
                        pa = np.nan
                        ua = np.nan

                    boa = 0.5 * (
                        safe_div(tp, tp + fn) + safe_div(tn, tn + fp)
                    )

                    cfg["PA"].append(pa)
                    cfg["UA"].append(ua)
                    cfg["BOA"].append(boa)

    # Build summary table
    rows: List[Mapping[str, Union[str, float, int]]] = []
    for name, cfg in exps.items():
        pa_low, pa_mid, pa_high = buckets(np.array(cfg["PA"]))
        ua_low, ua_mid, ua_high = buckets(np.array(cfg["UA"]))
        rows.append(
            {
                "Experiment": name,
                "Threshold": f"{cfg['threshold']:.2f}",
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
    test_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    models: Union[torch.nn.Module, Sequence[torch.nn.Module]],
    experiment: str,
    *,
    device: str = "cuda",
    thresholds: np.ndarray = np.linspace(0.0, 1.0, 101),
    verbose: bool = True,
) -> Dict[str, Union[str, float, int, np.ndarray]]:
    """Search for the threshold that maximises median BOA over patches.

    For a single experiment (e.g. “cloud shadow”), this function evaluates
    balanced overall accuracy across a range of candidate thresholds and
    returns the threshold yielding the highest median BOA. The evaluation
    procedure mirrors :func:`evaluate_with_thresholds`, but aggregates
    results by threshold instead of using a fixed threshold.

    Parameters
    ----------
    test_loader : iterable of tuples
        As described in :func:`evaluate_metrics`.
    models : torch.nn.Module or sequence of torch.nn.Module
        Model(s) used to generate class probabilities.
    experiment : str
        Name of the experiment to optimise. Must be one of the keys in
        :data:`EXPERIMENTS`.
    device : str, optional
        Device used for inference (default: "cuda").
    thresholds : np.ndarray, optional
        Array of threshold candidates in the interval [0, 1] (default:
        101 values spaced by 0.01).
    verbose : bool, optional
        Whether to display a progress bar during evaluation (default:
        True).

    Returns
    -------
    dict
        A dictionary containing the experiment name, the array of
        thresholds tested, an array of median BOA values for each threshold,
        the index of the optimal threshold, the optimal threshold value
        itself, the best median BOA and the number of patches evaluated.
    """
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}")

    pos_ids = EXPERIMENTS[experiment]["pos"]
    neg_ids = [c for c in range(4) if c not in pos_ids]

    # Normalise models into a list and set evaluation mode
    if isinstance(models, (list, tuple)):
        model_list = list(models)
    else:
        model_list = [models]
    for m in model_list:
        m.to(device)
        m.eval()

    # Prepare accumulator: dictionary mapping threshold to list of BOA values
    boa_by_threshold: Dict[float, List[float]] = {float(t): [] for t in thresholds}

    with torch.no_grad():
        for imgs, gts in tqdm(
            test_loader,
            desc=f"Optimising threshold for {experiment}",
            disable=not verbose,
        ):
            imgs = imgs.to(device).float()
            gts = gts.to(device)

            # Compute probabilities for the entire batch
            probs = get_predictions(model_list, imgs, return_probs=True)

            for i in range(imgs.size(0)):
                gt = gts[i].cpu().numpy()
                # Sum probabilities over positive classes
                ppos = probs[i, pos_ids].sum(dim=0).cpu().numpy()

                for t in thresholds:
                    t_val = float(t)
                    pred_pos = ppos >= t_val

                    tp = np.logical_and(pred_pos, np.isin(gt, pos_ids)).sum()
                    fn = np.logical_and(~pred_pos, np.isin(gt, pos_ids)).sum()
                    fp = np.logical_and(pred_pos, np.isin(gt, neg_ids)).sum()
                    tn = np.logical_and(~pred_pos, np.isin(gt, neg_ids)).sum()

                    boa = 0.5 * (
                        safe_div(tp, tp + fn) + safe_div(tn, tn + fp)
                    )
                    boa_by_threshold[t_val].append(boa)

    # Compute median BOA for each threshold
    median_boa = np.array([
        np.nanmedian(boa_by_threshold[t]) for t in thresholds
    ])
    best_idx = int(np.nanargmax(median_boa))

    # Compute number of patches evaluated (same for all thresholds)
    n_patches = sum(len(v) for v in boa_by_threshold.values()) // len(thresholds)

    return {
        "experiment": experiment,
        "thresholds": thresholds,
        "median_boas": median_boa,
        "best_idx": best_idx,
        "best_threshold": float(thresholds[best_idx]),
        "best_median_boa": float(median_boa[best_idx]),
        "n_patches": int(n_patches),
    }