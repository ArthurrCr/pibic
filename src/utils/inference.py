"""Inference utilities for cloud segmentation models."""

from typing import List, Union

import torch

from utils.constants import SENTINEL_BANDS

# L1C normalization statistics from CloudS2Mask
L1C_MEAN = {
    "B01": 0.072623697227855,
    "B02": 0.06608867585127501,
    "B03": 0.061940767467830685,
    "B04": 0.06330473795822207,
    "B05": 0.06858655023065205,
    "B06": 0.08539433443008514,
    "B07": 0.09401670610922229,
    "B08": 0.09006412206990828,
    "B8A": 0.09915093732164396,
    "B09": 0.035429756513690985,
    "B10": 0.003632839439909688,
    "B11": 0.06855744750648961,
    "B12": 0.0486043830034996,
}

L1C_STD = {
    "B01": 0.020152047138155018,
    "B02": 0.022698212883948143,
    "B03": 0.023073879486441455,
    "B04": 0.02668270641026416,
    "B05": 0.0263763340626224,
    "B06": 0.027439342904551974,
    "B07": 0.02896087163616576,
    "B08": 0.028661147214616267,
    "B8A": 0.0301365958005653,
    "B09": 0.013482676031864258,
    "B10": 0.0019204000834290252,
    "B11": 0.023938917594669776,
    "B12": 0.020069414811037536,
}


def get_normalization_stats(
    device: Union[str, torch.device],
    fp16_mode: bool = False,
    required_bands: List[str] = SENTINEL_BANDS,
) -> tuple:
    """
    Get normalization statistics for Sentinel-2 L1C bands.

    Returns mean and std tensors shaped for broadcasting with image batches.

    Args:
        device: Target device for the tensors.
        fp16_mode: If True, returns half-precision tensors.
        required_bands: List of band names to include.

    Returns:
        Tuple of (mean, std) tensors with shape (1, n_bands, 1, 1).
    """
    mean_subset = torch.tensor([L1C_MEAN[b] for b in required_bands])
    std_subset = torch.tensor([L1C_STD[b] for b in required_bands])

    mean = mean_subset.view(1, len(required_bands), 1, 1).to(device)
    std = std_subset.view(1, len(required_bands), 1, 1).to(device)

    if fp16_mode:
        mean, std = mean.half(), std.half()

    return mean, std


def normalize_images(
    images: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """
    Normalize Sentinel-2 images using CloudS2Mask statistics.

    Args:
        images: Input tensor with shape (B, C, H, W).
        mean: Mean tensor from get_normalization_stats.
        std: Std tensor from get_normalization_stats.

    Returns:
        Normalized images tensor.
    """
    return (images / 32767.0 - mean) / std


def ensemble_inference(
    models: List[torch.nn.Module],
    images: torch.Tensor,
    return_probs: bool = False,
) -> torch.Tensor:
    """
    Perform ensemble inference by averaging model predictions.

    Args:
        models: List of models for ensemble.
        images: Input batch of images.
        return_probs: If True, returns averaged probabilities.
            If False, returns predicted class indices.

    Returns:
        Averaged probabilities (B, C, H, W) if return_probs=True,
        or predicted classes (B, H, W) if return_probs=False.
    """
    with torch.no_grad():
        probs = torch.stack(
            [torch.softmax(m(images), dim=1) for m in models]
        ).mean(dim=0)

    if return_probs:
        return probs
    return torch.argmax(probs, dim=1)


def get_predictions(
    models: Union[torch.nn.Module, List[torch.nn.Module]],
    images: torch.Tensor,
    use_ensemble: bool = True,
    return_probs: bool = False,
) -> torch.Tensor:
    """
    Get predictions from a model or ensemble.

    Args:
        models: Single model or list of models.
        images: Input batch of images.
        use_ensemble: If True and multiple models provided, uses ensemble.
        return_probs: If True, returns probabilities instead of class indices.

    Returns:
        Predictions tensor (probabilities or class indices).
    """
    if not isinstance(models, list):
        models = [models]

    if use_ensemble and len(models) > 1:
        return ensemble_inference(models, images, return_probs=return_probs)

    with torch.no_grad():
        output = models[0](images)
        probs = torch.softmax(output, dim=1)

    if return_probs:
        return probs
    return torch.argmax(probs, dim=1)


def load_models(
    model_paths: List[str],
    device: Union[str, torch.device],
) -> List[torch.nn.Module]:
    """
    Load PyTorch models from a list of file paths.

    Args:
        model_paths: List of paths to saved model files.
        device: Device to load models onto.

    Returns:
        List of loaded models in evaluation mode.
    """
    models = []
    for model_path in model_paths:
        print(f"Loading model: {model_path}")
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        models.append(model)
    return models