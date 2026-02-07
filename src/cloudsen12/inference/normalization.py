"""Normalization statistics and utilities for Sentinel-2 L1C imagery."""

from typing import Dict, List, Tuple, Union

import torch

from cloudsen12.config.constants import SENTINEL_BANDS

L1C_MEAN: Dict[str, float] = {
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

L1C_STD: Dict[str, float] = {
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return mean and std tensors shaped for broadcasting (1, C, 1, 1).

    Args:
        device: Target device for the tensors.
        fp16_mode: If True, returns half-precision tensors.
        required_bands: List of band names to include.

    Returns:
        Tuple of (mean, std) tensors.
    """
    n = len(required_bands)
    mean = torch.tensor([L1C_MEAN[b] for b in required_bands]).view(1, n, 1, 1).to(device)
    std = torch.tensor([L1C_STD[b] for b in required_bands]).view(1, n, 1, 1).to(device)

    if fp16_mode:
        mean, std = mean.half(), std.half()

    return mean, std


def normalize_images(
    images: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Normalize Sentinel-2 images using CloudS2Mask statistics.

    Applies: (images / 32767.0 - mean) / std

    Args:
        images: Input tensor (B, C, H, W).
        mean: Mean tensor from get_normalization_stats.
        std: Std tensor from get_normalization_stats.

    Returns:
        Normalized images tensor.
    """
    return (images / 32767.0 - mean) / std