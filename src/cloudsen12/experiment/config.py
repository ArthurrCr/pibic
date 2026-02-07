"""Experiment configuration and reproducibility utilities."""

import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Type

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Set random seed across all generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment.

    Attributes:
        name: Unique experiment identifier.
        model_class: PyTorch model class to instantiate.
        encoder_name: Encoder architecture name.
        encoder_weights: Pretrained weights identifier.
        in_channels: Number of input channels (13 for Sentinel-2).
        num_classes: Number of output classes (4 for CloudSEN12).
        batch_size: Training batch size.
        learning_rate: Initial learning rate.
        num_epochs: Maximum training epochs.
        patience: Early stopping patience.
        loss_name: Loss function key from LOSS_REGISTRY.
        loss_kwargs: Extra arguments for the loss function.
        scheduler_factor: Factor for ReduceLROnPlateau.
        scheduler_patience: Patience for ReduceLROnPlateau.
        seed: Random seed.
    """

    name: str
    model_class: Type[nn.Module]
    encoder_name: str = "tu-regnetz_d8"
    encoder_weights: Optional[str] = "imagenet"
    in_channels: int = 13
    num_classes: int = 4
    batch_size: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 50
    patience: int = 5
    loss_name: str = "ce"
    loss_kwargs: Dict = field(default_factory=dict)
    scheduler_factor: float = 0.1
    scheduler_patience: int = 3
    seed: int = 42