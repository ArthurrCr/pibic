"""Loss functions for cloud segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for multi-class segmentation."""

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        
        if targets.dim() == 4:
            targets = targets.squeeze(1)
        
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probs * targets_onehot).sum(dims)
        cardinality = probs.sum(dims) + targets_onehot.sum(dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 4:
            targets = targets.squeeze(1)

        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """Tversky Loss with asymmetric FP/FN weighting."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        
        if targets.dim() == 4:
            targets = targets.squeeze(1)

        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        tp = (probs * targets_onehot).sum(dims)
        fp = (probs * (1 - targets_onehot)).sum(dims)
        fn = ((1 - probs) * targets_onehot).sum(dims)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss for thin cloud detection."""

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        
        if targets.dim() == 4:
            targets = targets.squeeze(1)

        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        tp = (probs * targets_onehot).sum(dims)
        fp = (probs * (1 - targets_onehot)).sum(dims)
        fn = ((1 - probs) * targets_onehot).sum(dims)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = (1.0 - tversky) ** self.gamma
        return focal_tversky.mean()


class DiceCELoss(nn.Module):
    """Combined Dice + CrossEntropy Loss."""

    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 4:
            targets = targets.squeeze(1)

        dice_loss = self.dice(logits, targets)
        ce_loss = F.cross_entropy(logits, targets)
        return self.dice_weight * dice_loss + self.ce_weight * ce_loss


class DiceFocalLoss(nn.Module):
    """Combined Dice + Focal Loss."""

    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice = DiceLoss()
        self.focal = FocalLoss(gamma=gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(logits, targets)
        focal_loss = self.focal(logits, targets)
        return self.dice_weight * dice_loss + self.focal_weight * focal_loss


LOSS_REGISTRY = {
    "ce": (nn.CrossEntropyLoss, {}),
    "dice": (DiceLoss, {}),
    "focal": (FocalLoss, {"gamma": 2.0, "alpha": 0.25}),
    "tversky": (TverskyLoss, {"alpha": 0.3, "beta": 0.7}),
    "focal_tversky": (FocalTverskyLoss, {"alpha": 0.3, "beta": 0.7, "gamma": 0.75}),
    "dice_ce": (DiceCELoss, {"dice_weight": 0.5, "ce_weight": 0.5}),
    "dice_focal": (DiceFocalLoss, {"dice_weight": 0.5, "focal_weight": 0.5}),
}


def get_loss(name: str, **kwargs) -> nn.Module:
    """Get loss function by name."""
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name}. Available: {list(LOSS_REGISTRY.keys())}")
    
    loss_class, default_kwargs = LOSS_REGISTRY[name]
    final_kwargs = {**default_kwargs, **kwargs}
    
    if name == "ce":
        return loss_class()
    return loss_class(**final_kwargs)