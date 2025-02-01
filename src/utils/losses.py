import torch
import torch.nn as nn

class BCE_Dice_Loss(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        # BCE
        bce_loss = self.bce(preds, targets)
        
        # Sigmoid para converter logits em probabilidades
        preds_prob = torch.sigmoid(preds)

        # Dice
        intersection = (preds_prob * targets).sum(dim=(1,2,3))
        union = preds_prob.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
        dice = 1 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = dice.mean()

        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss

class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds, targets):
        # Focal Loss
        bce = self.bce(preds, targets)
        probas = torch.sigmoid(preds)
        p_t = probas * targets + (1 - probas) * (1 - targets)
        focal_loss = (self.alpha * (1 - p_t) ** self.gamma * bce).mean()

        # Dice Loss
        preds_prob = torch.sigmoid(preds)
        intersection = (preds_prob * targets).sum(dim=(1,2,3))
        union = preds_prob.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
        dice_loss = 1 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = dice_loss.mean()

        return focal_loss + dice_loss
    

class TverskyBCE(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        # BCE com peso para a classe positiva
        bce_loss = self.bce(preds, targets)
        
        # Tversky
        preds_prob = torch.sigmoid(preds)
        TP = (preds_prob * targets).sum(dim=(1,2,3))
        FP = (preds_prob * (1 - targets)).sum(dim=(1,2,3))
        FN = ((1 - preds_prob) * targets).sum(dim=(1,2,3))
        tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        tversky_loss = 1 - tversky.mean()

        return bce_loss + tversky_loss