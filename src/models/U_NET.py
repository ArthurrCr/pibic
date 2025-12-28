"""U-Net segmentation models for cloud detection in Sentinel-2 imagery.


ImageNet pretrained weights were used for encoder initialization. 
For the 13-band input, the first convolutional layer weights were replicated 
cyclically across channels and rescaled following the segmentation-models-pytorch 
default behavior.
"""

from typing import Optional

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class CloudUNet(nn.Module):
    """
    U-Net segmentation model for cloud detection in Sentinel-2 images.

    Supports multiple encoder backbones and can be configured for different
    numbers of input bands and output classes.

    Attributes:
        unet: The underlying U-Net model from segmentation_models_pytorch.
    """

    def __init__(
        self,
        encoder_name: str = "tu-regnetz_d8",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 13,
        num_classes: int = 4,
        freeze_encoder: bool = False,
    ):
        """
        Initialize the cloud segmentation model.

        Args:
            encoder_name: Name of the encoder backbone. Examples:
                - "tu-regnetz_d8": RegNetZ-D8 via timm
            encoder_weights: Pretrained weights for encoder. Usually None
                for multi-band inputs since pretrained weights expect 3 channels.
            in_channels: Number of input bands (default: 13 for Sentinel-2).
            num_classes: Number of output classes. Default is 4:
                0=clear, 1=thick cloud, 2=thin cloud, 3=cloud shadow.
            freeze_encoder: If True, freeze encoder parameters during training.
        """
        super().__init__()

        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )

        if freeze_encoder:
            for param in self.unet.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input tensor with shape (B, in_channels, H, W).

        Returns:
            Output logits with shape (B, num_classes, H, W).
        """
        return self.unet(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels for each pixel.

        Applies softmax to get probability distribution, then argmax
        to determine the most likely class.

        Args:
            x: Input tensor with shape (B, in_channels, H, W).

        Returns:
            Class mask with shape (B, H, W), where each pixel value
            is in {0, 1, 2, 3}.
        """
        self.eval()
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        labels = probs.argmax(dim=1)
        return labels
