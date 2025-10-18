import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class CloudSEN12UnetMobV2(nn.Module):
    """
    Esta classe reproduz a arquitetura U-Net com backbone MobileNetV2,
    frequentemente chamada de 'UnetMobV2' nos trabalhos do CloudSEN12,
    configurada para lidar com 13 bandas (Sentinel-2) e 4 classes 
    (clear, thick cloud, thin cloud, cloud shadow).
    """
    def __init__(self, 
                 in_channels: int = 13, 
                 classes: int = 4, 
                 freeze_encoder: bool = False):
        """
        Args:
            in_channels (int): Número de bandas de entrada (ex.: 13 bandas do Sentinel-2).
            classes (int): Número de classes de saída (ex.: 4 para clear, thick, thin, shadow).
            freeze_encoder (bool): Se True, congela parâmetros do encoder (backbone).
        """
        super(CloudSEN12UnetMobV2, self).__init__()
        
        # Cria o modelo U-Net com MobileNetV2 como encoder.
        # encoder_weights=None pois normalmente não há pesos pré-treinados para 13 bandas.
        self.model = smp.Unet(
            encoder_name="mobilenet_v2",   # backbone MobileNetV2
            encoder_weights=None,          # sem pesos pré-treinados (apenas ImageNet tem 3 canais)
            in_channels=in_channels,
            classes=classes
        )
        
        # Opcional: congela os parâmetros do encoder
        if freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do modelo.
        
        Args:
            x (torch.Tensor): Imagem de entrada com shape (B, 13, H, W).
        
        Returns:
            torch.Tensor: Logits de saída com shape (B, classes, H, W).
        """
        return self.model(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retorna o rótulo multiclasses (exclusivas) de cada pixel.
        
        Usa softmax para obter distribuição de probabilidade, 
        e argmax para determinar a classe mais provável.
        
        Args:
            x (torch.Tensor): Imagem de entrada (B, 13, H, W).

        Returns:
            torch.Tensor: Máscara de classes (B, H, W), 
                          cada pixel ∈ {0,1,2,3}.
        """
        self.eval()
        logits = self.forward(x)                  # (B, 4, H, W)
        probs = torch.softmax(logits, dim=1)      # (B, 4, H, W)
        labels = probs.argmax(dim=1)              # (B, H, W)
        return labels
