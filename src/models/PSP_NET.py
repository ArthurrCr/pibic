import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class CloudPSPNet(nn.Module):
    """
    PSPNet para segmentação multiespectral (13 bandas) em 4 classes exclusivas 
    (ex.: clear, thick cloud, thin cloud, cloud shadow).
    """
    def __init__(self,
                 encoder_name='resnet34',
                 encoder_weights=None,
                 in_channels=13,
                 classes=4,
                 freeze_encoder=False):
        """
        Args:
            encoder_name (str): Nome do encoder (ex.: 'resnet34').
            encoder_weights (str ou None): Pesos pré-treinados para o encoder 
                (geralmente None para 13 bandas).
            in_channels (int): Nº de canais de entrada (13 para Sentinel-2).
            classes (int): Nº de classes (4 para clear, thick cloud, thin cloud, shadow).
            freeze_encoder (bool): Se True, congela os parâmetros do encoder.
        """
        super(CloudPSPNet, self).__init__()
        
        # Cria o modelo PSPNet com os parâmetros especificados.
        self.pspnet = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
        
        # Opcional: congela os parâmetros do encoder
        if freeze_encoder:
            for param in self.pspnet.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executa o forward pass do modelo.
        
        Args:
            x (torch.Tensor): Tensor (B, 13, H, W).
        
        Returns:
            torch.Tensor: Logits (B, classes, H, W).
        """
        return self.pspnet(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Realiza a predição multiclasse usando softmax + argmax.
        Cada pixel recebe o rótulo mais provável em {0,1,2,3}.
        
        Args:
            x (torch.Tensor): Tensor (B, 13, H, W).
        
        Returns:
            torch.Tensor: Mapa de rótulos (B, H, W) com valores em {0,1,2,3}.
        """
        self.eval()
        logits = self.forward(x)              # (B, 4, H, W)
        probs = torch.softmax(logits, dim=1)  # (B, 4, H, W)
        labels = probs.argmax(dim=1)          # (B, H, W)
        return labels
