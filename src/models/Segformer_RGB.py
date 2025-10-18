import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class CloudSegformer_RGB(nn.Module):
    """
    Segformer para segmentação de imagem RGB (3 bandas) em 4 classes exclusivas
    (ex.: clear, thick cloud, thin cloud, cloud shadow).
    """
    def __init__(self,
                 encoder_name='mit_b4',
                 encoder_weights='imagenet',
                 in_channels=3,
                 classes=4,
                 freeze_encoder=False):

        super(CloudSegformer_RGB, self).__init__()
        
        # Cria o modelo Segformer com os parâmetros especificados.
        self.segformer = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
        
        # Opcional: congela os parâmetros do encoder
        if freeze_encoder:
            for param in self.segformer.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executa o forward pass do modelo.
        
        Args:
            x (torch.Tensor): Tensor de entrada (B, 3, H, W).

        Returns:
            torch.Tensor: Logits (B, classes, H, W).
        """
        return self.segformer(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Realiza a predição de segmentação (4 classes exclusivas).
        Aplica softmax e retorna o índice da classe mais provável.
        
        Args:
            x (torch.Tensor): Tensor de entrada (B, 3, H, W).
        
        Returns:
            torch.Tensor: Mapa de rótulos (B, H, W) com valores em {0,1,2,3}.
        """
        self.eval()
        logits = self.forward(x)             # (B, 4, H, W)
        probs = torch.softmax(logits, dim=1) # (B, 4, H, W)
        labels = probs.argmax(dim=1)         # (B, H, W)
        return labels
