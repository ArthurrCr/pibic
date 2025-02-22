import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision.models import resnet50, ResNet50_Weights


class SpatialAttention(nn.Module):
    """Mecanismo de atenção espacial baseado em MOHAJERANI et al. (2019a)."""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        x: Tensor (B, C, H, W)
        Retorna: Tensor com mesmo shape, aplicando atenção no espaço.
        """
        return x * self.conv(x)


class ResNetBackbone(nn.Module):
    """
    Backbone ResNet-50 ajustado para lidar com 13 bandas de entrada.
    Retorna features c2, c3, c4, c5 (camadas intermediárias).
    """
    def __init__(self, in_channels=13):
        super().__init__()
        base_model = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Ajusta a conv1 para aceitar 13 canais
        old_conv = base_model.conv1
        base_model.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        # Copia pesos dos 3 canais pré-treinados e inicializa aleatoriamente os canais extras
        with torch.no_grad():
            base_model.conv1.weight[:, :3, :, :] = old_conv.weight
            init.kaiming_normal_(base_model.conv1.weight[:, 3:, :, :])

        # Define as camadas conforme a arquitetura original
        self.stem = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool
        )
        self.layer1 = base_model.layer1  # => c2 (B, 256, H/4, W/4)
        self.layer2 = base_model.layer2  # => c3 (B, 512, H/8, W/8)
        self.layer3 = base_model.layer3  # => c4 (B, 1024, H/16, W/16)
        self.layer4 = base_model.layer4  # => c5 (B, 2048, H/32, W/32)

    def forward(self, x):
        """
        x: Tensor shape (B, 13, H, W)
        Retorna c2, c3, c4, c5
          - c2: (B, 256, H/4,  W/4)
          - c3: (B, 512, H/8,  W/8)
          - c4: (B,1024, H/16, W/16)
          - c5: (B,2048, H/32, W/32)
        """
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5


class FPN(nn.Module):
    """
    FPN melhorado com atenção espacial (LIN et al., 2017 + MOHAJERANI et al., 2019a).
    Recebe c2, c3, c4, c5 do backbone e gera p2, p3, p4, p5 com mesma dimensionalidade.
    """
    def __init__(self, out_channels=256):
        super().__init__()
        # Camadas laterais de redução para 'out_channels'
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(256,  out_channels, 1),  # para c2
            nn.Conv2d(512,  out_channels, 1),  # para c3
            nn.Conv2d(1024, out_channels, 1),  # para c4
            nn.Conv2d(2048, out_channels, 1)   # para c5
        ])
        
        # Módulos de atenção espacial em cada nível
        self.attentions = nn.ModuleList([
            SpatialAttention(out_channels) for _ in range(4)
        ])
        
        # Convoluções de suavização (3x3) pós-fusão
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in range(4)
        ])

    def forward(self, features):
        """
        features: [c2, c3, c4, c5]
        Retorna [p2, p3, p4, p5], cada um com 'out_channels'.
        """
        c2, c3, c4, c5 = features
        
        # Projeção lateral
        p5 = self.lateral_convs[3](c5)
        p4 = self.lateral_convs[2](c4)
        p3 = self.lateral_convs[1](c3)
        p2 = self.lateral_convs[0](c2)
        
        # Top-down + atenção
        p4 = self.attentions[2]( p4 + F.interpolate(p5, size=p4.shape[-2:], mode='nearest') )
        p3 = self.attentions[1]( p3 + F.interpolate(p4, size=p3.shape[-2:], mode='nearest') )
        p2 = self.attentions[0]( p2 + F.interpolate(p3, size=p2.shape[-2:], mode='nearest') )
        
        # Não esquece que p5 também pode ter atenção
        p5 = self.attentions[3](p5)

        # Suavização (opcional, mas comum no FPN)
        p2 = self.smooth_convs[0](p2)
        p3 = self.smooth_convs[1](p3)
        p4 = self.smooth_convs[2](p4)
        p5 = self.smooth_convs[3](p5)
        
        return [p2, p3, p4, p5]


class FPNHeadSegmentation(nn.Module):
    """
    Cabeça de segmentação com fusão multi-escala (LIN et al., 2017) e
    conexão residual de características de baixo nível (RONNEBERGER et al., 2015).
    """
    def __init__(self, in_channels=256, num_classes=4):
        """
        in_channels: canais vindos do FPN (normalmente 256).
        num_classes: número de classes (4 por padrão).
        """
        super().__init__()
        # Pesos aprendíveis para fusão (um para cada nível p2, p3, p4, p5)
        self.weights = nn.Parameter(torch.ones(4))

        self.fuse_conv = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )
        
        # Conexão residual do c2
        self.c2_conv = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, pyramid_features, c2):
        """
        pyramid_features: [p2, p3, p4, p5], cada (B, 256, H/4, W/4) (por ex.)
        c2: (B, 256, H/4, W/4) do backbone
        Retorna: (B, num_classes, H/4, W/4) (mais tarde será upsampleado).
        """
        p2, p3, p4, p5 = pyramid_features
        
        # Redimensiona tudo para o tamanho de p2 (topo da pirâmide)
        h, w = p2.shape[-2], p2.shape[-1]
        feats_resized = [
            F.interpolate(f, size=(h, w), mode='bilinear', align_corners=False)
            for f in [p2, p3, p4, p5]
        ]
        
        # Fusão ponderada
        # Normaliza os pesos para que somem 1 (via softmax)
        w_norm = torch.softmax(self.weights, dim=0)
        fused = 0
        for i, f in enumerate(feats_resized):
            fused += w_norm[i] * f  # soma ponderada

        # Cria o mapa de previsão a partir da fusão
        pred_fuse = self.fuse_conv(fused)

        # Conexão residual de baixo nível (c2)
        pred_c2 = self.c2_conv(c2)
        
        # Soma final
        return pred_fuse + pred_c2


class CloudFPN(nn.Module):
    """
    Modelo FPN final, integrando:
      - Backbone ResNet-50 (ajustado para 13 bandas)
      - FPN com atenção espacial
      - Cabeça de segmentação (4 classes por padrão)
    """
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = ResNetBackbone(in_channels=13)
        self.fpn = FPN(out_channels=256)
        self.head = FPNHeadSegmentation(in_channels=256, num_classes=num_classes)

    def forward(self, x):
        """
        x: Tensor (B, 13, H, W) => 13 bandas de Sentinel-2
        Retorna: (B, num_classes, H, W)
        """
        original_size = x.shape[-2:]  # (H, W)
        
        # Extração de características
        c2, c3, c4, c5 = self.backbone(x)
        
        # Pirâmide de características FPN
        pyramid = self.fpn([c2, c3, c4, c5])  # [p2, p3, p4, p5]

        # Cabeça de segmentação
        logits = self.head(pyramid, c2)
        
        # Upsample para retornar ao tamanho original
        logits = F.interpolate(logits, size=original_size, mode='bilinear', align_corners=False)
        return logits
