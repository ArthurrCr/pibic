import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import pytorch_lightning as pl

class GeoStatisticalLayer(nn.Module):
    """Inspirado em TAYEBI et al. (2023) para características geoestatísticas"""
    def __init__(self):
        super().__init__()
        self.gabor = nn.Conv2d(3, 16, kernel_size=5, padding=2)  # Filtros estilo Gabor
        self.reduce = nn.Conv2d(19, 3, kernel_size=1)  # Mantém canais de entrada originais

    def forward(self, x):
        texture = self.gabor(x)
        return self.reduce(torch.cat([x, texture], dim=1))

class SpatialAttention(nn.Module):
    """Mecanismo de atenção espacial baseado em MOHAJERANI et al. (2019a)"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.conv(x)

class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        self.stem = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool
        )
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5

class FPN(nn.Module):
    """FPN melhorado com atenção espacial (LIN et al., 2017 + MOHAJERANI et al., 2019a)"""
    def __init__(self, out_channels=256):
        super().__init__()
        
        # Camadas laterais
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(256, out_channels, 1),
            nn.Conv2d(512, out_channels, 1),
            nn.Conv2d(1024, out_channels, 1),
            nn.Conv2d(2048, out_channels, 1)
        ])
        
        # Camadas de atenção
        self.attentions = nn.ModuleList([
            SpatialAttention(out_channels) for _ in range(4)
        ])
        
        # Suavização
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in range(4)
        ])

    def forward(self, features):
        c2, c3, c4, c5 = features
        
        # Projeção lateral
        p5 = self.lateral_convs[3](c5)
        p4 = self.lateral_convs[2](c4)
        p3 = self.lateral_convs[1](c3)
        p2 = self.lateral_convs[0](c2)
        
        # Top-down com atenção
        p4 = self.attentions[2](p4 + F.interpolate(p5, p4.shape[-2:], mode='nearest'))
        p3 = self.attentions[1](p3 + F.interpolate(p4, p3.shape[-2:], mode='nearest'))
        p2 = self.attentions[0](p2 + F.interpolate(p3, p2.shape[-2:], mode='nearest'))
        
        # Suavização final
        return [self.smooth_convs[i](p) for i, p in enumerate([p2, p3, p4, p5])]

class FPNHeadSegmentation(nn.Module):
    """Cabeça de segmentação com fusão multi-escala aprimorada"""
    def __init__(self, in_channels=256, num_classes=1):
        super().__init__()
        
        # Pesos aprendíveis para fusão (LIN et al., 2017)
        self.weights = nn.Parameter(torch.ones(4))
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_classes, 1)
        )
        
        # Conexão residual do C2 (RONNEBERGER et al., 2015)
        self.c2_conv = nn.Conv2d(256, num_classes, 1)

    def forward(self, pyramid_features, c2):
        p2, p3, p4, p5 = pyramid_features
        
        # Fusão ponderada
        h, w = p2.shape[-2:]
        features = [
            F.interpolate(f, (h,w), mode='bilinear') 
            for f in [p2, p3, p4, p5]
        ]
        fused = sum(w*f for w,f in zip(F.softmax(self.weights,0), features))
        
        # Combinação com características de baixo nível
        return self.fuse_conv(fused) + self.c2_conv(c2)

class CloudLoss(nn.Module):
    """Loss combinada para segmentação de nuvens (ZHU; WOODCOCK, 2012)"""
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target)
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        dice = 1 - (2. * intersection) / (pred.sum() + target.sum() + 1e-8)
        return self.alpha * dice + (1 - self.alpha) * bce

class CloudFPN(pl.LightningModule):
    """Modelo final integrando todas as melhorias"""
    def __init__(self, num_classes=1, crf=False):
        super().__init__()
        self.geo_layer = GeoStatisticalLayer()
        self.backbone = ResNetBackbone()
        self.fpn = FPN()
        self.head = FPNHeadSegmentation(num_classes=num_classes)
        self.loss_fn = CloudLoss()
        self.crf = crf

    def forward(self, x):
        # Pré-processamento geoestatístico
        x = self.geo_layer(x)
        
        # Extração de características
        c2, c3, c4, c5 = self.backbone(x)
        pyramid = self.fpn([c2, c3, c4, c5])
        
        # Segmentação
        logits = self.head(pyramid, c2)
        logits = F.interpolate(logits, x.shape[-2:], mode='bilinear')
        
        # Pós-processamento (YUAN; HU, 2015)
        if self.crf and not self.training:
            return self.crf_refinement(x, logits)
        return logits