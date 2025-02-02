import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# --------------------------------------------------
# 1. Backbone ResNet50 para extração de features
# --------------------------------------------------
class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        """
        Utiliza o ResNet50 pré-treinado para extrair mapas de features:
          - C2: saída de layer1 (após maxpool)
          - C3: saída de layer2
          - C4: saída de layer3
          - C5: saída de layer4
        """
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.conv1 = resnet.conv1       # Saída: 64 canais, stride=2
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool   # Reduz pela metade a resolução
        self.layer1 = resnet.layer1     # C2 (geralmente 256 canais)
        self.layer2 = resnet.layer2     # C3 (512 canais)
        self.layer3 = resnet.layer3     # C4 (1024 canais)
        self.layer4 = resnet.layer4     # C5 (2048 canais)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)   # Após o maxpool, a resolução já diminui
        c2 = self.layer1(x)   # Exemplo: saída com 256 canais
        c3 = self.layer2(c2)  # Exemplo: 512 canais
        c4 = self.layer3(c3)  # Exemplo: 1024 canais
        c5 = self.layer4(c4)  # Exemplo: 2048 canais
        return c2, c3, c4, c5

# --------------------------------------------------
# 2. Módulo FPN (Feature Pyramid Network)
# --------------------------------------------------
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        """
        Args:
          in_channels_list: lista com o número de canais de [C2, C3, C4, C5]
          out_channels: número de canais para cada mapa da pirâmide (ex.: 256)
        """
        super().__init__()
        # Camadas laterais (1x1 convolutions)
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
        
        # Camadas de suavização (3x3 convolutions com padding=1)
        self.smooth_convs = nn.ModuleList()
        for _ in in_channels_list:
            self.smooth_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )
    
    def forward(self, features):
        """
        Args:
          features: tupla/lista contendo (c2, c3, c4, c5)
        Retorna:
          Uma lista de mapas de features: [p2, p3, p4, p5]
        """
        c2, c3, c4, c5 = features

        # Projeção lateral
        p5 = self.lateral_convs[3](c5)
        p4 = self.lateral_convs[2](c4)
        p3 = self.lateral_convs[1](c3)
        p2 = self.lateral_convs[0](c2)

        # Top-down: upsampling e soma
        p4 = p4 + F.interpolate(p5, size=p4.shape[-2:], mode='nearest')
        p3 = p3 + F.interpolate(p4, size=p3.shape[-2:], mode='nearest')
        p2 = p2 + F.interpolate(p3, size=p2.shape[-2:], mode='nearest')

        # Suavização com convoluções 3x3
        p5 = self.smooth_convs[3](p5)
        p4 = self.smooth_convs[2](p4)
        p3 = self.smooth_convs[1](p3)
        p2 = self.smooth_convs[0](p2)

        return [p2, p3, p4, p5]

# --------------------------------------------------
# 3. FPN-Net: Backbone + FPN
# --------------------------------------------------
class FPNNet(nn.Module):
    def __init__(self, backbone_pretrained=True, out_channels=256):
        """
        Combina o backbone ResNet50 com a FPN.
        """
        super().__init__()
        # Para ResNet50, os canais de saída dos blocos são:
        # C2: 256, C3: 512, C4: 1024, C5: 2048
        self.backbone = ResNetBackbone(pretrained=backbone_pretrained)
        self.fpn = FPN(in_channels_list=[256, 512, 1024, 2048], out_channels=out_channels)

    def forward(self, x):
        # Extração de features pelo backbone
        features = self.backbone(x)   # c2, c3, c4, c5
        # Construção da pirâmide de features
        pyramid_features = self.fpn(features)
        return pyramid_features
