import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch.nn.init as init

# =============================================================================
# Componentes CBAM (Convolutional Block Attention Module)
# =============================================================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out) * x


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        return out


# =============================================================================
# Módulo ASPP (Atrous Spatial Pyramid Pooling)
# =============================================================================
class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=[1, 3, 6, 12, 18], dropout=0.1):
        """
        Para segmentação de nuvens, ajustamos o dropout para 0.1 
        e mantemos a extração de contexto multiescala.
        """
        super(ASPP, self).__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout
        
        for d in dilations:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=self.dropout)
            ))
        self.global_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=self.dropout)
        )
        self.project = nn.Sequential(
            nn.Conv2d((len(dilations) + 1) * out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=self.dropout)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        feats = [conv(x) for conv in self.convs]
        gp = F.interpolate(self.global_conv(x), size=size, mode='bilinear', align_corners=False)
        x_cat = torch.cat(feats + [gp], dim=1)
        x_out = self.project(x_cat)
        return x_out


# =============================================================================
# Módulo de Preservação de Resolução Espacial
# =============================================================================
class SpatialResolutionPreservation(nn.Module):
    """
    Processa a feature de alta resolução (ex.: saída de layer1)
    para enfatizar detalhes espaciais, antes de fundir com o decoder.
    """
    def __init__(self, in_channels, out_channels):
        super(SpatialResolutionPreservation, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


# =============================================================================
# Decoder FPN com Conexões de Salto Tipo U-Net Integrado com CBAM
# =============================================================================
class FPNDecoder(nn.Module):
    """
    O decoder integra as features de diferentes escalas usando conexões tipo U-Net.
    Concatena a informação upsampled com a lateral, segue de convoluções e CBAM.
    """
    def __init__(self, in_channels_list, out_channels):
        super(FPNDecoder, self).__init__()
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.cbam = CBAM(out_channels)
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def forward(self, features):
        # features: [c1, c2, c3, c4], onde c4 vem do ASPP
        c1, c2, c3, c4 = features
        
        p4 = self.lateral_convs[3](c4)
        
        # Nível 3
        p4_up = self.upsample(p4)
        c3_lat = self.lateral_convs[2](c3)
        p3 = self.conv3(torch.cat([p4_up, c3_lat], dim=1))
        
        # Nível 2
        p3_up = self.upsample(p3)
        c2_lat = self.lateral_convs[1](c2)
        p2 = self.conv2(torch.cat([p3_up, c2_lat], dim=1))
        
        # Nível 1
        p2_up = self.upsample(p2)
        c1_lat = self.lateral_convs[0](c1)
        p1 = self.conv1(torch.cat([p2_up, c1_lat], dim=1))
        
        # Refinamento com CBAM + convolução final
        p1 = self.cbam(p1)
        out = self.final_conv(p1)
        return out


# =============================================================================
# UpdatedCloudNet – Versão Otimizada para Segmentação com 13 bandas e 4 classes
# =============================================================================
class UpdatedCloudNet(nn.Module):
    def __init__(self, num_classes=4):
        """
        Modelo ajustado para:
          - 13 bandas de entrada
          - 4 classes de saída (ex.: Clear, Thick Cloud, Thin Cloud, Shadow).
        """
        super(UpdatedCloudNet, self).__init__()
        # Carrega ResNet-18 pré-treinado
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Ajuste para 13 canais de entrada
        old_conv = resnet.conv1
        in_channels = 13
        resnet.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False  # o BN subsequente já lida com bias
        )
        
        # Inicializa os pesos dos 3 primeiros canais com os do modelo pré-treinado
        # e randomiza os 10 canais restantes.
        with torch.no_grad():
            resnet.conv1.weight[:, :3, :, :] = old_conv.weight
            init.kaiming_normal_(resnet.conv1.weight[:, 3:, :, :])

        # Define a parte inicial (stem)
        self.initial = nn.Sequential(
            resnet.conv1,  # (B, 64, H/2, W/2)
            resnet.bn1,
            resnet.relu,
            resnet.maxpool  # (B, 64, H/4, W/4)
        )
        self.layer1 = resnet.layer1   # (B, 64,  H/4,  W/4)
        self.layer2 = resnet.layer2   # (B, 128, H/8,  W/8)
        self.layer3 = resnet.layer3   # (B, 256, H/16, W/16)
        self.layer4 = resnet.layer4   # (B, 512, H/32, W/32)
        
        # ASPP para o nível mais profundo
        self.aspp = ASPP(in_ch=512, out_ch=256, dilations=[1, 3, 6, 12, 18], dropout=0.1)
        
        # Decoder estilo FPN
        decoder_channels = 128
        self.fpn_decoder = FPNDecoder(
            in_channels_list=[64, 128, 256, 256],  # c1=64, c2=128, c3=256, c4_aspp=256
            out_channels=decoder_channels
        )
        
        # Preservação espacial (usa saída de layer1)
        self.spatial_preservation = SpatialResolutionPreservation(
            in_channels=64,
            out_channels=decoder_channels
        )
        
        # Cabeça final de segmentação (num_classes=4 => (B,4,H,W))
        self.seg_head = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)
    
    def forward(self, x):
        """
        x: (B, 13, H, W) -> 13 bandas Sentinel-2
        Retorna: (B, num_classes, H, W).
        """
        # Encoder
        x = self.initial(x)       # (B, 64, H/4, W/4)
        c1 = self.layer1(x)       # (B, 64, H/4, W/4)
        c2 = self.layer2(c1)      # (B, 128, H/8, W/8)
        c3 = self.layer3(c2)      # (B, 256, H/16, W/16)
        c4 = self.layer4(c3)      # (B, 512, H/32, W/32)
        
        # ASPP no nível mais profundo
        c4_aspp = self.aspp(c4)   # (B, 256, H/32, W/32)
        
        # Decoder
        fpn_out = self.fpn_decoder([c1, c2, c3, c4_aspp])  # (B, 128, H/4, W/4)
        
        # Preservação espacial + fusão
        preserved = self.spatial_preservation(c1)          # (B, 128, H/4, W/4)
        combined = fpn_out + preserved                     # (B, 128, H/4, W/4)
        
        # Upsample para resolução original
        out = F.interpolate(combined, scale_factor=4, mode='bilinear', align_corners=False)
        
        # Camada de segmentação
        seg = self.seg_head(out)  # (B, num_classes, H, W)
        return seg
