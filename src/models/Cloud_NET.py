import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ===============================
# 1. Módulos de Atenção (CBAM)
# ===============================

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

# ====================================
# 2. Módulo ASPP (Atrous Spatial Pyramid Pooling)
# ====================================

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=[1, 3, 6, 12, 18], dropout=0.2):
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
        x = self.project(torch.cat(feats + [gp], dim=1))
        return x
    
# ====================================
# 3. Decoder FPN Integrado com CBAM
# ====================================

class FPNDecoder(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        """
        in_channels_list: lista com número de canais para cada feature map de diferentes escalas.
        out_channels: número de canais para o mapa final do decoder.
        """
        super(FPNDecoder, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        
        # Módulo CBAM para refinar o mapa na resolução mais alta
        self.cbam = CBAM(out_channels)
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def forward(self, features):
        # features: lista de mapas [c1, c2, c3, c4]
        # c1: maior resolução; c4: menor resolução.
        c1, c2, c3, c4 = features
        
        p4 = self.lateral_convs[3](c4)
        p3 = self.lateral_convs[2](c3) + self.upsample(p4)
        p2 = self.lateral_convs[1](c2) + self.upsample(p3)
        p1 = self.lateral_convs[0](c1) + self.upsample(p2)
        
        # Refinamento com convoluções adicionais
        p4 = self.fpn_convs[3](p4)
        p3 = self.fpn_convs[2](p3)
        p2 = self.fpn_convs[1](p2)
        p1 = self.fpn_convs[0](p1)
        
        # Aplica atenção no mapa de maior resolução
        p1 = self.cbam(p1)
        out = self.final_conv(p1)
        return out

# ====================================
# 4. UpdatedCloudNet: Arquitetura Completa
# ====================================

class UpdatedCloudNet(nn.Module):
    def __init__(self, num_classes=1):
        """
        num_classes: número de classes para segmentação (1 para máscara binária).
        """
        super(UpdatedCloudNet, self).__init__()
        # Utiliza ResNet-18 pré-treinado como backbone
        resnet = models.resnet18(pretrained=True)
        
        # Camadas iniciais do ResNet-18
        self.initial = nn.Sequential(
            resnet.conv1,   # (B, 64, H/2, W/2)
            resnet.bn1,
            resnet.relu,
            resnet.maxpool  # (B, 64, H/4, W/4)
        )
        self.layer1 = resnet.layer1  # (B, 64, H/4, W/4)
        self.layer2 = resnet.layer2  # (B, 128, H/8, W/8)
        self.layer3 = resnet.layer3  # (B, 256, H/16, W/16)
        self.layer4 = resnet.layer4  # (B, 512, H/32, W/32)
        
        # ASPP aplicado na camada mais profunda (layer4)
        self.aspp = ASPP(in_ch=512, out_ch=256, dilations=[1, 3, 6, 12, 18])
        
        # Decoder FPN: integra features de layer1, layer2, layer3 e a saída do ASPP
        # Note que os canais esperados são: layer1 -> 64, layer2 -> 128, layer3 -> 256, e ASPP(layer4) -> 256.
        self.fpn_decoder = FPNDecoder(in_channels_list=[64, 128, 256, 256], out_channels=64)
        
        # Cabeça final de segmentação
        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x):
        # Extração de features com o backbone
        x = self.initial(x)       # (B, 64, H/4, W/4)
        c1 = self.layer1(x)       # (B, 64, H/4, W/4)
        c2 = self.layer2(c1)      # (B, 128, H/8, W/8)
        c3 = self.layer3(c2)      # (B, 256, H/16, W/16)
        c4 = self.layer4(c3)      # (B, 512, H/32, W/32)
        
        # Processa c4 com o ASPP para capturar contexto multiescala
        c4_aspp = self.aspp(c4)   # (B, 256, H/32, W/32)
        
        # Integra as features com o FPN
        fpn_out = self.fpn_decoder([c1, c2, c3, c4_aspp])  # (B, 64, H/4, W/4)
        
        # Upsample para a resolução original
        out = F.interpolate(fpn_out, scale_factor=4, mode='bilinear', align_corners=False)
        
        # Gera a máscara de segmentação
        seg = self.seg_head(out)
        return seg
