import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Blocos Básicos do YOLO (CSPDarknet Style)
# ----------------------------------------------------------------------------
class ConvBnSilu(nn.Module):
    """Convolução com SiLU (YOLOv8)"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CSPBlock(nn.Module):
    """Bloco CSP com atenção para nuvens (YOLO + Cloud-Net)"""
    def __init__(self, in_channels, out_channels, num_blocks=1):
        super().__init__()
        hidden_dim = out_channels // 2
        self.conv1 = ConvBnSilu(in_channels, hidden_dim, 1)
        self.conv2 = ConvBnSilu(in_channels, hidden_dim, 1)
        self.blocks = nn.Sequential(*[ConvBnSilu(hidden_dim, hidden_dim, 3) for _ in range(num_blocks)])
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnSilu(hidden_dim, hidden_dim // 4, 1),
            ConvBnSilu(hidden_dim // 4, hidden_dim, 1),
            nn.Sigmoid()
        )
        self.conv_out = ConvBnSilu(hidden_dim * 2, out_channels, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = self.blocks(x2)
        x2 = x2 * self.attention(x2)  # Atenção espacial
        return self.conv_out(torch.cat([x1, x2], dim=1))


# ----------------------------------------------------------------------------
# Backbone CSPDarknet (YOLO Style)
# ----------------------------------------------------------------------------
class CSPDarknet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvBnSilu(3, 32, 3, padding=1)
        self.stage1 = CSPBlock(32, 64, num_blocks=1)
        self.stage2 = CSPBlock(64, 128, num_blocks=2)
        self.stage3 = CSPBlock(128, 256, num_blocks=2)
        self.stage4 = CSPBlock(256, 512, num_blocks=1)

        # Downsample layers
        self.downsample1 = ConvBnSilu(32, 64, 3, stride=2, padding=1)
        self.downsample2 = ConvBnSilu(64, 128, 3, stride=2, padding=1)
        self.downsample3 = ConvBnSilu(128, 256, 3, stride=2, padding=1)
        self.downsample4 = ConvBnSilu(256, 512, 3, stride=2, padding=1)

    def forward(self, x):
        x = self.stem(x)           # [B,32,H,W]
        x = self.downsample1(x)    # [B,64,H/2,W/2]
        c1 = self.stage1(x)        
        x = self.downsample2(c1)   # [B,128,H/4,W/4]
        c2 = self.stage2(x)        
        x = self.downsample3(c2)   # [B,256,H/8,W/8]
        c3 = self.stage3(x)        
        x = self.downsample4(c3)   # [B,512,H/16,W/16]
        c4 = self.stage4(x)        
        return c1, c2, c3, c4


# ----------------------------------------------------------------------------
# Neck PAN (YOLO Style com Multi-Scale Fusion)
# ----------------------------------------------------------------------------
class PANNeck(nn.Module):
    """Path Aggregation Network adaptado para segmentação"""
    def __init__(self):
        super().__init__()
        # Top-down path (FPN)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lat_conv1 = ConvBnSilu(512, 256, 1)
        self.lat_conv2 = ConvBnSilu(256, 128, 1)
        self.lat_conv3 = ConvBnSilu(128, 64, 1)
        
        # Bottom-up path (PAN)
        self.down_conv1 = ConvBnSilu(64, 128, 3, stride=2)
        self.down_conv2 = ConvBnSilu(128, 256, 3, stride=2)
        self.down_conv3 = ConvBnSilu(256, 512, 3, stride=2)

    def forward(self, c1, c2, c3, c4):
        # Top-down
        p4 = self.lat_conv1(c4)                # [B,256,H/16,W/16]
        p3 = self.upsample(p4) + self.lat_conv2(c3)  # [B,128,H/8,W/8]
        p2 = self.upsample(p3) + self.lat_conv3(c2)  # [B,64,H/4,W/4]
        
        # Bottom-up
        n2 = self.down_conv1(p2) + p3          # [B,128,H/8,W/8]
        n3 = self.down_conv2(n2) + p4          # [B,256,H/16,W/16]
        n4 = self.down_conv3(n3)               # [B,512,H/32,W/32]
        
        return p2, n2, n3, n4


# ----------------------------------------------------------------------------
# Head de Segmentação (YOLO Adaptado)
# ----------------------------------------------------------------------------
class SegmentationHead(nn.Module):
    """Head que mantém a filosofia YOLO com multi-scale features"""
    def __init__(self, num_classes=1):
        super().__init__()
        self.conv1 = ConvBnSilu(64, 64, 3)
        self.conv2 = ConvBnSilu(128, 128, 3)
        self.conv3 = ConvBnSilu(256, 256, 3)
        
        self.fusion = ConvBnSilu(64+128+256, 256, 1)
        self.output = nn.Conv2d(256, num_classes, 1)

    def forward(self, p2, n2, n3):
        # Processa features em diferentes escalas
        p2 = self.conv1(p2)        # [B,64,H/4,W/4]
        n2 = self.conv2(n2)        # [B,128,H/8,W/8]
        n3 = self.conv3(n3)        # [B,256,H/16,W/16]
        
        # Fusão multi-resolução
        p2_up = F.interpolate(p2, scale_factor=4, mode="bilinear") 
        n2_up = F.interpolate(n2, scale_factor=2, mode="bilinear")
        fused = torch.cat([p2_up, n2_up, n3], dim=1)  # [B,448,H/4,W/4]
        
        return self.output(self.fusion(fused))


# ----------------------------------------------------------------------------
# Modelo Final
# ----------------------------------------------------------------------------
class YOLOCloud(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.backbone = CSPDarknet()
        self.neck = PANNeck()
        self.head = SegmentationHead(num_classes)
        
    def forward(self, x):
        c1, c2, c3, c4 = self.backbone(x)
        p2, n2, n3, _ = self.neck(c1, c2, c3, c4)
        mask = self.head(p2, n2, n3)
        return F.interpolate(mask, scale_factor=4, mode="bilinear", align_corners=False)