import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBnSilu(nn.Module):
    """Convolução com SiLU e padding 'same'"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2  # Padding 'same'
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResidualCSPBlock(nn.Module):
    """Bloco CSP Residual com atenção - Adaptado do Cloud-Net"""
    def __init__(self, in_channels, out_channels, num_blocks=1):
        super().__init__()
        hidden_dim = out_channels // 2
        
        self.conv1 = ConvBnSilu(in_channels, hidden_dim, 1)
        self.conv2 = ConvBnSilu(in_channels, hidden_dim, 1)
        
        # Blocos residuais
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                ConvBnSilu(hidden_dim, hidden_dim, 3),
                ConvBnSilu(hidden_dim, hidden_dim, 3)
            ) for _ in range(num_blocks)]
        )
        
        # Atenção adaptativa do Cloud-Net
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnSilu(hidden_dim, hidden_dim // 4, 1),
            nn.Conv2d(hidden_dim // 4, hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.fusion = ConvBnSilu(hidden_dim * 2, out_channels, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = self.blocks(x2) + x2  # Conexão residual
        x2 = x2 * self.attention(x2)
        return self.fusion(torch.cat([x1, x2], dim=1))

class CSPDarknet(nn.Module):
    """Backbone com downsampling adaptativo"""
    def __init__(self):
        super().__init__()
        self.stem = ConvBnSilu(3, 32)
        
        # Downsampling com atenção à paridade das dimensões
        self.stage1 = nn.Sequential(
            ConvBnSilu(32, 64, stride=2),
            ResidualCSPBlock(64, 64, 1)
        )
        
        self.stage2 = nn.Sequential(
            ConvBnSilu(64, 128, stride=2),
            ResidualCSPBlock(128, 128, 2)
        )
        
        self.stage3 = nn.Sequential(
            ConvBnSilu(128, 256, stride=2),
            ResidualCSPBlock(256, 256, 2)
        )
        
        self.stage4 = nn.Sequential(
            ConvBnSilu(256, 512, stride=2),
            ResidualCSPBlock(512, 512, 1)
        )

    def forward(self, x):
        s1 = self.stem(x)       # [B,32,H,W]
        s2 = self.stage1(s1)    # [B,64,H/2,W/2]
        s3 = self.stage2(s2)    # [B,128,H/4,W/4]
        s4 = self.stage3(s3)    # [B,256,H/8,W/8]
        s5 = self.stage4(s4)    # [B,512,H/16,W/16]
        return s2, s3, s4, s5

class AdaptivePAN(nn.Module):
    """Neck PAN com up/downsampling adaptativo"""
    def __init__(self):
        super().__init__()
        # Top-down
        self.lat_conv1 = ConvBnSilu(512, 256)
        self.lat_conv2 = ConvBnSilu(256, 256)
        self.lat_conv3 = ConvBnSilu(128, 256)
        
        # Bottom-up
        self.down_conv1 = ConvBnSilu(256, 256, stride=2)
        self.down_conv2 = ConvBnSilu(256, 256, stride=2)
        self.down_conv3 = ConvBnSilu(256, 512, stride=2)

    def forward(self, s2, s3, s4, s5):
        # Top-down com interpolação adaptativa
        p5 = self.lat_conv1(s5)
        p4 = F.interpolate(p5, size=s4.shape[2:], mode='nearest') + self.lat_conv2(s4)
        p3 = F.interpolate(p4, size=s3.shape[2:], mode='nearest') + self.lat_conv3(s3)
        
        # Bottom-up
        n4 = self.down_conv1(p3) + p4
        n5 = self.down_conv2(n4) + p5
        n6 = self.down_conv3(n5)
        
        return p3, n4, n5, n6

class CloudSegmentationHead(nn.Module):
    """Head de segmentação estilo U-Net com skip connections (CORRIGIDO)"""
    def __init__(self, num_classes=1):
        super().__init__()
        # Decoder com canais ajustados
        self.up1 = nn.ConvTranspose2d(256, 256, 2, stride=2)  # Alterado de 512 para 256
        self.conv1 = ResidualCSPBlock(256 + 256, 256)  # Concatena com s4 (256 canais)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = ResidualCSPBlock(128 + 128, 128)  # Concatena com s3 (128 canais)
        
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = ResidualCSPBlock(64 + 64, 64)    # Concatena com s2 (64 canais)
        
        self.final = nn.Sequential(
            ConvBnSilu(64, 32),
            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, x, s2, s3, s4):
        # x: [B,256,H/16,W/16] (n5 do neck)
        x = self.up1(x)                     # [B,256,H/8,W/8]
        x = torch.cat([x, s4], dim=1)       # [B,512,H/8,W/8]
        x = self.conv1(x)                   # [B,256,H/8,W/8]
        
        x = self.up2(x)                     # [B,128,H/4,W/4]
        x = torch.cat([x, s3], dim=1)       # [B,256,H/4,W/4]
        x = self.conv2(x)                   # [B,128,H/4,W/4]
        
        x = self.up3(x)                     # [B,64,H/2,W/2]
        x = torch.cat([x, s2], dim=1)       # [B,128,H/2,W/2]
        x = self.conv3(x)                   # [B,64,H/2,W/2]
        
        return self.final(x)                # [B,num_classes,H/2,W/2]
    
class YOLOCloud(nn.Module):
    """Arquitetura final integrando técnicas do YOLO, Cloud-Net e U-Net"""
    def __init__(self, num_classes=1):
        super().__init__()
        self.backbone = CSPDarknet()
        self.neck = AdaptivePAN()
        self.head = CloudSegmentationHead(num_classes)

    def forward(self, x):
        s2, s3, s4, s5 = self.backbone(x)
        p3, n4, n5, _ = self.neck(s2, s3, s4, s5)
        mask = self.head(n5, s2, s3, s4)
        return F.interpolate(mask, size=x.shape[2:], mode='bilinear', align_corners=False)