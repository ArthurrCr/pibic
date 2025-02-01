import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1) SEBlock com Fator de Redução Ajustado (reduction=8)
# ============================================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):  # Reduction ajustado
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# ============================================================
# 2) Bloco Multi-Escala com SE Ajustado
# ============================================================
class MultiScaleConvResSE(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=2, reduction=8):  # Reduction=8
        super(MultiScaleConvResSE, self).__init__()
        # Caminhos local e global
        self.conv_local = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_global = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        # Merge + SE
        self.merge_conv = nn.Sequential(
            nn.Conv2d(2*out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()
        self.se = SEBlock(out_channels, reduction=reduction)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.skip_conv(x)
        out_local = self.conv_local(x)
        out_global = self.conv_global(x)
        out = self.merge_conv(torch.cat([out_local, out_global], dim=1))
        out = self.se(out) + identity
        return self.relu(out)

# ============================================================
# 3) ASPP Ampliado com Mais Dilatações
# ============================================================
class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=[1, 3, 6, 12, 18]):  # Mais dilatações
        super(ASPP, self).__init__()
        self.convs = nn.ModuleList()
        for d in dilations:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ))
        
        # Global pooling
        self.global_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d((len(dilations)+1)*out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[2:]
        feats = [conv(x) for conv in self.convs]
        gp = F.interpolate(self.global_conv(x), size=size, mode='bilinear', align_corners=False)
        return self.project(torch.cat(feats + [gp], dim=1))

# ============================================================
# 4) CloudNet Atualizado (Decoder Melhorado)
# ============================================================
class CloudNetPaperPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_ch=32):
        super(CloudNetPaperPlus, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            MultiScaleConvResSE(in_channels, base_ch, dilation=2, reduction=8),
            MultiScaleConvResSE(base_ch, base_ch, dilation=2, reduction=8),
            nn.MaxPool2d(2, 2)
        )
        self.enc2 = nn.Sequential(
            MultiScaleConvResSE(base_ch, 2*base_ch, dilation=2, reduction=8),
            MultiScaleConvResSE(2*base_ch, 2*base_ch, dilation=2, reduction=8),
            nn.MaxPool2d(2, 2)
        )
        
        # Bottleneck com ASPP Ampliado
        self.aspp = ASPP(2*base_ch, 4*base_ch)
        
        # Decoder com Upsampling Bilinear
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(4*base_ch, 2*base_ch, 3, padding=1)
        )
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(2*base_ch, base_ch, 3, padding=1)
        )
        
        # Blocos pós-concatenação (3 blocos por nível)
        self.post_dec2 = nn.Sequential(
            MultiScaleConvResSE(4*base_ch, 2*base_ch, reduction=8),
            MultiScaleConvResSE(2*base_ch, 2*base_ch, reduction=8),
            MultiScaleConvResSE(2*base_ch, 2*base_ch, reduction=8)  # Bloco adicional
        )
        self.post_dec1 = nn.Sequential(
            MultiScaleConvResSE(2*base_ch, base_ch, reduction=8),
            MultiScaleConvResSE(base_ch, base_ch, reduction=8),
            MultiScaleConvResSE(base_ch, base_ch, reduction=8)  # Bloco adicional
        )
        
        self.out_conv = nn.Conv2d(base_ch, out_channels, 1)
        self._init_weights()

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        
        # Bottleneck
        xb = self.aspp(x2)
        
        # Decoder
        xu2 = self.post_dec2(torch.cat([self.dec2(xb), x2], dim=1))
        xu1 = self.post_dec1(torch.cat([self.dec1(xu2), x1], dim=1))
        
        return self.out_conv(xu1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)