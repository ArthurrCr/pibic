import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualDoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),  # Spatial dropout para features espaciais
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.1) 
        )
        self.residual = nn.Identity()
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.final_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv(x)
        x += residual
        return self.final_activation(x)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        
        # Encoder
        in_ch = in_channels
        for feature in features:
            self.downs.append(ResidualDoubleConv(in_ch, feature))
            in_ch = feature
        
        # Bottleneck
        self.bottleneck = ResidualDoubleConv(features[-1], features[-1]*2)
        
        # Decoder com atenção
        for feature in reversed(features):
            self.ups.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(feature*2, feature, kernel_size=3, padding=1, bias=False)
                )
            )
            self.attention_blocks.append(AttentionBlock(F_g=feature, F_l=feature, F_int=feature//2))
            self.ups.append(ResidualDoubleConv(feature*2, feature))
        
        # Camada final
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # Inicialização
        self._initialize_weights()

    def forward(self, x):
        skip_connections = []
        
        # Downsampling
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, 2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Upsampling com atenção
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            # Upsample
            x = self.ups[idx](x)
            
            # Atenção
            skip = skip_connections[idx//2]
            attn = self.attention_blocks[idx//2](x, skip)
            
            # Concatenação
            if x.shape != attn.shape:
                x = self._resize(x, size=attn.shape[2:])
            x = torch.cat([attn, x], dim=1)
            
            # Conv dupla
            x = self.ups[idx+1](x)
        
        return self.final_conv(x)
    
    def _resize(self, x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
