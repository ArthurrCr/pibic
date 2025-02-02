import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------
# Bloco Residual com Duas Convoluções e Dropout
# ---------------------------------------------------
class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        """
        Bloco convolucional residual:
          - Duas camadas convolucionais 3x3 com BatchNorm e ReLU.
          - Conexão residual (ajustada com 1x1 se necessário).
          - Dropout (opcional) após cada ativação para regularização.
        """
        super(ResidualConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        
        # Se o número de canais mudar, ajusta a conexão residual com um 1x1
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()
    
    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.dropout is not None:
            out = self.dropout(out)
        out += residual
        out = F.relu(out)
        return out

# ---------------------------------------------------
# Arquitetura CloudNet (baseada em U-Net com blocos residuais)
# ---------------------------------------------------
class CloudNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_filters=64, dropout=0.2):
        """
        Parâmetros:
          - in_channels: número de canais de entrada (ex.: 3 para RGB).
          - out_channels: número de canais da máscara de saída (ex.: 1 para segmentação binária).
          - base_filters: número de filtros na primeira camada; os demais dobram progressivamente.
          - dropout: probabilidade de dropout nos blocos residuais.
        """
        super(CloudNet, self).__init__()
        # Encoder
        self.enc1 = ResidualConvBlock(in_channels, base_filters, dropout=dropout)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = ResidualConvBlock(base_filters, base_filters * 2, dropout=dropout)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = ResidualConvBlock(base_filters * 2, base_filters * 4, dropout=dropout)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc4 = ResidualConvBlock(base_filters * 4, base_filters * 8, dropout=dropout)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Camada intermediária (bottleneck)
        self.bottleneck = ResidualConvBlock(base_filters * 8, base_filters * 16, dropout=dropout)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.dec4 = ResidualConvBlock(base_filters * 16, base_filters * 8, dropout=dropout)
        
        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = ResidualConvBlock(base_filters * 8, base_filters * 4, dropout=dropout)
        
        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = ResidualConvBlock(base_filters * 4, base_filters * 2, dropout=dropout)
        
        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = ResidualConvBlock(base_filters * 2, base_filters, dropout=dropout)
        
        self.out_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)    # Saída: base_filters canais
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)   # Saída: base_filters * 2 canais
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)   # Saída: base_filters * 4 canais
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)   # Saída: base_filters * 8 canais
        p4 = self.pool4(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)  # Saída: base_filters * 16 canais
        
        # Decoder com upsampling e concatenação (skip connections)
        u4 = self.up4(b)
        cat4 = torch.cat([u4, e4], dim=1)
        d4 = self.dec4(cat4)
        
        u3 = self.up3(d4)
        cat3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(cat3)
        
        u2 = self.up2(d3)
        cat2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(cat2)
        
        u1 = self.up1(d2)
        cat1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(cat1)
        
        out = self.out_conv(d1)
        return out
