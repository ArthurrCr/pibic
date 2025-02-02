import torch
import torch.nn as nn
import torch.nn.functional as F

def crop_tensor(tensor, target_tensor):
    """
    Corta (crop) o tensor para que suas dimensões espaciais (altura e largura)
    coincidam com as do target_tensor.
    """
    _, _, h, w = target_tensor.size()
    _, _, H, W = tensor.size()
    delta_h = (H - h) // 2
    delta_w = (W - w) // 2
    return tensor[:, :, delta_h:delta_h + h, delta_w:delta_w + w]

class DoubleConv(nn.Module):
    """
    Bloco de duas convoluções 3x3 (válidas, sem padding) seguidas de ReLU.
    Cada bloco reduz as dimensões espaciais em 4 pixels (2 pixels por convolução).
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),  # sem padding
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),  # sem padding
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNetOriginal(nn.Module):
    """
    Implementação da U-Net original adaptada para entrada de 384x384 pixels.
    Saída produzida: 196x196 pixels.
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(UNetOriginal, self).__init__()
        # Caminho de contração (encoder)
        self.down1 = DoubleConv(in_channels, 64)    # saída: 380x380
        self.pool1 = nn.MaxPool2d(2)                # 190x190
        
        self.down2 = DoubleConv(64, 128)            # 186x186
        self.pool2 = nn.MaxPool2d(2)                # 93x93
        
        self.down3 = DoubleConv(128, 256)           # 89x89
        self.pool3 = nn.MaxPool2d(2)                # 44x44
        
        self.down4 = DoubleConv(256, 512)           # 40x40
        self.pool4 = nn.MaxPool2d(2)                # 20x20
        
        # Camada intermediária (bottleneck)
        self.bottleneck = DoubleConv(512, 1024)     # 16x16
        
        # Caminho de expansão (decoder)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # 16->32
        self.upconv4 = DoubleConv(1024, 512)          
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)   # 32->56
        self.upconv3 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)   # 56->104
        self.upconv2 = DoubleConv(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)    # 104->200
        self.upconv1 = DoubleConv(128, 64)
        
        # Camada final
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        d1 = self.down1(x)        # 380x380
        p1 = self.pool1(d1)       # 190x190
        
        d2 = self.down2(p1)       # 186x186
        p2 = self.pool2(d2)       # 93x93
        
        d3 = self.down3(p2)       # 89x89
        p3 = self.pool3(d3)       # 44x44
        
        d4 = self.down4(p3)       # 40x40
        p4 = self.pool4(d4)       # 20x20
        
        bn = self.bottleneck(p4)  # 16x16
        
        # Decoder
        u4 = self.up4(bn)         # 32x32
        d4_crop = crop_tensor(d4, u4)
        u4 = torch.cat([u4, d4_crop], dim=1)
        u4 = self.upconv4(u4)     # 28x28
        
        u3 = self.up3(u4)         # 56x56
        d3_crop = crop_tensor(d3, u3)
        u3 = torch.cat([u3, d3_crop], dim=1)
        u3 = self.upconv3(u3)     # 52x52
        
        u2 = self.up2(u3)         # 104x104
        d2_crop = crop_tensor(d2, u2)
        u2 = torch.cat([u2, d2_crop], dim=1)
        u2 = self.upconv2(u2)     # 100x100
        
        u1 = self.up1(u2)         # 200x200
        d1_crop = crop_tensor(d1, u1)
        u1 = torch.cat([u1, d1_crop], dim=1)
        u1 = self.upconv1(u1)     # 196x196
        
        out = self.final_conv(u1)
        return out