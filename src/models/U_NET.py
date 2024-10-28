import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc_conv1 = self.double_conv(in_channels, 64)
        self.enc_conv2 = self.double_conv(64, 128)
        self.enc_conv3 = self.double_conv(128, 256)
        self.enc_conv4 = self.double_conv(256, 512)
        self.enc_conv5 = self.double_conv(512, 1024)
        
        # Decoder
        self.up_trans1 = self.up_trans(1024, 512)
        self.dec_conv1 = self.double_conv(1024, 512)
        
        self.up_trans2 = self.up_trans(512, 256)
        self.dec_conv2 = self.double_conv(512, 256)
        
        self.up_trans3 = self.up_trans(256, 128)
        self.dec_conv3 = self.double_conv(256, 128)
        
        self.up_trans4 = self.up_trans(128, 64)
        self.dec_conv4 = self.double_conv(128, 64)
        
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.enc_conv1(x)
        x2 = self.enc_conv2(self.max_pool2x2(x1))
        x3 = self.enc_conv3(self.max_pool2x2(x2))
        x4 = self.enc_conv4(self.max_pool2x2(x3))
        x5 = self.enc_conv5(self.max_pool2x2(x4))
        
        # Decoder
        d1 = self.up_trans1(x5)
        d1 = self.crop_and_concat(d1, x4)
        d1 = self.dec_conv1(d1)
        
        d2 = self.up_trans2(d1)
        d2 = self.crop_and_concat(d2, x3)
        d2 = self.dec_conv2(d2)
        
        d3 = self.up_trans3(d2)
        d3 = self.crop_and_concat(d3, x2)
        d3 = self.dec_conv3(d3)
        
        d4 = self.up_trans4(d3)
        d4 = self.crop_and_concat(d4, x1)
        d4 = self.dec_conv4(d4)
        
        out = self.out(d4)
        return out
    
    def max_pool2x2(self, x):
        return F.max_pool2d(x, kernel_size=2, stride=2)
    
    def up_trans(self, in_channels, out_channels):
        return nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
    
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # Opcional: Batch Normalization
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # Opcional: Batch Normalization
            nn.ReLU(inplace=True),
        )
    
    def crop_and_concat(self, upsampled, bypass):
        # Ajustar o tamanho se necessário (devido a diferenças de padding)
        if upsampled.size()[2:] != bypass.size()[2:]:
            diffY = bypass.size()[2] - upsampled.size()[2]
            diffX = bypass.size()[3] - upsampled.size()[3]
            upsampled = F.pad(upsampled, [diffX // 2, diffX - diffX // 2,
                                          diffY // 2, diffY - diffY // 2])
        return torch.cat((upsampled, bypass), 1)

def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice = dice_loss(pred, target)
        return bce_loss + dice
