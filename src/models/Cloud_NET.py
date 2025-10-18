import torch
import torch.nn as nn
import torch.nn.functional as F

# ====================================================
# Blocos auxiliares
# ====================================================
class BNReLU(nn.Module):
    """
    Equivale a: BatchNormalization + ReLU.
    """
    def __init__(self, num_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        return F.relu(self.bn(x), inplace=True)


class ContrArm(nn.Module):
    """
    Equivalente ao 'contr_arm' do código Keras:
      - 2 convs (3x3), cada uma com BN + ReLU
      - 'Skip' com conv(1x1) -> BNReLU -> concat -> soma final + ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1   = BNReLU(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2   = BNReLU(out_channels)

        mid_channels = out_channels // 2
        self.skip_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0)
        self.skip_bn   = BNReLU(mid_channels)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))

        skip = self.skip_bn(self.skip_conv(x))
        cat  = torch.cat([x, skip], dim=1)  # Concat canal
        out  = out + cat  # soma elementwise
        out  = F.relu(out, inplace=True)
        return out


class ImprvContrArm(nn.Module):
    """
    Equivalente a 'imprv_contr_arm':
      - 3 convs (3x3) c/ BN+ReLU
      - 2 caminhos 'skip': conv(1x1) do input, e conv(1x1) do penúltimo output
      - soma final + ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1   = BNReLU(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2   = BNReLU(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.bn3   = BNReLU(out_channels)

        # Caminhos skip
        mid_channels = out_channels // 2
        self.skip_in_conv  = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.skip_in_bn    = BNReLU(mid_channels)

        self.skip_mid_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.skip_mid_bn   = BNReLU(out_channels)

    def forward(self, x):
        out1 = self.bn1(self.conv1(x))
        out2 = self.bn2(self.conv2(out1))
        out3 = self.bn3(self.conv3(out2))

        skip_in  = self.skip_in_bn(self.skip_in_conv(x))
        skip_in  = torch.cat([x, skip_in], dim=1)

        skip_mid = self.skip_mid_bn(self.skip_mid_conv(out1))

        out = out3 + skip_in + skip_mid
        out = F.relu(out, inplace=True)
        return out


class Bridge(nn.Module):
    """
    Equivalente ao 'bridge':
      - 2 convs (3x3) c/ BN+ReLU, dropout entre elas
      - skip conv(1x1) do input -> soma final
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1   = BNReLU(out_channels)

        self.conv2   = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.dropout = nn.Dropout2d(0.15)
        self.bn2     = BNReLU(out_channels)

        mid_channels = out_channels // 2
        self.skip_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.skip_bn   = BNReLU(mid_channels)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.conv2(out)
        out = self.dropout(out)
        out = self.bn2(out)

        skip = self.skip_bn(self.skip_conv(x))
        skip = torch.cat([x, skip], dim=1)

        out = out + skip
        out = F.relu(out, inplace=True)
        return out


class ConvBlockExpPath(nn.Module):
    """
    'conv_block_exp_path': 2 convs seguidas c/ BN+ReLU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1   = BNReLU(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2   = BNReLU(out_channels)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))
        return out


class ConvBlockExpPath3(nn.Module):
    """
    'conv_block_exp_path3': 3 convs seguidas c/ BN+ReLU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1   = BNReLU(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2   = BNReLU(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn3   = BNReLU(out_channels)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))
        return out


# ====================================================
# Blocos de "improvement" (skip connections multi-escala)
# ====================================================
class ImproveFFBlock4(nn.Module):
    """
    Equivale a improve_ff_block4(c4, c3, c2, c1, c5).
    Cada tensor é repetido n vezes, depois sofre pooling
    e tudo é somado + pure_ff, no final ReLU.
    """
    def __init__(self):
        super().__init__()
        self.pool_2  = nn.MaxPool2d(kernel_size=2,  stride=2)
        self.pool_4  = nn.MaxPool2d(kernel_size=4,  stride=4)
        self.pool_8  = nn.MaxPool2d(kernel_size=8,  stride=8)
        self.pool_16 = nn.MaxPool2d(kernel_size=16, stride=16)

    def forward(self, in1, in2, in3, in4, pure_ff):
        # in1 (repetido 1x -> 2 total)
        x1 = torch.cat([in1, in1], dim=1)
        x1 = self.pool_2(x1)

        # in2 (repetido 3x -> 4 total)
        x2 = in2
        for _ in range(3):
            x2 = torch.cat([x2, in2], dim=1)
        x2 = self.pool_4(x2)

        # in3 (repetido 7x -> 8 total)
        x3 = in3
        for _ in range(7):
            x3 = torch.cat([x3, in3], dim=1)
        x3 = self.pool_8(x3)

        # in4 (repetido 15x -> 16 total)
        x4 = in4
        for _ in range(15):
            x4 = torch.cat([x4, in4], dim=1)
        x4 = self.pool_16(x4)

        # Todos têm shape compatível => soma
        out = x1 + x2 + x3 + x4 + pure_ff
        out = F.relu(out, inplace=True)
        return out


class ImproveFFBlock3(nn.Module):
    """
    Equivale a improve_ff_block3(c3, c2, c1, c4).
    """
    def __init__(self):
        super().__init__()
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_4 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pool_8 = nn.MaxPool2d(kernel_size=8, stride=8)

    def forward(self, in1, in2, in3, pure_ff):
        # in1 (1 vez -> concat => 2 total)
        x1 = torch.cat([in1, in1], dim=1)
        x1 = self.pool_2(x1)

        # in2 (3 vezes -> 4 total)
        x2 = in2
        for _ in range(3):
            x2 = torch.cat([x2, in2], dim=1)
        x2 = self.pool_4(x2)

        # in3 (7 vezes -> 8 total)
        x3 = in3
        for _ in range(7):
            x3 = torch.cat([x3, in3], dim=1)
        x3 = self.pool_8(x3)

        out = x1 + x2 + x3 + pure_ff
        out = F.relu(out, inplace=True)
        return out


class ImproveFFBlock2(nn.Module):
    """
    Equivale a improve_ff_block2(c2, c1, c3).
    """
    def __init__(self):
        super().__init__()
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_4 = nn.MaxPool2d(kernel_size=4, stride=4)

    def forward(self, in1, in2, pure_ff):
        # in1 (1 vez -> 2 total)
        x1 = torch.cat([in1, in1], dim=1)
        x1 = self.pool_2(x1)

        # in2 (3 vezes -> 4 total)
        x2 = in2
        for _ in range(3):
            x2 = torch.cat([x2, in2], dim=1)
        x2 = self.pool_4(x2)

        out = x1 + x2 + pure_ff
        out = F.relu(out, inplace=True)
        return out


class ImproveFFBlock1(nn.Module):
    """
    Equivale a improve_ff_block1(c1, c2).
    """
    def __init__(self):
        super().__init__()
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, in1, pure_ff):
        x1 = torch.cat([in1, in1], dim=1)
        x1 = self.pool_2(x1)

        out = x1 + pure_ff
        out = F.relu(out, inplace=True)
        return out


# ====================================================
# Modelo final
# ====================================================
class CloudNet(nn.Module):
    """
    UNet estilo Keras, adaptado para:
      - Entrada: 13 bandas, tamanho 512x512
      - Saída: 4 classes
    """
    def __init__(self, input_rows=512, input_cols=512,
                 num_of_channels=13, num_of_classes=4):
        super().__init__()
        self.input_rows = input_rows
        self.input_cols = input_cols

        # Parte inicial
        self.first_conv = nn.Conv2d(num_of_channels, 16, kernel_size=3, padding=1)

        # Down (contr_arm)
        self.contr_arm_1 = ContrArm(in_channels=16,  out_channels=32)
        self.pool1       = nn.MaxPool2d(2)

        self.contr_arm_2 = ContrArm(in_channels=32,  out_channels=64)
        self.pool2       = nn.MaxPool2d(2)

        self.contr_arm_3 = ContrArm(in_channels=64,  out_channels=128)
        self.pool3       = nn.MaxPool2d(2)

        self.contr_arm_4 = ContrArm(in_channels=128, out_channels=256)
        self.pool4       = nn.MaxPool2d(2)

        # Down (imprv_contr_arm) + pool
        self.imprv_contr_arm = ImprvContrArm(in_channels=256, out_channels=512)
        self.pool5           = nn.MaxPool2d(2)

        # "Miolo"
        self.bridge = Bridge(in_channels=512, out_channels=1024)

        # Up7
        self.convT7 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.improve_ff_block4    = ImproveFFBlock4()
        # Concat => (512 + 512) = 1024 canais
        self.conv_block_exp_path3_7 = ConvBlockExpPath3(in_channels=1024, out_channels=512)

        # Up8
        self.convT8 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.improve_ff_block3    = ImproveFFBlock3()
        # Concat => (256 + 256) = 512 canais
        self.conv_block_exp_path_8 = ConvBlockExpPath(in_channels=512, out_channels=256)

        # Up9
        self.convT9 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.improve_ff_block2    = ImproveFFBlock2()
        # Concat => (128 + 128) = 256 canais
        self.conv_block_exp_path_9 = ConvBlockExpPath(in_channels=256, out_channels=128)

        # Up10
        self.convT10 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.improve_ff_block1    = ImproveFFBlock1()
        # Concat => (64 + 64) = 128 canais
        self.conv_block_exp_path_10 = ConvBlockExpPath(in_channels=128, out_channels=64)

        # Up11
        self.convT11 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        # Concat => (32 + 32) = 64 canais
        self.conv_block_exp_path_11 = ConvBlockExpPath(in_channels=64, out_channels=32)

        # Saída final (4 classes)
        self.final_conv = nn.Conv2d(32, num_of_classes, kernel_size=1)

    def forward(self, x):
        # Down
        x1 = F.relu(self.first_conv(x))       # -> (B,16,512,512)

        c1 = self.contr_arm_1(x1)             # -> (B,32,512,512)
        p1 = self.pool1(c1)                   # -> (B,32,256,256)

        c2 = self.contr_arm_2(p1)             # -> (B,64,256,256)
        p2 = self.pool2(c2)                   # -> (B,64,128,128)

        c3 = self.contr_arm_3(p2)             # -> (B,128,128,128)
        p3 = self.pool3(c3)                   # -> (B,128,64,64)

        c4 = self.contr_arm_4(p3)             # -> (B,256,64,64)
        p4 = self.pool4(c4)                   # -> (B,256,32,32)

        c5 = self.imprv_contr_arm(p4)         # -> (B,512,32,32)
        p5 = self.pool5(c5)                   # -> (B,512,16,16)

        br = self.bridge(p5)                  # -> (B,1024,16,16)

        # Up7
        up7 = self.convT7(br)                 # -> (B,512,32,32)
        prevup7 = self.improve_ff_block4(c4, c3, c2, c1, c5)  # -> (B,512,32,32)
        cat7 = torch.cat([up7, prevup7], dim=1)  # -> (B,1024,32,32)
        out7 = self.conv_block_exp_path3_7(cat7) # -> (B,512,32,32)
        out7 = F.relu(out7 + c5 + up7, inplace=True)

        # Up8
        up8 = self.convT8(out7)               # -> (B,256,64,64)
        prevup8 = self.improve_ff_block3(c3, c2, c1, c4)  # -> (B,256,64,64)
        cat8 = torch.cat([up8, prevup8], dim=1)  # -> (B,512,64,64)
        out8 = self.conv_block_exp_path_8(cat8)   # -> (B,256,64,64)
        out8 = F.relu(out8 + c4 + up8, inplace=True)

        # Up9
        up9 = self.convT9(out8)               # -> (B,128,128,128)
        prevup9 = self.improve_ff_block2(c2, c1, c3)  # -> (B,128,128,128)
        cat9 = torch.cat([up9, prevup9], dim=1)      # -> (B,256,128,128)
        out9 = self.conv_block_exp_path_9(cat9)      # -> (B,128,128,128)
        out9 = F.relu(out9 + c3 + up9, inplace=True)

        # Up10
        up10 = self.convT10(out9)             # -> (B,64,256,256)
        prevup10 = self.improve_ff_block1(c1, c2)  # -> (B,64,256,256)
        cat10 = torch.cat([up10, prevup10], dim=1)  # -> (B,128,256,256)
        out10 = self.conv_block_exp_path_10(cat10)  # -> (B,64,256,256)
        out10 = F.relu(out10 + c2 + up10, inplace=True)

        # Up11
        up11 = self.convT11(out10)            # -> (B,32,512,512)
        cat11 = torch.cat([up11, c1], dim=1)  # -> (B,64,512,512)
        out11 = self.conv_block_exp_path_11(cat11)   # -> (B,32,512,512)
        out11 = F.relu(out11 + c1 + up11, inplace=True)

        # Final: gera 4 canais (logits), sem ativação
        logits = self.final_conv(out11)
        return logits
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Usa softmax + argmax para retornar as classes preditas (exclusivas).
        
        Ex.: Se num_of_classes=4 => cada pixel ∈ {0,1,2,3}.
        """
        self.eval()
        logits = self.forward(x)               # (B, 4, H, W)
        probs = torch.softmax(logits, dim=1)   # (B, 4, H, W)
        labels = probs.argmax(dim=1)           # (B, H, W)
        return labels