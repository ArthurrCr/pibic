import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------
# 1. Bloco de Convolução com BatchNorm e SiLU
# ------------------------------------------
class ConvBnSilu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2  # padding "same"
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# ------------------------------
# 2. Focus Layer
# ------------------------------
class Focus(nn.Module):
    """
    Divide a imagem de entrada em 4 patches (pegando pixels alternados)
    e os concatena no canal, aumentando a densidade de informação.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        patch_tl = x[..., ::2, ::2]
        patch_tr = x[..., ::2, 1::2]
        patch_bl = x[..., 1::2, ::2]
        patch_br = x[..., 1::2, 1::2]
        x = torch.cat((patch_tl, patch_tr, patch_bl, patch_br), dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

# ------------------------------
# 3. Módulo Bottleneck simples
# ------------------------------
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        self.conv1 = ConvBnSilu(in_channels, out_channels, kernel_size=1)
        self.conv2 = ConvBnSilu(out_channels, out_channels, kernel_size=3)
        self.shortcut = shortcut and in_channels == out_channels
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            y = y + x
        return y

# ------------------------------
# 4. Módulo C2f (variante dos blocos CSP)
# ------------------------------
class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True):
        """
        A ideia é dividir o fluxo em duas ramificações:
         - Uma ramificação simples (conv1)
         - Outra que passa por n blocos (conv2 + n x Bottleneck)
        Em seguida, as duas ramificações são concatenadas e processadas.
        """
        super().__init__()
        hidden_channels = out_channels // 2
        self.conv1 = ConvBnSilu(in_channels, hidden_channels, kernel_size=1)
        self.conv2 = ConvBnSilu(in_channels, hidden_channels, kernel_size=1)
        self.blocks = nn.Sequential(*[
            Bottleneck(hidden_channels, hidden_channels, shortcut=shortcut) for _ in range(n)
        ])
        self.conv3 = ConvBnSilu(2 * hidden_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y2 = self.blocks(y2)
        return self.conv3(torch.cat([y1, y2], dim=1))

# ------------------------------
# 5. Módulo SPPF (Spatial Pyramid Pooling – Fast)
# ------------------------------
class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, pool_kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = ConvBnSilu(in_channels, hidden_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=1, padding=pool_kernel_size//2)
        self.conv2 = ConvBnSilu(hidden_channels * 4, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))

# ------------------------------
# 6. Neck: PAN (Path Aggregation Network)
# ------------------------------
class PAN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        """
        Recebe uma lista de feature maps de diferentes escalas (por exemplo, de 3 níveis)
        e realiza a agregação top-down (com upsampling e concatenação).
        """
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels_list
        ])
        # Dois níveis de FPN (suavização) para 3 escalas
        self.fpn_convs = nn.ModuleList([
            ConvBnSilu(out_channels * 2, out_channels, kernel_size=3),
            ConvBnSilu(out_channels * 2, out_channels, kernel_size=3)
        ])
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
    
    def forward(self, features):
        # Supondo features = [feat_small, feat_medium, feat_large] (da maior para a menor resolução)
        lateral_feats = [l_conv(f) for l_conv, f in zip(self.lateral_convs, features)]
        # Top-down: p_large -> p_medium
        p_large = lateral_feats[2]
        p_medium = self.fpn_convs[1](torch.cat([lateral_feats[1], self.upsample(p_large)], dim=1))
        p_small = self.fpn_convs[0](torch.cat([lateral_feats[0], self.upsample(p_medium)], dim=1))
        return [p_small, p_medium, p_large]

# ------------------------------
# 7. Cabeça de Detecção Decoupled
# ------------------------------
class YOLOv8Head(nn.Module):
    def __init__(self, in_channels, num_classes, num_outputs=5):
        """
        Para cada escala, cria dois ramos:
         - Um para classificação (num_classes saídas)
         - Outro para regressão (por exemplo, 4 coordenadas + 1 objeto)
        """
        super().__init__()
        self.cls_conv = nn.Sequential(
            ConvBnSilu(in_channels, in_channels, kernel_size=3),
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )
        self.reg_conv = nn.Sequential(
            ConvBnSilu(in_channels, in_channels, kernel_size=3),
            nn.Conv2d(in_channels, num_outputs, kernel_size=1)
        )
    
    def forward(self, x):
        cls_out = self.cls_conv(x)
        reg_out = self.reg_conv(x)
        return cls_out, reg_out

# ------------------------------
# 8. Modelo YOLOv8 (Aproximação Simplificada)
# ------------------------------
class YOLOv8Net(nn.Module):
    def __init__(self, num_classes=80):
        """
        Esta arquitetura segue os principais blocos do YOLOv8:
          - Backbone: Focus, seguido por blocos C2f e SPPF
          - Neck: PAN para agregação de múltiplas escalas
          - Head: Cabeça decoupled para cada escala
        """
        super().__init__()
        # Backbone
        self.focus = Focus(3, 64, kernel_size=3)
        self.c2f1 = C2f(64, 128, n=1)
        self.c2f2 = C2f(128, 256, n=3)
        self.c2f3 = C2f(256, 512, n=3)
        self.c2f4 = C2f(512, 1024, n=1)
        self.sppf = SPPF(1024, 1024, pool_kernel_size=5)
        # Selecionamos 3 escalas para o neck:
        # - feat_small: saída de c2f2 (alta resolução, ~256 canais)
        # - feat_medium: saída de c2f3 (~512 canais)
        # - feat_large: saída de sppf (após c2f4 e SPPF, ~1024 canais)
        self.pan = PAN(in_channels_list=[256, 512, 1024], out_channels=256)
        # Cabeças para cada escala
        self.head_small = YOLOv8Head(256, num_classes, num_outputs=5)
        self.head_medium = YOLOv8Head(256, num_classes, num_outputs=5)
        self.head_large = YOLOv8Head(256, num_classes, num_outputs=5)
    
    def forward(self, x):
        x = self.focus(x)            # [B, 64, H/2, W/2]
        x = self.c2f1(x)             # [B, 128, H/2, W/2]
        feat_small = self.c2f2(x)      # [B, 256, H/2, W/2] (alta resolução)
        feat_medium = self.c2f3(feat_small)  # [B, 512, H/?, W/?]
        feat_large = self.c2f4(feat_medium)  # [B, 1024, H/?, W/?]
        feat_large = self.sppf(feat_large)   # [B, 1024, H/?, W/?]
        
        # Neck: PAN para fusão de escalas
        pan_features = self.pan([feat_small, feat_medium, feat_large])
        # pan_features: lista com 3 escalas, todas com 256 canais
        
        # Aplicação da cabeça em cada escala
        cls_small, reg_small = self.head_small(pan_features[0])
        cls_medium, reg_medium = self.head_medium(pan_features[1])
        cls_large, reg_large = self.head_large(pan_features[2])
        
        return {
            "cls_small": cls_small,
            "reg_small": reg_small,
            "cls_medium": cls_medium,
            "reg_medium": reg_medium,
            "cls_large": cls_large,
            "reg_large": reg_large,
        }