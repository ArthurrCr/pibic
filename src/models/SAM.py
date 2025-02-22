import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry

class CloudSAM(nn.Module):
    def __init__(self, model_type='vit_b', freeze_image_encoder=False, input_resolution=(384, 384)):
        super().__init__()
        
        self.input_resolution = input_resolution  # Ex: (384, 384)
        # Carrega o modelo base do SAM (neste exemplo, vit_b)
        self.sam = sam_model_registry.get(model_type)()
        
        # Ajusta os positional embeddings para a nova resolução (não flattenado)
        self.adjust_pos_embed()
        
        # Congela o image encoder se desejado
        if freeze_image_encoder:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False
        
        # Definição do classificador customizado para segmentação binária.
        # Aqui assumimos que o image encoder produzirá 256 canais.
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)  # Saída: logits para segmentação binária
        )
        
        # Calcula o patch size (usando o atributo ou o kernel_size da projeção)
        try:
            patch_size = self.sam.image_encoder.patch_embed.patch_size
            if isinstance(patch_size, tuple):
                patch_size = patch_size[0]
        except AttributeError:
            patch_size = self.sam.image_encoder.patch_embed.proj.kernel_size[0]
        
        # A resolução do mapa de características é dada por:
        feature_map_size = (self.input_resolution[0] // patch_size, 
                            self.input_resolution[1] // patch_size)
        # Fator de upscale necessário para que, após o upscale, a resolução seja a original.
        upscale_factor = self.input_resolution[0] // feature_map_size[0]  # Ex: 384/24 = 16
        
        # Bloco de upsampling para restaurar a resolução original
        self.upscale = nn.Sequential(
            nn.Upsample(scale_factor=upscale_factor, mode='bilinear', align_corners=True),
            nn.Conv2d(1, 1, 3, padding=1)
        )
        
    def adjust_pos_embed(self):
        """
        Ajusta os positional embeddings do image_encoder para que tenham formato (1, H, W, C),
        compatível com a saída do patch_embed.
        """
        # Recupera o patch size
        try:
            patch_size = self.sam.image_encoder.patch_embed.patch_size
            if isinstance(patch_size, tuple):
                patch_size = patch_size[0]
        except AttributeError:
            patch_size = self.sam.image_encoder.patch_embed.proj.kernel_size[0]
        
        # Novo grid de patches para a resolução de entrada.
        new_grid_size = (self.input_resolution[0] // patch_size, 
                         self.input_resolution[1] // patch_size)
        
        # Obtém o positional embedding original (forma: (1, N, dim))
        pos_embed = self.sam.image_encoder.pos_embed  
        dim = pos_embed.shape[-1]
        total_tokens = pos_embed.numel() // dim
        
        # Se houver token de classe (normalmente para 1024x1024 temos 4097 tokens)
        if total_tokens == 4097:
            cls_token = pos_embed[:, :1, :]
            pos_tokens = pos_embed[:, 1:, :]
            old_grid_dim = int((total_tokens - 1) ** 0.5)
        elif total_tokens == 4096:
            cls_token = None
            pos_tokens = pos_embed
            old_grid_dim = int(total_tokens ** 0.5)
        else:
            cls_token = None
            old_grid_dim = int(total_tokens ** 0.5)
            pos_tokens = pos_embed
        
        # Reorganiza os tokens para um grid: (1, old_grid_dim, old_grid_dim, dim)
        pos_tokens = pos_tokens.reshape(1, old_grid_dim, old_grid_dim, dim)
        # Permuta para (1, dim, old_grid_dim, old_grid_dim) para a interpolação
        pos_tokens = pos_tokens.permute(0, 3, 1, 2)
        # Interpola para o novo grid (ex.: de 64x64 para 24x24)
        pos_tokens = F.interpolate(pos_tokens, size=new_grid_size, mode='bilinear', align_corners=False)
        # Permuta de volta para (1, new_grid_size[0], new_grid_size[1], dim)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1)
        # Neste caso, descartamos o token de classe (caso exista), pois usaremos apenas os patches.
        new_pos_embed = pos_tokens  # shape: (1, new_grid_size[0], new_grid_size[1], dim)
        
        self.sam.image_encoder.pos_embed = nn.Parameter(new_pos_embed)
        
    def forward(self, x):
        # O image encoder já retorna o tensor no formato (B, C, H, W)
        image_embeddings = self.sam.image_encoder(x)
        # Não é necessário permutar; use o tensor diretamente
        logits = self.classifier(image_embeddings)
        out = self.upscale(logits)
        return out
        
    @torch.no_grad()
    def predict(self, x, threshold=0.5):
        self.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits) > threshold