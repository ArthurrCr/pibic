import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt


class CloudDataset(Dataset):
    def __init__(self, images_dir, masks_dir, file_list, transform=None):
        """
        file_list (list): lista dos nomes dos arquivos que serão usados (treino ou validação).
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_files = file_list
        
        if len(self.image_files) == 0:
            raise ValueError(f"Nenhuma imagem encontrada no diretório '{images_dir}'.")

        # Verificação de existência de imagem e máscara correspondente
        for img_file in self.image_files:
            img_path = os.path.join(self.images_dir, img_file)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Imagem '{img_file}' não encontrada em '{self.images_dir}'.")
            mask_path = os.path.join(self.masks_dir, img_file)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Máscara correspondente para '{img_file}' não encontrada em '{self.masks_dir}'.")

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_file)
        mask_path = os.path.join(self.masks_dir, img_file)

        # Carrega imagem e máscara
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))
        
        # Ajuste o threshold conforme necessário (ex: 127 para máscaras 0-255)
        mask = (mask > 127).astype(np.uint8)  # Binarização

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
            # Converte para tensor se for numpy array
            if isinstance(mask, np.ndarray):
                mask = torch.tensor(mask, dtype=torch.float)
            else:
                mask = mask.float()
            mask = mask.unsqueeze(0)  # Adiciona dimensão do canal
        else:
            # Converte imagem para tensor [C, H, W] e normaliza
            image = torch.from_numpy(image.transpose(2,0,1)).float() / 255.0
            # Converte máscara para tensor [1, H, W]
            mask = torch.from_numpy(mask).float().unsqueeze(0)

        return image, mask
    

def visualize_batch(images, masks, ncols=2, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    batch_size = images.shape[0]
    nrows = (batch_size + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols*2, figsize=(8*ncols, 8*nrows))
    axes = np.array(axes)
    if axes.ndim == 1:
        axes = np.expand_dims(axes, axis=0)
    
    idx = 0
    for row in range(nrows):
        for col in range(ncols):
            if idx >= batch_size:
                axes[row, 2*col].axis('off')
                axes[row, 2*col+1].axis('off')
                continue
            
            img = images[idx].clone()
            msk = masks[idx].clone()
            
            # Desnormalização
            for t, m, s in zip(img, mean, std):
                t.mul_(s).add_(m)
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = np.clip(img_np, 0, 1)
            
            msk_np = msk.squeeze(0).cpu().numpy()
            
            axes[row, 2*col].imshow(img_np)
            axes[row, 2*col].axis('off')
            axes[row, 2*col+1].imshow(msk_np, cmap='gray')
            axes[row, 2*col+1].axis('off')
            
            idx += 1
    
    plt.tight_layout()
    plt.show()
    
class CloudTestDataset(Dataset):
    def __init__(self, images_dir, file_list, transform=None):
        self.images_dir = images_dir
        self.file_list = file_list
        self.transform = transform
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        image_name = self.file_list[idx]
        image_path = os.path.join(self.images_dir, image_name)
        
        image = np.array(Image.open(image_path).convert("RGB"))
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = torch.from_numpy(image.transpose(2,0,1)).float() / 255.0
        
        return image