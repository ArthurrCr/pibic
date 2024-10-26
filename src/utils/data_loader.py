import torch
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class CloudDataset(Dataset):
    def __init__(self, images_dir, masks_dir, file_list, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_file = self.file_list[idx]
        img_path = os.path.join(self.images_dir, img_file)
        mask_path = os.path.join(self.masks_dir, img_file)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # Aplicar transformações adicionais, se necessário
        if self.transform:
            image = self.transform(image)

        # Converter máscara para tensor
        mask = torch.from_numpy(np.array(mask) // 255).long()

        return image, mask
