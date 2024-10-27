import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class CloudDataset(Dataset):
    def __init__(self, images_dir, masks_dir, file_list, image_transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.file_list = file_list
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_file = self.file_list[idx]
        img_path = os.path.join(self.images_dir, img_file)
        mask_path = os.path.join(self.masks_dir, img_file)

        # Carregar imagem e máscara
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # Aplicar transformações
        if self.image_transform:
            image = self.image_transform(image)
        else:
            image = transforms.ToTensor()(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = transforms.ToTensor()(mask)

        # Converter a máscara para rótulos binários
        mask = (mask > 0).float()

        return image, mask
    
class CloudTestDataset(Dataset):
    def __init__(self, images_dir, file_list, image_transform=None):
        self.images_dir = images_dir
        self.file_list = file_list
        self.image_transform = image_transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_file = self.file_list[idx]
        img_path = os.path.join(self.images_dir, img_file)

        # Carregar imagem
        image = Image.open(img_path).convert('RGB')

        # Aplicar transformações
        if self.image_transform:
            image = self.image_transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, img_file  # Retornamos o nome do arquivo para referência


class MultiFolderDataset(Dataset):
    def __init__(self, images_dirs, masks_dirs, file_list, image_transform=None, mask_transform=None):
        self.images_dirs = images_dirs
        self.masks_dirs = masks_dirs
        self.file_list = file_list
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        # Mapear cada arquivo ao seu diretório correspondente
        self.file_to_dir = {}
        for img_file in file_list:
            if os.path.exists(os.path.join(images_dirs[0], img_file)):
                self.file_to_dir[img_file] = (images_dirs[0], masks_dirs[0])
            elif os.path.exists(os.path.join(images_dirs[1], img_file)):
                self.file_to_dir[img_file] = (images_dirs[1], masks_dirs[1])
            else:
                raise FileNotFoundError(f"Arquivo {img_file} não encontrado nos diretórios fornecidos.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_file = self.file_list[idx]
        images_dir, masks_dir = self.file_to_dir[img_file]

        img_path = os.path.join(images_dir, img_file)
        mask_path = os.path.join(masks_dir, img_file)

        # Carregar imagem e máscara
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # Aplicar transformações
        if self.image_transform:
            image = self.image_transform(image)
        else:
            image = transforms.ToTensor()(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = transforms.ToTensor()(mask)

        # Converter a máscara para rótulos binários
        mask = (mask > 0).float()

        return image, mask

