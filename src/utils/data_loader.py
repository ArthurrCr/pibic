import torch
from torch.utils.data import Dataset, DataLoader
import rasterio as rio
import tacoreader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class CloudSEN12Dataset(Dataset):
    def __init__(self, tdf, transform=None, cache_enabled=False):
        """
        tdf: TortillaDataFrame já filtrado.
        transform: função (opcional) de data augmentation. Ex.: pipeline do Albumentations que espera as chaves "image" e "mask".
        cache_enabled: se True, armazena img_data e mask_data em self.cache,
                       podendo aumentar o uso de RAM em datasets grandes.
        """
        self.tdf = tdf
        self.transform = transform
        self.cache_enabled = cache_enabled
        self.cache = {}

    def __len__(self):
        return len(self.tdf)

    def __getitem__(self, idx):
        sample = self.tdf.read(idx)
        # Em CloudSEN12: sample.read(0) retorna o caminho da imagem e sample.read(1) o caminho da máscara
        img_path = sample.read(0)
        mask_path = sample.read(1)

        if self.cache_enabled:
            if (img_path, mask_path) not in self.cache:
                img_data, mask_data = self._read_files(img_path, mask_path)
                self.cache[(img_path, mask_path)] = (img_data, mask_data)
            else:
                img_data, mask_data = self.cache[(img_path, mask_path)]
        else:
            img_data, mask_data = self._read_files(img_path, mask_path)

        # As imagens e máscaras são lidas no formato (C, H, W). 
        # Para o Albumentations precisamos de (H, W, C)
        img_data = np.transpose(img_data, (1, 2, 0))
        mask_data = np.transpose(mask_data, (1, 2, 0))
        
        if self.transform:
            # O pipeline de augmentation espera receber um dicionário com as chaves "image" e "mask"
            augmented = self.transform(image=img_data, mask=mask_data)
            img_tensor = augmented["image"]
            mask_tensor = augmented["mask"]
        else:
            # Caso não haja transform, converte para tensor mantendo a ordem original (C, H, W)
            # Como já transpus os dados para (H, W, C), transpor novamente para (C, H, W)
            img_tensor = torch.from_numpy(np.transpose(img_data, (2, 0, 1))).float()
            mask_tensor = torch.from_numpy(np.transpose(mask_data, (2, 0, 1))).float()

        return img_tensor, mask_tensor

    def _read_files(self, img_path, mask_path):
        with rio.open(img_path) as src:
            img_data = src.read()  # (C, H, W)
        with rio.open(mask_path) as msk:
            mask_data = msk.read()  # (C, H, W) ou (1, H, W)
        return img_data, mask_data

def create_dataloaders(
    parts,
    real_proj_shape=509,
    label_type="high",
    batch_size=8,
    num_workers=2,
    train_transform=None,
    val_transform=None,
    test_transform=None,
    cache_enabled=False
):
    """
    Cria DataLoaders para os splits de train, validation e test.
    Permite definir transformações distintas para cada split.
    """
    ds = tacoreader.load(parts)

    train_tdf = ds[
        (ds["real_proj_shape"] == real_proj_shape)
        & (ds["label_type"] == label_type)
        & (ds["tortilla:data_split"] == "train")
    ]
    val_tdf = ds[
        (ds["real_proj_shape"] == real_proj_shape)
        & (ds["label_type"] == label_type)
        & (ds["tortilla:data_split"] == "validation")
    ]
    test_tdf = ds[
        (ds["real_proj_shape"] == real_proj_shape)
        & (ds["label_type"] == label_type)
        & (ds["tortilla:data_split"] == "test")
    ]

    print(f"Train samples: {len(train_tdf)}")
    print(f"Val samples:   {len(val_tdf)}")
    print(f"Test samples:  {len(test_tdf)}")

    train_dataset = CloudSEN12Dataset(train_tdf, train_transform, cache_enabled) if len(train_tdf) else None
    val_dataset   = CloudSEN12Dataset(val_tdf, val_transform, cache_enabled) if len(val_tdf) else None
    test_dataset  = CloudSEN12Dataset(test_tdf, test_transform, cache_enabled) if len(test_tdf) else None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers) if train_dataset else None
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) if test_dataset else None

    return train_loader, val_loader, test_loader

def visualize_multiclass_batch(
    imgs,
    masks,
    class_labels={0: "Clear", 1: "Thick Cloud", 2: "Thin Cloud", 3: "Cloud Shadow"},
    rgb_bands=(3, 2, 1),
    max_items=4
):
    """
    Exibe até max_items itens do batch (imgs, masks), assumindo 4 classes.
    """
    cmap = mcolors.ListedColormap(["black", "red", "blue", "green"])
    bounds = [0, 1, 2, 3, 4]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    batch_size = imgs.shape[0]
    n_items = min(batch_size, max_items)

    fig, axes = plt.subplots(2, n_items, figsize=(5 * n_items, 8))
    if n_items == 1:
        axes = np.array([axes]).T

    for i in range(n_items):
        img_np = imgs[i].cpu().numpy()
        mask_np = masks[i].cpu().numpy()

        # Se a máscara tiver 4 canais, assume-se que são probabilidades e escolhe o canal com maior valor.
        if mask_np.shape[0] == 4:
            mask_2d = np.argmax(mask_np, axis=0)
        else:
            mask_2d = mask_np[0]

        # Seleciona as bandas RGB (a ordem é definida em rgb_bands)
        rgb = img_np[list(rgb_bands), :, :]
        rgb = np.transpose(rgb, (1, 2, 0))
        rgb = np.clip(rgb / 3000.0, 0, 1)

        axes[0, i].imshow(rgb)
        axes[0, i].set_title(f"RGB item {i}")
        axes[0, i].axis("off")

        im = axes[1, i].imshow(mask_2d, cmap=cmap, norm=norm)
        axes[1, i].set_title(f"Mask item {i}")
        axes[1, i].axis("off")

    plt.tight_layout()
    cbar = plt.colorbar(im, ax=axes[1, -1], ticks=[0.5, 1.5, 2.5, 3.5])
    cbar.ax.set_yticklabels([class_labels[0], class_labels[1], class_labels[2], class_labels[3]])
    plt.show()
