import torch
from torch.utils.data import Dataset, DataLoader
import rasterio as rio
import tacoreader
import numpy as np
import torch
import numpy as np
import rasterio as rio
from torch.utils.data import Dataset

class CloudSEN12Dataset(Dataset):
    def __init__(self, tdf, transform=None, cache_enabled=False, normalize=False, use_rgb=False):
        """
        tdf: TortillaDataFrame já filtrado.
        transform: pipeline Albumentations ou None.
        cache_enabled: se True, armazena img_data e mask_data em self.cache (consome mais RAM).
        normalize: se True, normaliza as imagens dividindo por 10_000 (ou o valor desejado).
        use_rgb: se True, retorna apenas as bandas RGB (R:3, G:2, B:1).
        """
        self.tdf = tdf
        self.transform = transform
        self.cache_enabled = cache_enabled
        self.cache = {}
        self.normalize = normalize
        self.use_rgb = use_rgb  # Flag para selecionar apenas as bandas RGB

    def __len__(self):
        return len(self.tdf)

    def __getitem__(self, idx):
        # Lê paths da imagem e máscara
        sample = self.tdf.read(idx)
        img_path = sample.read(0)
        mask_path = sample.read(1)

        # Se cache_enabled=True, carrega do cache (se disponível)
        if self.cache_enabled:
            if (img_path, mask_path) not in self.cache:
                img_data, mask_data = self._read_files(img_path, mask_path)
                self.cache[(img_path, mask_path)] = (img_data, mask_data)
            else:
                img_data, mask_data = self.cache[(img_path, mask_path)]
        else:
            img_data, mask_data = self._read_files(img_path, mask_path)

        # Transpõe para (H, W, C) - forma esperada pelo Albumentations
        img_data = np.transpose(img_data, (1, 2, 0))  # (C,H,W)->(H,W,C)
        mask_data = np.transpose(mask_data, (1, 2, 0))  # (C,H,W)->(H,W,C)

        # Se a flag use_rgb for True, seleciona apenas as bandas RGB (R:3, G:2, B:1)
        if self.use_rgb:
            img_data = img_data[:, :, [3, 2, 1]]  # Seleciona as bandas RGB (R:3, G:2, B:1)

        # Se a flag normalize for True, normaliza a imagem
        if self.normalize:
            img_data = img_data / 10_000  # ou outro valor de normalização, conforme necessário

        # Aplica transformações, se houver
        if self.transform:
            augmented = self.transform(image=img_data, mask=mask_data)
            img_tensor = augmented["image"].float()
            mask_tensor = augmented["mask"].long()
        else:
            img_tensor = torch.from_numpy(np.transpose(img_data, (2, 0, 1))).float()
            mask_tensor = torch.from_numpy(np.transpose(mask_data, (2, 0, 1))).long()

        return img_tensor, mask_tensor

    def _read_files(self, img_path, mask_path):
        with rio.open(img_path) as src:
            img_data = src.read()  # (C, H, W)
        with rio.open(mask_path) as msk:
            mask_data = msk.read() # (C, H, W) ou (1, H, W)
        return img_data, mask_data

    def _read_files(self, img_path, mask_path):
        with rio.open(img_path) as src:
            img_data = src.read()  # (C, H, W)
        with rio.open(mask_path) as msk:
            mask_data = msk.read() # (C, H, W) ou (1, H, W)
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
    cache_enabled=False,
    normalize=False,
    use_rgb=False  
):
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

    # Passando a flag normalize e use_rgb para o dataset
    train_dataset = CloudSEN12Dataset(train_tdf, train_transform, cache_enabled, normalize, use_rgb) if len(train_tdf) else None
    val_dataset   = CloudSEN12Dataset(val_tdf, val_transform, cache_enabled, normalize, use_rgb) if len(val_tdf) else None
    test_dataset  = CloudSEN12Dataset(test_tdf, test_transform, cache_enabled, normalize, use_rgb) if len(test_tdf) else None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers) if train_dataset else None
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) if val_dataset else None
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) if test_dataset else None

    return train_loader, val_loader, test_loader

class MaskBandDataset(Dataset):
    def __init__(self, base_dataset, band_idx, fill_value=0.0):
        """
        base_dataset: instância do seu CloudSEN12Dataset.
        band_idx: índice da banda a ser mascarada.
        fill_value: valor com o qual a banda será preenchida (default: 0.0).
        """
        self.base_dataset = base_dataset
        self.band_idx = band_idx
        self.fill_value = fill_value

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Obtenha a amostra original
        img_tensor, mask_tensor = self.base_dataset[idx]
        # Crie uma cópia para não modificar o original
        img_tensor = img_tensor.clone()
        # Zera a banda específica
        img_tensor[self.band_idx, :, :] = self.fill_value
        return img_tensor, mask_tensor
