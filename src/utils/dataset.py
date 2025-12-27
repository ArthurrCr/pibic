"""Dataset classes and DataLoader utilities for CloudSEN12."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import rasterio as rio
import tacoreader
import torch
from torch.utils.data import DataLoader, Dataset


class CloudSEN12Dataset(Dataset):
    """
    PyTorch Dataset for CloudSEN12 satellite imagery.

    Loads Sentinel-2 images and corresponding cloud masks from a TortillaDataFrame.

    Attributes:
        tdf: Filtered TortillaDataFrame containing image paths.
        transform: Optional Albumentations transform pipeline.
        cache_enabled: If True, caches loaded data in memory.
        normalize: If True, normalizes images by dividing by 10,000.
        use_rgb: If True, returns only RGB bands (B04, B03, B02).
    """

    def __init__(
        self,
        tdf: Any,
        transform: Optional[Any] = None,
        cache_enabled: bool = False,
        normalize: bool = False,
        use_rgb: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            tdf: TortillaDataFrame already filtered for desired samples.
            transform: Albumentations pipeline or None.
            cache_enabled: If True, stores data in memory (higher RAM usage).
            normalize: If True, normalizes images by dividing by 10,000.
            use_rgb: If True, returns only RGB bands (indices 3, 2, 1).
        """
        self.tdf = tdf
        self.transform = transform
        self.cache_enabled = cache_enabled
        self.cache: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
        self.normalize = normalize
        self.use_rgb = use_rgb

    def __len__(self) -> int:
        return len(self.tdf)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.tdf.read(idx)
        img_path = sample.read(0)
        mask_path = sample.read(1)

        if self.cache_enabled:
            cache_key = (img_path, mask_path)
            if cache_key not in self.cache:
                img_data, mask_data = self._read_files(img_path, mask_path)
                self.cache[cache_key] = (img_data, mask_data)
            else:
                img_data, mask_data = self.cache[cache_key]
        else:
            img_data, mask_data = self._read_files(img_path, mask_path)

        # Transpose to (H, W, C) for Albumentations
        img_data = np.transpose(img_data, (1, 2, 0))
        mask_data = np.transpose(mask_data, (1, 2, 0))

        if self.use_rgb:
            img_data = img_data[:, :, [3, 2, 1]]

        if self.normalize:
            img_data = img_data / 10_000

        if self.transform:
            augmented = self.transform(image=img_data, mask=mask_data)
            img_tensor = augmented["image"].float()
            mask_tensor = augmented["mask"].long()
        else:
            img_tensor = torch.from_numpy(np.transpose(img_data, (2, 0, 1))).float()
            mask_tensor = torch.from_numpy(np.transpose(mask_data, (2, 0, 1))).long()

        return img_tensor, mask_tensor

    def _read_files(
        self, img_path: str, mask_path: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read image and mask files using rasterio."""
        with rio.open(img_path) as src:
            img_data = src.read()
        with rio.open(mask_path) as msk:
            mask_data = msk.read()
        return img_data, mask_data


class MaskBandDataset(Dataset):
    """
    Dataset wrapper that masks a specific band with a fill value.

    Useful for ablation studies to evaluate band importance.

    Attributes:
        base_dataset: Original CloudSEN12Dataset instance.
        band_idx: Index of the band to mask.
        fill_value: Value to fill the masked band with.
    """

    def __init__(
        self,
        base_dataset: CloudSEN12Dataset,
        band_idx: int,
        fill_value: float = 0.0,
    ):
        """
        Initialize the mask band dataset.

        Args:
            base_dataset: Original CloudSEN12Dataset instance.
            band_idx: Index of the band to be masked.
            fill_value: Value to fill the masked band (default: 0.0).
        """
        self.base_dataset = base_dataset
        self.band_idx = band_idx
        self.fill_value = fill_value

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_tensor, mask_tensor = self.base_dataset[idx]
        img_tensor = img_tensor.clone()
        img_tensor[self.band_idx, :, :] = self.fill_value
        return img_tensor, mask_tensor


def create_dataloaders(
    parts: Any,
    real_proj_shape: int = 509,
    label_type: str = "high",
    batch_size: int = 8,
    num_workers: int = 2,
    train_transform: Optional[Any] = None,
    val_transform: Optional[Any] = None,
    test_transform: Optional[Any] = None,
    cache_enabled: bool = False,
    normalize: bool = False,
    use_rgb: bool = False,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Create train, validation, and test DataLoaders from dataset parts.

    Args:
        parts: Dataset parts to load via tacoreader.
        real_proj_shape: Expected projection shape for filtering.
        label_type: Label quality type ("high", "low", etc.).
        batch_size: Batch size for DataLoaders.
        num_workers: Number of worker processes for data loading.
        train_transform: Transform pipeline for training data.
        val_transform: Transform pipeline for validation data.
        test_transform: Transform pipeline for test data.
        cache_enabled: If True, enables caching in datasets.
        normalize: If True, normalizes images.
        use_rgb: If True, uses only RGB bands.

    Returns:
        Tuple of (train_loader, val_loader, test_loader). Any loader may be
        None if the corresponding split has no samples.
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

    train_dataset = (
        CloudSEN12Dataset(
            train_tdf, train_transform, cache_enabled, normalize, use_rgb
        )
        if len(train_tdf)
        else None
    )
    val_dataset = (
        CloudSEN12Dataset(val_tdf, val_transform, cache_enabled, normalize, use_rgb)
        if len(val_tdf)
        else None
    )
    test_dataset = (
        CloudSEN12Dataset(test_tdf, test_transform, cache_enabled, normalize, use_rgb)
        if len(test_tdf)
        else None
    )

    train_loader = (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        if train_dataset
        else None
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        if val_dataset
        else None
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        if test_dataset
        else None
    )

    return train_loader, val_loader, test_loader