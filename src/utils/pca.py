"""PCA utilities for dimensionality reduction of satellite imagery."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class PCADataset(Dataset):
    """
    Dataset wrapper that applies PCA transformation to images.

    Transforms images from the original band space to PCA component space.

    Attributes:
        base_dataset: Original CloudSEN12Dataset instance.
        pca: Trained PCA object.
        pca_components: Number of components to retain (None keeps all).
    """

    def __init__(
        self,
        base_dataset: Dataset,
        pca: PCA,
        pca_components: Optional[int] = None,
    ):
        """
        Initialize the PCA dataset.

        Args:
            base_dataset: Original CloudSEN12Dataset.
            pca: Trained PCA object.
            pca_components: Number of components to retain. If None, keeps all.
        """
        self.base_dataset = base_dataset
        self.pca = pca
        self.pca_components = pca_components

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, mask = self.base_dataset[idx]

        img_np = img.numpy()
        c, h, w = img_np.shape
        img_reshaped = img_np.transpose(1, 2, 0).reshape(-1, c)

        img_transformed = self.pca.transform(img_reshaped)

        if self.pca_components is not None:
            img_transformed = img_transformed[:, : self.pca_components]

        n_components = img_transformed.shape[1]
        img_pca = img_transformed.reshape(h, w, n_components).transpose(2, 0, 1)

        return torch.tensor(img_pca, dtype=torch.float32), mask


def compute_pca(
    dataset: Dataset,
    n_components: Optional[int] = None,
    sample_size: int = 1_000_000,
) -> PCA:
    """
    Compute PCA using sampled pixels from the dataset.

    Args:
        dataset: CloudSEN12Dataset to sample from.
        n_components: Number of PCA components. If None, keeps all.
        sample_size: Maximum number of pixels to sample for fitting.

    Returns:
        Trained PCA object.
    """
    pixels = []
    total_pixels = 0

    for i in tqdm(range(len(dataset)), desc="Collecting samples for PCA"):
        img, _ = dataset[i]
        img_np = img.numpy()
        c, h, w = img_np.shape
        img_flat = img_np.transpose(1, 2, 0).reshape(-1, c)

        n_pixels = img_flat.shape[0]

        if total_pixels + n_pixels > sample_size:
            remaining = sample_size - total_pixels
            idx = np.random.choice(n_pixels, remaining, replace=False)
            pixels.append(img_flat[idx])
            break

        pixels.append(img_flat)
        total_pixels += n_pixels

        if total_pixels >= sample_size:
            break

    pixels = np.concatenate(pixels, axis=0)
    print(f"Total pixels sampled: {pixels.shape[0]}")

    pca = PCA(n_components=n_components)
    pca.fit(pixels)

    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    print("\nCumulative explained variance by component:")
    for i, var in enumerate(explained_variance):
        print(f"Component {i + 1}: {var:.4f}")

    return pca


def apply_pca_to_loaders(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    pca_components: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Apply PCA transformation to all dataloaders.

    Computes PCA on the training set and applies it to all splits.

    Args:
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        test_loader: Test DataLoader.
        pca_components: Number of PCA components to retain.

    Returns:
        Tuple of (train_pca_loader, val_pca_loader, test_pca_loader).
    """
    train_dataset = train_loader.dataset
    pca = compute_pca(train_dataset, n_components=pca_components)

    train_pca_dataset = PCADataset(train_dataset, pca, pca_components)
    val_pca_dataset = PCADataset(val_loader.dataset, pca, pca_components)
    test_pca_dataset = PCADataset(test_loader.dataset, pca, pca_components)

    def get_loader_params(loader: DataLoader, is_train: bool) -> Dict[str, Any]:
        """Extract configuration parameters from a DataLoader."""
        return {
            "batch_size": loader.batch_size,
            "num_workers": loader.num_workers,
            "pin_memory": loader.pin_memory,
            "shuffle": is_train,
        }

    train_pca_loader = DataLoader(
        train_pca_dataset, **get_loader_params(train_loader, is_train=True)
    )
    val_pca_loader = DataLoader(
        val_pca_dataset, **get_loader_params(val_loader, is_train=False)
    )
    test_pca_loader = DataLoader(
        test_pca_dataset, **get_loader_params(test_loader, is_train=False)
    )

    return train_pca_loader, val_pca_loader, test_pca_loader