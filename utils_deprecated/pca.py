import numpy as np
from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import types

class CloudSEN12PCADataset(Dataset):
    def __init__(self, base_dataset, pca, pca_components=None):
        """
        base_dataset: CloudSEN12Dataset original
        pca: Objeto PCA treinado
        pca_components: Número de componentes para reter (None = mantém todos)
        """
        self.base_dataset = base_dataset
        self.pca = pca
        self.pca_components = pca_components

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, mask = self.base_dataset[idx]
        
        # Converter para numpy e remodelar
        img_np = img.numpy()
        c, h, w = img_np.shape
        img_reshaped = img_np.transpose(1, 2, 0).reshape(-1, c)  # (H*W, C)

        # Aplicar PCA
        img_transformed = self.pca.transform(img_reshaped)
        
        # Reduzir componentes se especificado
        if self.pca_components is not None:
            img_transformed = img_transformed[:, :self.pca_components]
        
        # Remodelar para imagem
        n_components = img_transformed.shape[1]
        img_pca = img_transformed.reshape(h, w, n_components).transpose(2, 0, 1)
        
        return torch.tensor(img_pca, dtype=torch.float32), mask

def compute_pca(dataset, n_components=None, sample_size=1000000):
    """
    Calcula PCA usando amostragem do dataset
    
    Args:
        dataset: CloudSEN12Dataset
        n_components: Número de componentes para PCA
        sample_size: Máximo de pixels para amostrar
        
    Returns:
        PCA treinado
    """
    pixels = []
    total_pixels = 0
    
    # Coletar amostras de pixels
    for i in tqdm(range(len(dataset)), desc="Coletando amostras para PCA"):
        img, _ = dataset[i]
        img_np = img.numpy()
        c, h, w = img_np.shape
        img_flat = img_np.transpose(1, 2, 0).reshape(-1, c)
        
        # Amostragem aleatória
        n_pixels = img_flat.shape[0]
        if total_pixels + n_pixels > sample_size:
            idx = np.random.choice(n_pixels, sample_size - total_pixels, replace=False)
            pixels.append(img_flat[idx])
            break
        
        pixels.append(img_flat)
        total_pixels += n_pixels
        if total_pixels >= sample_size:
            break
    
    pixels = np.concatenate(pixels, axis=0)
    print(f"Total de pixels amostrados: {pixels.shape[0]}")
    
    # Treinar PCA
    pca = PCA(n_components=n_components)
    pca.fit(pixels)
    
    # Mostrar variância explicada
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    print("\nVariância explicada por componente:")
    for i, var in enumerate(explained_variance):
        print(f"Componente {i+1}: {var:.4f}")
    
    return pca

def apply_pca_to_loaders(train_loader, val_loader, test_loader, pca_components=None):
    """
    Aplica PCA aos dataloaders
    
    Args:
        train_loader: Dataloader de treino original
        val_loader: Dataloader de validação original
        test_loader: Dataloader de teste original
        pca_components: Número de componentes a reter
        
    Returns:
        Tupla com novos dataloaders (train, val, test) com PCA aplicado
    """
    # Calcular PCA usando dataset de treino
    train_dataset = train_loader.dataset
    pca = compute_pca(train_dataset, n_components=pca_components)
    
    # Criar novos datasets com PCA
    train_pca_dataset = CloudSEN12PCADataset(train_dataset, pca, pca_components)
    val_pca_dataset = CloudSEN12PCADataset(val_loader.dataset, pca, pca_components)
    test_pca_dataset = CloudSEN12PCADataset(test_loader.dataset, pca, pca_components)
    
    # Criar novos dataloaders mantendo configuração original
    def get_loader_params(loader):
        """Extrai parâmetros de configuração do DataLoader"""
        params = {
            'batch_size': loader.batch_size,
            'num_workers': loader.num_workers,
            'pin_memory': loader.pin_memory,
        }
        
        # Determinar se deve embaralhar (apenas para treino)
        if loader == train_loader:
            params['shuffle'] = True
        else:
            params['shuffle'] = False
            
        return params
    
    train_pca_loader = DataLoader(train_pca_dataset, **get_loader_params(train_loader))
    val_pca_loader = DataLoader(val_pca_dataset, **get_loader_params(val_loader))
    test_pca_loader = DataLoader(test_pca_dataset, **get_loader_params(test_loader))
    
    return train_pca_loader, val_pca_loader, test_pca_loader