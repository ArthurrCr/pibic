import torch
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from typing import List, Tuple, Optional

# Constantes globais
SENTINEL_BANDS = ["B01","B02","B03","B04","B05","B06",
                  "B07","B08","B8A","B09","B10","B11","B12"]
CLASS_NAMES = ['Clear','Thick Cloud','Thin Cloud','Cloud Shadow']
EXPERIMENTS = {
    'cloud/no cloud': {'pos': [1, 2]},
    'cloud shadow': {'pos': [3]},
    'valid/invalid' : {'pos': [0]} 
}


def load_models(model_paths: List[str], pytorch_device: torch.device) -> List:
    """Carrega modelos PyTorch de uma lista de caminhos."""
    models = []
    for model_path in model_paths:
        print(f"Carregando modelo: {model_path}")
        model = torch.load(model_path, map_location=pytorch_device, weights_only=False)
        model.eval()
        models.append(model)
    return models


def get_normalization_stats(pytorch_device, fp16_mode, required_bands):
    """Retorna estatísticas de normalização para bandas Sentinel-2 L1C. do CloudS2Mask."""

    L1C_mean = {
        "B01": 0.072623697227855,  "B02": 0.06608867585127501,
        "B03": 0.061940767467830685, "B04": 0.06330473795822207,
        "B05": 0.06858655023065205, "B06": 0.08539433443008514,
        "B07": 0.09401670610922229, "B08": 0.09006412206990828,
        "B8A": 0.09915093732164396, "B09": 0.035429756513690985,
        "B10": 0.003632839439909688, "B11": 0.06855744750648961,
        "B12": 0.0486043830034996,
    }
    L1C_std = {
        "B01": 0.020152047138155018, "B02": 0.022698212883948143,
        "B03": 0.023073879486441455, "B04": 0.02668270641026416,
        "B05": 0.0263763340626224,   "B06": 0.027439342904551974,
        "B07": 0.02896087163616576,  "B08": 0.028661147214616267,
        "B8A": 0.0301365958005653,   "B09": 0.013482676031864258,
        "B10": 0.0019204000834290252,"B11": 0.023938917594669776,
        "B12": 0.020069414811037536,
    }

    mean_subset = torch.tensor([L1C_mean[b] for b in required_bands])
    std_subset  = torch.tensor([L1C_std[b]  for b in required_bands])

    mean = mean_subset.view(1, len(required_bands), 1, 1).to(pytorch_device)
    std  = std_subset.view(1, len(required_bands), 1, 1).to(pytorch_device)
    if fp16_mode:
        mean, std = mean.half(), std.half()
    return mean, std


def ensemble_inference(models, images, return_probs: bool = False) -> torch.Tensor:
    """
    Realiza inferência em ensemble de modelos.
    
    Args:
        models: Lista de modelos
        images: Batch de imagens
        return_probs: Se True, retorna probabilidades médias; senão, retorna classes
    
    Returns:
        Probabilidades ou classes preditas
    """
    with torch.no_grad():
        probs = torch.stack([torch.softmax(m(images), 1) for m in models]).mean(0)
    return probs if return_probs else torch.argmax(probs, 1)


def get_predictions(models, images, use_ensemble=True, return_probs=False):
    """Função unificada para obter predições."""
    if use_ensemble and len(models) > 1:
        return ensemble_inference(models, images, return_probs=return_probs)
    else:
        output = models[0](images)
        probs = torch.softmax(output, 1)
        return probs if return_probs else torch.argmax(probs, 1)


def normalize_images(images, mean, std):
    """Normaliza imagens Sentinel-2. de acordo com o CloudS2Mask."""
    return (images / 32767.0 - mean) / std

def evaluate_clouds2mask(test_loader, models, device='cuda',
                        use_ensemble=True, normalize_imgs=True):
    """
    Avalia modelos e retorna matriz de confusão agregada.
    
    Args:
        test_loader: DataLoader com dados de teste
        models: Lista de modelos ou modelo único
        device: Dispositivo para execução
        processing_level: Nível de processamento ('L1C')
        use_ensemble: Se deve usar ensemble
        normalize_imgs: Se deve normalizar imagens  # ALTERADO
    
    Returns:
        Matriz de confusão 4x4
    """
    mean, std = get_normalization_stats(device, False, SENTINEL_BANDS)

    if not isinstance(models, list):
        models = [models]
    for m in models:
        m.to(device).eval()

    conf_matrix = np.zeros((4,4), dtype=np.int64)
    print("Iniciando avaliação...")

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Processando batches"):
            images, labels = images.to(device).float(), labels.to(device)

            if normalize_imgs:  # ALTERADO
                images = normalize_images(images, mean, std)

            preds = get_predictions(models, images, use_ensemble=use_ensemble)

            batch_conf = confusion_matrix(labels.cpu().numpy().ravel(),
                                          preds.cpu().numpy().ravel(),
                                          labels=[0,1,2,3])
            conf_matrix += batch_conf
            torch.cuda.empty_cache()
    return conf_matrix


def compute_metrics(conf_matrix):
    """Calcula métricas por classe a partir da matriz de confusão."""
    metrics = {}
    for i, name in enumerate(CLASS_NAMES):
        TP = conf_matrix[i,i]
        FN = conf_matrix[i,:].sum() - TP
        FP = conf_matrix[:,i].sum() - TP
        precision = TP / (TP+FP+1e-7)
        recall    = TP / (TP+FN+1e-7)
        f1        = 2*precision*recall/(precision+recall+1e-7)
        metrics[name] = {
            'F1-Score': f1,
            'Precision': precision,
            'Recall': recall,
            'Omission Error': FN/(TP+FN+1e-7),
            'Commission Error': FP/(TP+FP+1e-7),
            'Support': conf_matrix[i,:].sum()
        }
    metrics['Overall'] = {
        'Accuracy': np.diag(conf_matrix).sum()/conf_matrix.sum(),
        'Total Samples': conf_matrix.sum()
    }
    return metrics


def plot_confusion_matrix(
        conf_matrix: np.ndarray,
        normalize: bool = True,
        class_names: List[str] = CLASS_NAMES,
        title: Optional[str] = None,
        cmap: str = "Blues",
        figsize: Tuple[int, int] = (8, 6),
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
    """
    Plota uma matriz de confusão (absoluta ou normalizada linha‑a‑linha).

    Parameters
    ----------
    conf_matrix : np.ndarray
        Matriz quadrada de confusão (shape = [n_classes, n_classes]).
    normalize : bool, default=True
        Se True, converte cada linha em percentuais (%).  
    class_names : list of str
        Rótulos que aparecem nos eixos X e Y.
    title : str or None
        Título do gráfico. Se None, um título padrão é criado.
    cmap : str
        Colormap usado pelo `seaborn.heatmap`.
    figsize : (int, int)
        Tamanho da figura caso `ax` não seja fornecido.
    ax : matplotlib.axes.Axes or None
        Eixo alvo. Se None, é criado automaticamente.

    Returns
    -------
    matplotlib.axes.Axes
        O eixo contendo a matriz de confusão.
    """
    # --- Normalização (linha‑a‑linha) ---------------------------------
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            cm_show = conf_matrix.astype(float) / conf_matrix.sum(axis=1, keepdims=True)
        cm_show = np.nan_to_num(cm_show) * 100  # converte para %
        fmt, cbar_label = ".1f", "Porcentagem (%)"
    else:
        cm_show, fmt, cbar_label = conf_matrix, "d", "Contagem"

    # --- Criação da figura/axes ---------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm_show,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": cbar_label},
        ax=ax
    )

    ax.set_xlabel("Classe Predita")
    ax.set_ylabel("Classe Verdadeira")

    if title is None:
        title = "Matriz de Confusão"
        if normalize:
            title += " (normalizada)"

    ax.set_title(title)

    plt.tight_layout()
    return ax