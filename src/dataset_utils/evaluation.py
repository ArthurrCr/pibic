import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, f1_score, recall_score, precision_score
from matplotlib.colors import ListedColormap, BoundaryNorm
from s2cloudless import S2PixelCloudDetector

def generate_mask(sample):
    # --- Leitura e normalização das bandas ---
    with rasterio.open(sample.read(0)) as src:
        bands = src.read(range(1, src.count + 1))
    bands = bands.transpose(1, 2, 0)  # shape: (H, W, 13)
    bands = bands / 10000.0

    # --- Inicializa o detector s2cloudless ---
    cloud_detector = S2PixelCloudDetector(threshold=0.3, all_bands=True, average_over=4, dilation_size=2)

    # --- Adiciona dimensão de lote (batch) ---
    bands_batch = bands[np.newaxis, ...]  # shape: (1, H, W, 13)

    # --- Gera a máscara binária de nuvens ---
    cloud_masks = cloud_detector.get_cloud_masks(bands_batch)
    cloud_mask = cloud_masks[0]  # Remove a dimensão de lote: (H, W)

    return cloud_mask.astype(np.uint8)

def compute_metrics(sample, gt_mask, pred_mask):
    """
    Calcula as métricas (IoU, F1, Recall, Precision) entre a ground truth e a máscara predita.
    
    Ambos devem ser arrays 2D com os mesmos valores (0 para clear, 1 para cloud, 2 para shadow).
    """
    # --- Leitura e remapeamento da máscara ground truth ---
    with rasterio.open(sample.read(1)) as src_mask:
        ground_truth = src_mask.read(1)
    # Remapeamento:
    # - Mantém Thick Cloud (1)
    # - Converte o resto pra 0
    gt_mask = np.where(ground_truth == 1, 1, 0)

    gt_flat = gt_mask.flatten().astype(np.uint8)
    pred_flat = pred_mask.flatten().astype(np.uint8)
    iou = jaccard_score(gt_flat, pred_flat, average=None, labels=[0, 1], zero_division=0)
    metrics = {
        "IoU_Clear": iou[0],
        "IoU_Cloud": iou[1],
        "F1": f1_score(gt_flat, pred_flat, average="macro", zero_division=0),
        "Recall": recall_score(gt_flat, pred_flat, average="macro", zero_division=0),
        "Precision": precision_score(gt_flat, pred_flat, average="macro", zero_division=0)
    }
    return metrics

def visualize_results(sample, gt_mask, pred_mask):
    from matplotlib import colors
    # Carrega a imagem RGB local (supondo que os canais 4, 3, 2 correspondem a RGB)
    with rasterio.open(sample.read(0)) as src:
        # Lê os canais 4, 3 e 2
        bands = src.read([4, 3, 2])
    rgb = bands.transpose(1, 2, 0).astype(np.float32)
    # Normaliza para visualização
    perc = np.percentile(rgb, 98)
    rgb = np.clip(rgb / perc, 0, 1)

    # Define os mapas de cor customizados
    # Para ground truth: 0 (clear) → branco, 1 (thick cloud) → azul
    cmap_gt = colors.ListedColormap(['white', 'blue'])
    cmap_pred = colors.ListedColormap(['white', 'blue'])

    # Cria a figura com três subplots
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    ax[0].imshow(rgb)
    ax[0].set_title("Imagem RGB Local")
    ax[0].axis("off")

    ax[1].imshow(gt_mask, cmap=cmap_gt, vmin=0, vmax=1)
    ax[1].set_title("Ground Truth")
    ax[1].axis("off")

    ax[2].imshow(pred_mask, cmap=cmap_pred, vmin=0, vmax=1)
    ax[2].set_title("Predição")
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()