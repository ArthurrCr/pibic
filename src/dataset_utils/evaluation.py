import rasterio
import tempfile
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy.ma as ma
import tensorflow as tf
from s2cloudless import S2PixelCloudDetector


def generate_mask_s2cloudless(bands_array, threshold, avg_over, dil_size):
    """
    bands_array: array (H, W, 13) (float ou uint16 normalizado).
    Retorna np.uint8 (0 ou 1) como máscara de nuvem.
    """

    cloud_detector = S2PixelCloudDetector(
        threshold=threshold,
        all_bands=True,
        average_over=avg_over,
        dilation_size=dil_size
    )
    # s2cloudless pede shape (batch, H, W, C)
    batch = bands_array[np.newaxis, ...]  # (1, H, W, 13)
    mask = cloud_detector.get_cloud_masks(batch)[0]
    return mask.astype(np.uint8)


def compute_confusion_matrix_4classes(gt_flat, pred_flat):
    """
    Computa a matriz de confusão 4x4 (classes 0..3).
    Mesmo se a predição não tiver classe 2 ou 3, as colunas/linhas ficam zeradas.
    """
    cm = np.zeros((4, 4), dtype=np.int64)
    for g, p in zip(gt_flat, pred_flat):
        cm[g, p] += 1
    return cm

def compute_metrics_from_cm(cm):
    """
    Extrai métricas one-vs-all para classes 0, 1 e 3.
    Ignoramos explicitamente a classe 2 ('thin cloud').
    """
    eps = 1e-7
    results = {}
    # classes que queremos avaliar (0=clear, 1=thick cloud, 3=shadow)
    classes_interesse = [0, 1, 3]

    for c in classes_interesse:
        TP = cm[c, c]
        FP = np.sum(cm[:, c]) - TP
        FN = np.sum(cm[c, :]) - TP

        precision = TP / (TP + FP + eps)
        recall    = TP / (TP + FN + eps)
        f1        = 2 * precision * recall / (precision + recall + eps)
        iou       = TP / (TP + FP + FN + eps)

        results[f"precision_{c}"] = float(precision)
        results[f"recall_{c}"]    = float(recall)
        results[f"f1_{c}"]        = float(f1)
        results[f"iou_{c}"]       = float(iou)

    return results

def compute_metrics(gt_array, pred_mask):
    """
    gt_array:  (H, W) com valores em {0,1,2,3} => 0=clear,1=thick,2=thin,3=shadow
    pred_mask: (H, W) com valores em {0,1,3} => 0=clear,1=cloud,3=shadow
    Retorna métricas para classes 0,1,3. A classe 2 é ignorada no cálculo.
    """
    gt_flat   = gt_array.flatten()
    pred_flat = pred_mask.flatten()

    cm = compute_confusion_matrix_4classes(gt_flat, pred_flat)
    return compute_metrics_from_cm(cm)

def visualize_results(sample, cloud_mask, shadow_mask):
    """
    Exemplo de função para visualizar:
    (1) imagem RGB, (2) ground truth (0,1,2,3) e (3) predição combinada (0=clear,1=cloud,3=shadow).
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    # Ler a imagem RGB (usando bandas 4,3,2)
    with rasterio.open(sample.read(0)) as src:
        bands = src.read([4, 3, 2])  # shape: (3, H, W)
    rgb = bands.transpose(1, 2, 0).astype(np.float32)  # (H, W, 3)
    perc = np.percentile(rgb, 98)
    rgb = np.clip(rgb / perc, 0, 1)

    # Ler a ground truth (banda 1)
    with rasterio.open(sample.read(1)) as src_mask:
        ground_truth_full = src_mask.read(1)  # shape: (H, W)

    # Predição 3-classes: 0=clear,1=cloud,3=shadow
    pred_3class = np.zeros_like(cloud_mask, dtype=np.uint8)
    pred_3class[cloud_mask == 1]  = 1
    pred_3class[shadow_mask == 1] = 3

    # Configura a exibição da GT:
    #   0 -> white
    #   1 -> blue
    #   2 -> yellow
    #   3 -> gray
    cmap_gt = colors.ListedColormap(["white", "blue", "yellow", "gray"])

    # Configura a exibição da predição:
    #   0 -> white
    #   1 -> blue
    #   3 -> gray
    # (Aqui, a classe 2 não existe)
    cmap_pred = colors.ListedColormap(["white", "blue", "gray"])

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # 1) Imagem RGB
    ax[0].imshow(rgb)
    ax[0].set_title("Imagem RGB")
    ax[0].axis("off")

    # 2) Ground Truth (0,1,2,3)
    ax[1].imshow(ground_truth_full, cmap=cmap_gt, vmin=0, vmax=3)
    ax[1].set_title("Ground Truth")
    ax[1].axis("off")

    # 3) Predição (0,1,3)
    ax[2].imshow(pred_3class, cmap=cmap_pred, vmin=0, vmax=3)
    ax[2].set_title("Predição")
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()