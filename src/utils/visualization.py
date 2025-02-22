import torch
from torch.utils.data import DataLoader
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


def plot_metrics(history: dict):
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(16, 4))
    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"],   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()
    
    # Acurácia
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"],   label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()
    
    # IoU
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["train_iou"], label="Train IoU")
    plt.plot(epochs, history["val_iou"],   label="Val IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("IoU over Epochs")
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_sample_prediction(model, dataset, idx: int = 0, device: str = "cuda", rgb_bands=(3,2,1)):
    """
    Plota lado a lado:
      - Imagem original (RGB)
      - Máscara predita pelo modelo (multiclasse)
      - Máscara real (caso esteja disponível no dataset como segundo elemento)
    
    O modelo já realiza upsampling dos logits para o tamanho original da imagem.
    Caso a máscara predita ainda não esteja com o mesmo shape da imagem (por questões de arredondamento),
    ela é reamostrada (após o argmax) com interpolação 'nearest' para preservar as classes.
    
    Parâmetros:
      - model: modelo treinado que retorna logits de shape (1, num_classes, H, W)
      - dataset: objeto Dataset ou DataLoader
      - idx: índice da amostra a ser visualizada
      - device: dispositivo de computação ("cuda" ou "cpu")
      - rgb_bands: tupla com os índices das bandas para formar a imagem RGB
    """
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    model.eval()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tenta acessar o sample
    try:
        sample = dataset.dataset[idx] if hasattr(dataset, "dataset") else dataset[idx]
    except Exception as e:
        print(f"Erro ao acessar o índice {idx} do dataset: {e}")
        return

    # Se o sample for tuple/list, assume que:
    #  - sample[0] é a imagem
    #  - sample[1] é a máscara real (se disponível)
    if isinstance(sample, (tuple, list)):
        image = sample[0]
        if len(sample) > 1:
            true_mask = sample[1]
            has_true_mask = True
        else:
            true_mask = None
            has_true_mask = False
    else:
        image = sample
        true_mask = None
        has_true_mask = False

    # Prepara a imagem: (1, C, H, W)
    image = image.to(device).unsqueeze(0)
    original_size = image.shape[-2:]  # (H, W)

    with torch.no_grad():
        outputs = model(image)  # Espera (1, num_classes, H, W)
        # Já estão no tamanho original?
        preds = torch.argmax(outputs, dim=1)  # (1, H_pred, W_pred)
        if preds.shape[-2:] != original_size:
            preds = F.interpolate(preds.unsqueeze(1).float(), size=original_size, mode='nearest')
            preds = preds.squeeze(1).long()
    
    # Converte tensores para numpy
    image_np = image.squeeze(0).cpu().numpy()  # (C, H, W)
    preds_np = preds.squeeze(0).cpu().numpy()    # (H, W)

    # Prepara a imagem RGB com as bandas definidas
    rgb = image_np[list(rgb_bands), :, :]  # (3, H, W)
    rgb = np.transpose(rgb, (1, 2, 0))       # (H, W, 3)
    rgb = np.clip(rgb / 10000.0, 0, 1)

    # Configuração do colormap
    cmap = mcolors.ListedColormap(["black", "red", "blue", "green"])
    bounds = [0, 1, 2, 3, 4]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # Define extent para padronizar os eixos (usando as dimensões originais)
    extent = [0, original_size[1], original_size[0], 0]

    # Número de colunas: 3 se houver máscara real, senão 2
    ncols = 3 if has_true_mask else 2
    fig, axes = plt.subplots(1, ncols, figsize=(12, 4))
    if ncols == 2:
        axes = [axes[0], axes[1]]
    
    # Plot da imagem original (RGB)
    axes[0].imshow(rgb, extent=extent, aspect='equal')
    axes[0].set_title("Imagem Original (RGB)")
    axes[0].axis("off")

    # Plot da máscara predita
    im = axes[1].imshow(preds_np, cmap=cmap, norm=norm, extent=extent, aspect='equal', interpolation='nearest')
    axes[1].set_title("Máscara Predita")
    axes[1].axis("off")
    
    # Cria um eixo separado para a colorbar sem reduzir o espaço do subplot principal
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ticks=[0.5, 1.5, 2.5, 3.5])
    cbar.ax.set_yticklabels(["Clear", "Thick Cloud", "Thin Cloud", "Cloud Shadow"])

    # Se houver máscara real, plota-a
    if has_true_mask:
        if torch.is_tensor(true_mask):
            true_mask = true_mask.to(device)
            true_mask_np = true_mask.squeeze(0).cpu().numpy()  # assume (1, H, W)
        else:
            true_mask_np = true_mask

        axes[2].imshow(true_mask_np, cmap=cmap, norm=norm, extent=extent, aspect='equal', interpolation='nearest')
        axes[2].set_title("Máscara Real")
        axes[2].axis("off")

    plt.tight_layout()
    plt.show()