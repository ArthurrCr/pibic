import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


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


def plot_sample_prediction(model, dataset, idx: int = 0, device: str = "cuda"):
    """
    Plota lado a lado:
      - Imagem de entrada (RGB),
      - Máscara predita pelo modelo (binária).

    Se for multiclasse, você teria que ajustar a maneira como gera preds
    e a coloração (argmax e exibir cada classe com uma cor).
    """
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)

    try:
        image = dataset[idx]  # [3, H, W] se binário, ou (image, mask) dependendo do dataset
    except Exception as e:
        print(f"Erro ao acessar o índice {idx} do dataset: {e}")
        return

    # Se vier (img, mask), ignore a mask neste plot
    if isinstance(image, (tuple, list)):
        image = image[0]  # Pega só a imagem

    image = image.to(device).unsqueeze(0)  # [1, 3, H, W] ou [1, C, H, W]

    with torch.no_grad():
        outputs = model(image)
        probs   = torch.sigmoid(outputs)
        preds   = (probs > 0.5).float()  # binário. Para multiclasse, seria argmax(outputs, dim=1)

    # Converte para CPU/Numpy
    image_np = image.squeeze(0).cpu().numpy()  # [3, H, W]
    preds_np = preds.squeeze(0).squeeze(0).cpu().numpy()  # [H, W] para binário

    # Ajusta [3, H, W] -> [H, W, 3]
    image_np = np.transpose(image_np, (1, 2, 0))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # Imagem original
    axes[0].imshow(image_np)
    axes[0].set_title("Imagem Original")
    axes[0].axis("off")

    # Máscara Predita
    axes[1].imshow(preds_np, cmap="gray")
    axes[1].set_title("Máscara Predita")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
