import torch
import numpy as np
from torchmetrics.classification import ConfusionMatrix
from sklearn.metrics import classification_report

def evaluate_iou_report(model, loader, num_classes=4, class_names=None, device="cuda"):
    """
    Gera um 'relatório' de IoU por classe e a média (mIoU), 
    ao estilo classification_report, mas focado em IoU.

    Parâmetros:
      - model: modelo PyTorch, retorna logits (B, num_classes, H, W).
      - loader: DataLoader com (imgs, masks),
                onde masks têm shape (B, H, W) ou (B, 1, H, W) com valores em [0..num_classes-1].
      - num_classes: número de classes (ex.: 4 para nuvens: [clear, thick, thin, shadow]).
      - class_names: lista de nomes de classes, ex. ["Clear", "Thick Cloud", "Thin Cloud", "Shadow"].
      - device: "cuda" ou "cpu".

    Retorna:
      - iou_dict: dicionário com iou_dict["class_iou"] = valor, iou_dict["mean_iou"] = valor
      - (y_true_all, y_pred_all): arrays 1D contendo rótulos e predições flatten.
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    model.eval()
    model.to(device)

    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            if masks.dim() == 4 and masks.shape[1] == 1:
                # se (B,1,H,W), vira (B,H,W)
                masks = masks.squeeze(1)
            masks = masks.to(device)

            # Forward
            outputs = model(imgs)  # (B, num_classes, H, W)
            preds = torch.argmax(outputs, dim=1)  # (B, H, W)

            # Flatten
            y_true_list.append(masks.view(-1).cpu().numpy())
            y_pred_list.append(preds.view(-1).cpu().numpy())

    # Concatena todos os batches
    y_true_all = np.concatenate(y_true_list, axis=0)
    y_pred_all = np.concatenate(y_pred_list, axis=0)

    # Calcula a IoU por classe
    ious = []
    for c in range(num_classes):
        intersection = np.sum((y_true_all == c) & (y_pred_all == c))
        union = np.sum((y_true_all == c) | (y_pred_all == c))
        iou_c = intersection / union if union != 0 else 0.0
        ious.append(iou_c)

    mean_iou = np.mean(ious)

    # Monta um dicionário ou texto similar a "classification_report"
    iou_dict = {}
    for c, class_name in enumerate(class_names):
        iou_dict[class_name] = ious[c]
    iou_dict["mean_iou"] = mean_iou

    return iou_dict, (y_true_all, y_pred_all)

def print_iou_report(iou_dict):
    """
    Imprime um 'relatório' textual no estilo classification_report, mas para IoU.
    """
    lines = []
    lines.append("IoU Report:\n")
    for k, v in iou_dict.items():
        if k == "mean_iou":
            continue
        lines.append(f"  {k:20s}: {v:.4f}")
    lines.append(f"\nMean IoU       : {iou_dict['mean_iou']:.4f}")
    report_str = "\n".join(lines)
    print(report_str)

def evaluate_classification_report(model, loader, device="cuda"):
    """
    Gera um classification report (sklearn) para um modelo de segmentação multiclasses.
    - model: modelo PyTorch que retorna (B, num_classes, H, W).
    - loader: DataLoader com (imgs, masks), onde masks contêm valores em [0..num_classes-1].
    - device: "cuda" ou "cpu".
    
    Retorna: texto do classification_report e (y_true_all, y_pred_all) 
             para caso queira imprimir ou analisar.
    """
    model.eval()
    model.to(device)

    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            # Se a máscara vier como (B,1,H,W), converta para (B,H,W):
            if masks.dim() == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)
            masks = masks.to(device)

            # Forward
            outputs = model(imgs)  # (B, num_classes, H, W)
            preds = torch.argmax(outputs, dim=1)  # (B, H, W)

            # Achata (flatten) tudo para 1D
            # ex.: (B,H,W) -> (B*H*W,)
            y_true_flat = masks.view(-1).cpu().numpy()
            y_pred_flat = preds.view(-1).cpu().numpy()

            y_true_list.append(y_true_flat)
            y_pred_list.append(y_pred_flat)

    # Concatena todos os batches num vetor só
    y_true_all = np.concatenate(y_true_list, axis=0)
    y_pred_all = np.concatenate(y_pred_list, axis=0)

    # Gerar relatório. Ajuste 'target_names' se tiver nomes específicos.
    # labels=[0,1,2,3] e target_names para cada classe.
    report = classification_report(
        y_true_all,
        y_pred_all,
        labels=[0, 1, 2, 3],
        target_names=["Clear", "Thick Cloud", "Thin Cloud", "Cloud Shadow"],
        zero_division=0  # para evitar warnings se alguma classe não aparecer
    )
    return report, (y_true_all, y_pred_all)


def generate_confusion_matrix(model, data_loader, device="cuda", num_classes=2):
    """
    Gera a matriz de confusão (NxN) para todo o data_loader, usando TorchMetrics.
    Pode lidar tanto com problemas binários (num_classes=2) quanto multiclasses.
    """
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Configura a métrica de matriz de confusão de acordo com o problema
    if num_classes == 2:
        cm_metric = ConfusionMatrix(task="binary").to(device)
    else:
        cm_metric = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
    
    cm_metric.reset()

    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            # Para multiclasse, as masks devem estar no formato (B, H, W) com valores inteiros 0..num_classes-1.
            if masks.dim() == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)
            masks = masks.long().to(device)

            outputs = model(images)

            if num_classes == 2:
                # Binário: aplica sigmoide e threshold
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long()
            else:
                # Multiclasse: usa argmax no canal de classes
                preds = torch.argmax(outputs, dim=1)

            cm_metric.update(preds, masks)

    conf_matrix = cm_metric.compute().cpu().numpy()
    return conf_matrix

def compute_segmentation_metrics(conf_matrix, num_classes=2):
    """
    Calcula métricas de segmentação a partir de uma matriz de confusão NxN.
    
    Para binário (num_classes=2), a conf_matrix tem formato:
         [[TN, FP],
          [FN, TP]]
      Calcula Accuracy, Precision, Recall, F1, IoU, etc.
    
    Para multiclasses (num_classes > 2), a conf_matrix é NxN.
    Calcula as métricas por classe e as médias macro.
    """
    if num_classes == 2:
        TN, FP, FN, TP = conf_matrix.ravel()
        accuracy  = (TP + TN) / (TP + TN + FP + FN + 1e-6)
        precision = TP / (TP + FP + 1e-6)
        recall    = TP / (TP + FN + 1e-6)
        f1_score  = 2 * (precision * recall) / (precision + recall + 1e-6)
        iou       = TP / (TP + FP + FN + 1e-6)
        return {
            "accuracy":  accuracy,
            "precision": precision,
            "recall":    recall,
            "f1_score":  f1_score,
            "iou":       iou
        }
    else:
        total_samples = conf_matrix.sum()
        correct = sum(conf_matrix[c, c] for c in range(num_classes))
        accuracy = correct / (total_samples + 1e-6)
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []
        iou_per_class = []

        for c in range(num_classes):
            TP = conf_matrix[c, c]
            FP = conf_matrix[:, c].sum() - TP
            FN = conf_matrix[c, :].sum() - TP
            prec = TP / (TP + FP + 1e-6)
            rec  = TP / (TP + FN + 1e-6)
            f1   = 2 * (prec * rec) / (prec + rec + 1e-6)
            iou  = TP / (TP + FP + FN + 1e-6)
            precision_per_class.append(prec)
            recall_per_class.append(rec)
            f1_per_class.append(f1)
            iou_per_class.append(iou)
        
        macro_precision = np.mean(precision_per_class)
        macro_recall    = np.mean(recall_per_class)
        macro_f1        = np.mean(f1_per_class)
        macro_iou       = np.mean(iou_per_class)

        return {
            "accuracy": accuracy,
            "precision": macro_precision,
            "recall": macro_recall,
            "f1_score": macro_f1,
            "iou": macro_iou
        }