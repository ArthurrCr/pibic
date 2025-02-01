import torch
import numpy as np
from torchmetrics.classification import ConfusionMatrix

def generate_confusion_matrix(
    model, 
    data_loader, 
    device="cuda", 
    num_classes=2
):
    """
    Gera a matriz de confusão (NxN) para todo o data_loader, usando TorchMetrics.
    Pode lidar tanto com problemas binários (num_classes=2) quanto multiclasse.
    """
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Definir ConfusionMatrix do TorchMetrics
    if num_classes == 2:
        cm_metric = ConfusionMatrix(task="binary").to(device)
    else:
        cm_metric = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)

    cm_metric.reset()

    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            # Para multiclasse, as masks/labels devem ser long() [0..num_classes-1].
            masks = masks.long().to(device)

            outputs = model(images)

            if num_classes == 2:
                # Binário
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long()
            else:
                # Multiclasse: argmax no canal de classes
                preds = torch.argmax(outputs, dim=1)

            cm_metric.update(preds, masks)

    conf_matrix = cm_metric.compute().cpu().numpy()
    return conf_matrix


def compute_segmentation_metrics(conf_matrix, num_classes=2):
    """
    Calcula métricas de segmentação a partir de uma matriz de confusão NxN.
    - Para binário (num_classes=2), a conf_matrix tem formato:
         [[TN, FP],
          [FN, TP]]
      Calculamos Accuracy, Precision, Recall, F1, IoU, etc.

    - Para multiclasse (num_classes > 2), a conf_matrix é NxN.
      Aqui fazemos um exemplo de macro-averaging.
    """
    if num_classes == 2:
        TN, FP, FN, TP = conf_matrix.ravel()
        accuracy  = (TP + TN) / (TP + TN + FP + FN + 1e-6)
        precision = TP / (TP + FP + 1e-6)
        recall    = TP / (TP + FN + 1e-6)
        f1_score  = 2 * (precision * recall) / (precision + recall + 1e-6)
        iou       = TP / (TP + FP + FN + 1e-6)
        # dice    = 2 * TP / (2 * TP + FP + FN + 1e-6) # mesmo que F1 no binário

        return {
            "accuracy":  accuracy,
            "precision": precision,
            "recall":    recall,
            "f1_score":  f1_score,
            "iou":       iou
        }

    else:
        # Multiclasse NxN
        num_classes = conf_matrix.shape[0]
        total_samples = conf_matrix.sum()
        correct = sum(conf_matrix[c, c] for c in range(num_classes))
        accuracy = correct / (total_samples + 1e-6)

        precision_per_class = []
        recall_per_class    = []
        f1_per_class        = []
        iou_per_class       = []

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

        # macro average
        precision = np.mean(precision_per_class)
        recall    = np.mean(recall_per_class)
        f1_score  = np.mean(f1_per_class)
        iou       = np.mean(iou_per_class)

        return {
            "accuracy":  accuracy,
            "precision": precision,
            "recall":    recall,
            "f1_score":  f1_score,
            "iou":       iou
        }
