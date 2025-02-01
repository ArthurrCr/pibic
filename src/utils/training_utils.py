import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.data_loader import CloudDataset

# TorchMetrics
from torchmetrics.classification import (
    BinaryAccuracy, 
    JaccardIndex,
    ConfusionMatrix
)

# AMP
from torch.amp import autocast, GradScaler
import numpy as np

class BCE_Dice_Loss(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        # BCE
        bce_loss = self.bce(preds, targets)
        
        # Sigmoid para converter logits em probabilidades
        preds_prob = torch.sigmoid(preds)

        # Dice
        intersection = (preds_prob * targets).sum(dim=(1,2,3))
        union = preds_prob.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
        dice = 1 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = dice.mean()

        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


class EarlyStopping:
    """
    Implementação simples de Early Stopping:
    - Monitora uma métrica (loss ou IoU, por exemplo).
    - Para se 'mode' for 'min', busca o menor valor da métrica (ex: loss).
    - Para se 'mode' for 'max', busca o maior valor da métrica (ex: IoU).
    - Se não houver melhora por 'patience' épocas consecutivas, interrompe o treinamento.
    """
    def __init__(self, patience=2, mode='min', min_delta=1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_metric = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_metric):
        if self.best_metric is None:
            # primeira epoch
            self.best_metric = current_metric
            return
        
        if self.mode == 'min':
            # Queremos a menor métrica
            improvement = (self.best_metric - current_metric) > self.min_delta
            if improvement:
                self.best_metric = current_metric
                self.counter = 0
            else:
                self.counter += 1
        elif self.mode == 'max':
            # Queremos a maior métrica
            improvement = (current_metric - self.best_metric) > self.min_delta
            if improvement:
                self.best_metric = current_metric
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True


def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    lr: float = 1e-4,
    device: str = "cuda",
    checkpoint_dir: str = "checkpoints",
    resume_checkpoint: str = None,
    save_best: bool = True,
    metric_to_monitor: str = "val_iou",  # Pode ser 'val_loss' ou outra métrica
    mode: str = "max",  # 'max' para IoU, 'min' para loss
    patience: int = 3,  # paciência para early stopping
    min_delta: float = 1e-4,
    use_early_stopping: bool = True
):
    """
    Função de treinamento expandida com:
    - Scheduler de LR
    - Early Stopping
    - torchmetrics
    - TensorBoard
    - Validação da estrutura das máscaras e saídas

    Args:
        model: Modelo a ser treinado.
        train_loader (DataLoader): DataLoader para o conjunto de treinamento.
        val_loader (DataLoader): DataLoader para o conjunto de validação.
        num_epochs (int): Número de épocas para treinar.
        lr (float): Taxa de aprendizado.
        device (str): Dispositivo para treinamento ('cuda' ou 'cpu').
        checkpoint_dir (str): Diretório para salvar os checkpoints.
        resume_checkpoint (str, opcional): Caminho para um checkpoint existente para retomar o treinamento.
        save_best (bool): Se True, salva o melhor modelo baseado na métrica de monitoramento.
        metric_to_monitor (str): Métrica a ser monitorada para salvar o melhor modelo (ex: 'val_iou', 'val_loss').
        mode (str): 'max' se a métrica deve ser maximizada (e.g., IoU), 'min' se deve ser minimizada (e.g., loss).
        patience (int): Pacote de épocas sem melhora para acionar Early Stopping.
        min_delta (float): Variação mínima para considerar uma melhora na métrica de monitoramento.
        use_early_stopping (bool): Se True, aplica Early Stopping.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = BCE_Dice_Loss(bce_weight=0.5)  # Para binário. Ajuste para multi-classe se necessário (ex: CrossEntropyLoss).
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Exemplo de scheduler
    if mode == 'min':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3, verbose=True
        )
    else:  # mode == 'max'
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=3, verbose=True
        )

    scaler = GradScaler()

    os.makedirs(checkpoint_dir, exist_ok=True)

    early_stopping = EarlyStopping(patience=patience, mode=mode, min_delta=min_delta) if use_early_stopping else None

    best_metric = -np.Inf if mode == "max" else np.Inf
    best_epoch = -1

    start_epoch = 0
    if resume_checkpoint is not None:
        if os.path.isfile(resume_checkpoint):
            print(f"Carregando checkpoint '{resume_checkpoint}'")
            checkpoint = torch.load(resume_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_metric = checkpoint.get('best_metric', best_metric)
            print(f"Checkpoint carregado. Retomando treinamento a partir da época {start_epoch}.")
        else:
            print(f"Checkpoint '{resume_checkpoint}' não encontrado. Treinando do início.")

    # TorchMetrics para binário
    train_acc_metric = BinaryAccuracy().to(device)
    train_iou_metric = JaccardIndex(task="binary").to(device)
    val_acc_metric   = BinaryAccuracy().to(device)
    val_iou_metric   = JaccardIndex(task="binary").to(device)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_iou": [],
        "val_iou": []
    }
    
    for epoch in range(start_epoch, num_epochs):
        # ========== TREINO ==========
        model.train()
        train_loss = 0.0
        
        # Reset métricas
        train_acc_metric.reset()
        train_iou_metric.reset()

        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] (Train)")
        for images, masks in train_loop:
            images = images.to(device, non_blocking=True)
            masks  = masks.float().to(device, non_blocking=True)

            optimizer.zero_grad()
            
            # Mixed Precision no forward (só funciona em GPU)
            if device.type == 'cuda':
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)

                    # Checagens
                    if outputs.shape != masks.shape:
                        raise ValueError("As saídas do modelo e as máscaras devem ter a mesma forma.")
                    if masks.dtype != torch.float32:
                        raise ValueError("As máscaras devem ser float32 para BCEWithLogitsLoss.")

                    loss = criterion(outputs, masks)
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            # Backward
            if device.type == 'cuda':
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()

            # Métricas
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            train_acc_metric.update(preds, masks)
            train_iou_metric.update(preds, masks)
            
            train_loop.set_postfix(loss=f"{loss.item():.4f}")

        train_acc = train_acc_metric.compute().item()
        train_iou = train_iou_metric.compute().item()
        train_loss /= len(train_loader)
        
        # ========== VALIDAÇÃO ==========
        model.eval()
        val_loss = 0.0
        
        val_acc_metric.reset()
        val_iou_metric.reset()

        val_loop = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] (Val)")
        with torch.no_grad():
            for images, masks in val_loop:
                images = images.to(device, non_blocking=True)
                masks  = masks.float().to(device, non_blocking=True)

                if device.type == 'cuda':
                    with autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(images)
                        if outputs.shape != masks.shape:
                            raise ValueError("As saídas do modelo e as máscaras devem ter a mesma forma.")
                        loss = criterion(outputs, masks)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                
                val_loss += loss.item()

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                val_acc_metric.update(preds, masks)
                val_iou_metric.update(preds, masks)
                
                val_loop.set_postfix(loss=f"{loss.item():.4f}")

        val_acc = val_acc_metric.compute().item()
        val_iou = val_iou_metric.compute().item()
        val_loss /= len(val_loader)

        # Armazena histórico
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)

        # Print summary
        print(
            f"\nEpoch [{epoch+1}/{num_epochs}] Summary:\n"
            f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n"
            f"  Train Acc:  {train_acc:.4f} | Val Acc:  {val_acc:.4f}\n"
            f"  Train IoU:  {train_iou:.4f} | Val IoU:  {val_iou:.4f}\n"
        )

        # Atualiza scheduler
        if metric_to_monitor == "val_loss":
            scheduler.step(val_loss)
            current_metric = val_loss
        elif metric_to_monitor == "val_iou":
            scheduler.step(val_iou)
            current_metric = val_iou
        elif metric_to_monitor == "val_acc":
            scheduler.step(val_acc)
            current_metric = val_acc
        else:
            current_metric = history[metric_to_monitor][-1]
            scheduler.step(current_metric)

        # Verifica se melhorou
        is_better = False
        if mode == "max" and current_metric > best_metric:
            is_better = True
        elif mode == "min" and current_metric < best_metric:
            is_better = True

        if is_better:
            best_metric = current_metric
            best_epoch = epoch + 1
            if save_best:
                best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_metric': best_metric
                }, best_model_path)
                print(f"Melhor modelo salvo com {metric_to_monitor}: {best_metric:.4f} na época {epoch+1}.")

        # Salva checkpoint atual
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_metric': best_metric
        }, checkpoint_path)
        print(f"Checkpoint salvo em '{checkpoint_path}'.")

        # Early stopping
        if use_early_stopping:
            early_stopping(current_metric)
            if early_stopping.early_stop:
                print(f"Early Stopping disparado na época {epoch+1}. Melhor {metric_to_monitor}: {early_stopping.best_metric:.4f}")
                break
        
        torch.cuda.empty_cache()

    print(f"Treinamento finalizado. Melhor {metric_to_monitor}: {best_metric:.4f} na época {best_epoch}.")
    
    return history


def generate_confusion_matrix(
    model, 
    data_loader, 
    device="cuda", 
    num_classes=2
):
    """
    Gera a matriz de confusão (NxN) para todo o data_loader, usando TorchMetrics.
    Pode lidar tanto com problemas binários (num_classes=2) quanto multiclasse.

    Args:
        model: Modelo treinado.
        data_loader: DataLoader com (imagens, máscaras/labels).
        device (str): 'cuda' ou 'cpu'.
        num_classes (int): Número de classes (2 para binário, >2 para multiclasse).

    Returns:
        conf_matrix (np.ndarray): Matriz de confusão NxN em formato numpy.
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
            # Para multiclasse, as masks/labels devem ser long() e conter valores [0..num_classes-1].
            masks = masks.long().to(device)

            outputs = model(images)

            if num_classes == 2:
                # Binário
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long()
            else:
                # Multiclasse: argmax no canal de classes
                # Exemplo: outputs shape [B, num_classes, H, W]
                preds = torch.argmax(outputs, dim=1)

            cm_metric.update(preds, masks)

    conf_matrix = cm_metric.compute().cpu().numpy()
    return conf_matrix


def compute_segmentation_metrics(conf_matrix, num_classes=2):
    """
    Calcula métricas de segmentação a partir de uma matriz de confusão NxN:

    - Para binário (num_classes=2), a conf_matrix terá a forma:
         [[TN, FP],
          [FN, TP]]
      e calculamos Accuracy, Precision, Recall, F1, IoU, Dice.

    - Para multiclasse (num_classes > 2), a conf_matrix terá forma NxN.
      Para métricas como IoU, Dice etc. podemos usar micro ou macro average.
      Aqui, faremos *exemplo* de macro-averaging.

    Args:
        conf_matrix (np.ndarray): Matriz de confusão NxN.
        num_classes (int): Número de classes.

    Returns:
        dict: {"accuracy", "precision", "recall", "f1_score", "iou", "dice"} (médias macro no caso multiclasse).
    """
    if num_classes == 2:
        # Binário
        TN, FP, FN, TP = conf_matrix.ravel()

        accuracy  = (TP + TN) / (TP + TN + FP + FN + 1e-6)
        precision = TP / (TP + FP + 1e-6)
        recall    = TP / (TP + FN + 1e-6)
        f1_score  = 2 * (precision * recall) / (precision + recall + 1e-6)
        iou       = TP / (TP + FP + FN + 1e-6)
        dice      = 2 * TP / (2 * TP + FP + FN + 1e-6)  # Equivalente a F1 no binário

        return {
            "accuracy":  accuracy,
            "precision": precision,
            "recall":    recall,
            "f1_score":  f1_score,
            "iou":       iou,
            #"dice":      dice
        }

    else:
        # Multiclasse NxN
        # conf_matrix[i, j] = número de pixels da classe i preditos como j
        # i (linha) = rótulo verdadeiro, j (coluna) = rótulo predito
        # Exemplo de macro-averaging:

        # Acurácia (micro-avg)
        total_samples = conf_matrix.sum()
        correct = 0
        for c in range(num_classes):
            correct += conf_matrix[c, c]
        accuracy = correct / (total_samples + 1e-6)

        # Precisão, Recall, F1 por classe
        precision_per_class = []
        recall_per_class    = []
        f1_per_class        = []
        iou_per_class       = []
        dice_per_class      = []

        for c in range(num_classes):
            TP = conf_matrix[c, c]
            FP = conf_matrix[:, c].sum() - TP
            FN = conf_matrix[c, :].sum() - TP
            # TN seria o resto, mas para macro métricas não precisamos explicitamente

            prec = TP / (TP + FP + 1e-6)
            rec  = TP / (TP + FN + 1e-6)
            f1   = 2 * (prec * rec) / (prec + rec + 1e-6)
            iou  = TP / (TP + FP + FN + 1e-6)
            dice = 2 * TP / (2 * TP + FP + FN + 1e-6)

            precision_per_class.append(prec)
            recall_per_class.append(rec)
            f1_per_class.append(f1)
            iou_per_class.append(iou)
            dice_per_class.append(dice)

        # macro average
        precision = np.mean(precision_per_class)
        recall    = np.mean(recall_per_class)
        f1_score  = np.mean(f1_per_class)
        iou       = np.mean(iou_per_class)
        dice      = np.mean(dice_per_class)

        return {
            "accuracy":  accuracy,
            "precision": precision,
            "recall":    recall,
            "f1_score":  f1_score,
            "iou":       iou,
            "dice":      dice
        }


import numpy as np
from sklearn.model_selection import KFold

def cross_validate(
    model_class,        # Classe ou callable para instanciar seu modelo (ex: UNet, MyModel, etc.)
    images_dir,
    masks_dir,
    all_files,
    k=5,                # Número de folds
    num_epochs=10,
    lr=1e-4,
    device="cuda",
    batch_size=16,
    checkpoint_dir_base="checkpoints_fold",  
    metric_to_monitor="val_iou",  
    mode="max",  
    patience=3,
    min_delta=1e-4,
    use_early_stopping=True
):
    """
    Função para rodar cross-validation usando a train_model definida.
    
    Args:
        model_class (callable): Classe ou função que retorna a instância do modelo.
        images_dir (str): Caminho para as imagens.
        masks_dir (str): Caminho para as máscaras.
        all_files (list): Lista com o nome de todos os arquivos de imagem.
        k (int): Quantidade de folds (partições) para k-fold.
        num_epochs (int): Número de épocas de treinamento em cada fold.
        lr (float): Taxa de aprendizado (learning rate).
        device (str): 'cuda' ou 'cpu'.
        batch_size (int): Tamanho do batch.
        checkpoint_dir_base (str): Diretório (base) para salvar checkpoints de cada fold.
        metric_to_monitor (str): Métrica de monitoramento (ex: 'val_iou', 'val_loss').
        mode (str): 'max' se a métrica deve ser maximizada, 'min' se minimizada.
        patience (int): Paciência para early stopping.
        min_delta (float): Variação mínima para considerar melhora.
        use_early_stopping (bool): Se True, usa early stopping em cada fold.

    Returns:
        dict: Dicionário com as métricas médias de todos os folds.
    """
    
    # Instancia o objeto de k-fold do scikit-learn
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Aqui armazenaremos as métricas finais de cada fold (última época do treinamento)
    all_metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_iou": [],
        "val_iou": []
    }

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_files)):
        print(f"\n===== FOLD {fold_idx+1}/{k} =====")

        # Seleciona os arquivos para treino/validação neste fold
        train_files = [all_files[i] for i in train_idx]
        val_files   = [all_files[i] for i in val_idx]

        # Cria datasets
        train_dataset = CloudDataset(images_dir, masks_dir, file_list=train_files)
        val_dataset   = CloudDataset(images_dir, masks_dir, file_list=val_files)

        # Cria DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

        # Instancia um novo modelo
        model = model_class()  # Exemplo: model_class = UNet

        # Define um diretório de checkpoint específico para este fold
        fold_checkpoint_dir = f"{checkpoint_dir_base}_{fold_idx+1}"
        
        # Chama a função de treinamento para este fold
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            lr=lr,
            device=device,
            checkpoint_dir=fold_checkpoint_dir,     # para salvar checkpoints separados por fold
            resume_checkpoint=None,                 # não faz sentido retomar em cross-val
            save_best=True,
            metric_to_monitor=metric_to_monitor,
            mode=mode,
            patience=patience,
            min_delta=min_delta,
            use_early_stopping=use_early_stopping
        )

        # Pega as métricas da última época e guarda em all_metrics
        for key in all_metrics:
            all_metrics[key].append(history[key][-1])

    # Ao final dos k folds, calculamos a média de cada métrica
    avg_metrics = {key: np.mean(vals) for key, vals in all_metrics.items()}

    print("\n===== RESULTADOS FINAIS (K-FOLD) =====")
    for key, value in avg_metrics.items():
        print(f"{key}: {value:.4f}")

    return avg_metrics
