import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm.notebook import tqdm
import torch.amp
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
from torch.optim.lr_scheduler import ReduceLROnPlateau

class EarlyStopping:
    """
    Exemplo simples de EarlyStopping para treino multi-classes.
    """
    def __init__(self, patience=3, mode="max", min_delta=1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = None
        self.early_stop = False

    def __call__(self, current_metric):
        if self.best_metric is None:
            self.best_metric = current_metric
        else:
            improvement = (current_metric - self.best_metric) if self.mode == "max" else (self.best_metric - current_metric)
            if improvement < self.min_delta:
                self.counter += 1
            else:
                self.best_metric = current_metric
                self.counter = 0

        if self.counter >= self.patience:
            self.early_stop = True



def train_model_multiclass(
    model,
    train_loader,
    val_loader,
    num_epochs=10,
    lr=1e-4,
    device="cuda",
    checkpoint_dir=None,
    resume_checkpoint=None,
    save_best=True,
    metric_to_monitor="val_loss",
    mode="min",
    patience=3,
    min_delta=1e-4,
    use_early_stopping=True,
):
    """
    Função de treinamento para segmentação com 4 classes (multi-classes).
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    os.makedirs(checkpoint_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=0.1, patience=3, verbose=True)

    scaler = torch.amp.GradScaler()

    early_stopping = EarlyStopping(patience=patience, mode=mode, min_delta=min_delta) if use_early_stopping else None
    best_metric = -np.inf if mode == "max" else np.inf
    best_epoch = -1
    start_epoch = 0

    # Histórico de métricas
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_iou": [],
        "val_iou": []
    }

    # Retomar checkpoint, se especificado
    if resume_checkpoint is not None and os.path.isfile(resume_checkpoint):
        print(f"Carregando checkpoint '{resume_checkpoint}'")
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_metric = checkpoint.get('best_metric', best_metric)
        best_epoch = checkpoint.get('best_epoch', best_epoch)
        history = checkpoint.get('history', history)  # Recupera o histórico salvo, se existir
        print(f"Retomando da época {start_epoch} com best_metric={best_metric:.4f}")

    # Métricas multi-classe: 4 classes
    train_acc_metric = MulticlassAccuracy(num_classes=4, average='macro').to(device)
    train_iou_metric = MulticlassJaccardIndex(num_classes=4, average='macro').to(device)
    val_acc_metric   = MulticlassAccuracy(num_classes=4, average='macro').to(device)
    val_iou_metric   = MulticlassJaccardIndex(num_classes=4, average='macro').to(device)

    use_amp = (device.type == 'cuda')

    for epoch in range(start_epoch, num_epochs):
        # ======================= TREINO =======================
        model.train()
        epoch_train_loss = 0.0
        train_acc_metric.reset()
        train_iou_metric.reset()

        loop_train = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] (Train)")

        for images, masks in loop_train:
            images = images.to(device, non_blocking=True)
            if masks.dim() == 4:
                if masks.shape[1] == 1:
                    masks = masks.squeeze(1)
                elif masks.shape[-1] == 1:
                    masks = masks.squeeze(-1)
            masks = masks.long().to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)  # (B, 4, H, W)

            # Calcular a perda
            loss_value = criterion(outputs, masks)

            if use_amp:
                scaler.scale(loss_value).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_value.backward()
                optimizer.step()

            epoch_train_loss += loss_value.item()

            # Predição: argmax nas 4 classes
            preds = torch.argmax(outputs, dim=1)
            # Atualiza métricas
            train_acc_metric.update(preds, masks)
            train_iou_metric.update(preds, masks)

            loop_train.set_postfix(loss=f"{loss_value.item():.4f}")

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_acc = train_acc_metric.compute().item()
        train_iou = train_iou_metric.compute().item()

        # ======================= VALIDAÇÃO =======================
        model.eval()
        epoch_val_loss = 0.0
        val_acc_metric.reset()
        val_iou_metric.reset()

        loop_val = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] (Val)")
        with torch.no_grad():
            for images, masks in loop_val:
                images = images.to(device, non_blocking=True)
                if masks.dim() == 4:
                    if masks.shape[1] == 1:
                        masks = masks.squeeze(1)
                    elif masks.shape[-1] == 1:
                        masks = masks.squeeze(-1)
                masks = masks.long().to(device, non_blocking=True)

                # Forward pass
                outputs = model(images)  # (B, 4, H, W)
                loss_value = criterion(outputs, masks)
                epoch_val_loss += loss_value.item()

                preds = torch.argmax(outputs, dim=1)
                val_acc_metric.update(preds, masks)
                val_iou_metric.update(preds, masks)

                loop_val.set_postfix(loss=f"{loss_value.item():.4f}")

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_acc = val_acc_metric.compute().item()
        val_iou = val_iou_metric.compute().item()

        # Armazena histórico
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)

        print(
            f"\nEpoch [{epoch+1}/{num_epochs}] Summary:\n"
            f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n"
            f"  Train Acc:  {train_acc:.4f} | Val Acc:  {val_acc:.4f}\n"
            f"  Train IoU:  {train_iou:.4f} | Val IoU:  {val_iou:.4f}\n"
        )

        # Define qual métrica usar no scheduler e early stopping
        if metric_to_monitor == "val_loss":
            current_metric = avg_val_loss
        elif metric_to_monitor == "val_iou":
            current_metric = val_iou
        elif metric_to_monitor == "val_acc":
            current_metric = val_acc
        else:
            current_metric = history[metric_to_monitor][-1]

        scheduler.step(current_metric)

        # Verifica melhora
        is_better = False
        if mode == "max" and current_metric > best_metric:
            is_better = True
        elif mode == "min" and current_metric < best_metric:
            is_better = True

        if is_better:
            best_metric = current_metric
            best_epoch = epoch + 1
            if save_best:
                best_model_path = os.path.join(checkpoint_dir, "best_model_mc.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_metric': best_metric,
                    'best_epoch': best_epoch,
                    'history': history
                }, best_model_path)
                print(f"[Melhoria] Modelo salvo com {metric_to_monitor}: {best_metric:.4f} na época {epoch+1}.")

        # Salva checkpoint atual em um único arquivo fixo
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_metric': best_metric,
            'best_epoch': best_epoch,
            'history': history
        }, checkpoint_path)
        print(f"Checkpoint salvo em '{checkpoint_path}'.")

        # Early Stopping
        if use_early_stopping:
            early_stopping(current_metric)
            if early_stopping.early_stop:
                print(f"Early Stopping disparado na época {epoch+1}. Melhor {metric_to_monitor}: {early_stopping.best_metric:.4f}")
                break

        # Limpa cache de GPU, se estiver usando AMP
        if use_amp and torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Treinamento finalizado. Melhor {metric_to_monitor}: {best_metric:.4f} na época {best_epoch}.")
    return history

