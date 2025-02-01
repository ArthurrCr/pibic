import os
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
from utils.early_stopping import EarlyStopping
from utils.losses import BCE_Dice_Loss, FocalDiceLoss, TverskyBCE
from torchmetrics.classification import BinaryAccuracy, JaccardIndex

def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=10,
    lr=1e-4,
    device="cuda",
    checkpoint_dir="checkpoints",
    resume_checkpoint=None,
    save_best=True,
    metric_to_monitor="val_iou",
    mode="max",
    patience=3,
    min_delta=1e-4,
    use_early_stopping=True,
    loss = 'bce_dice'
):
    """
    Função de treinamento principal.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    if loss == 'bce_dice':
        criterion = BCE_Dice_Loss(bce_weight=0.5)
    elif loss == 'focal_dice':
        criterion = FocalDiceLoss(alpha=0.8, gamma=2)
    elif loss == 'tversky':
        criterion = TverskyBCE(alpha=0.7, beta=0.3)
        
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Scheduler de exemplo (ReduceLROnPlateau)
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

    # Retomar de checkpoint?
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

    # Métricas (binárias) com TorchMetrics
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

    # Loop de épocas
    for epoch in range(start_epoch, num_epochs):
        # ========== TREINO ==========
        model.train()
        train_loss = 0.0
        
        train_acc_metric.reset()
        train_iou_metric.reset()

        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] (Train)")
        for images, masks in train_loop:
            images = images.to(device, non_blocking=True)
            masks  = masks.float().to(device, non_blocking=True)

            optimizer.zero_grad()
            
            # Mixed Precision
            if device.type == 'cuda':
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                    if outputs.shape != masks.shape:
                        raise ValueError("As saídas do modelo e as máscaras devem ter a mesma forma.")
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

        # Resumo da época
        print(
            f"\nEpoch [{epoch+1}/{num_epochs}] Summary:\n"
            f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n"
            f"  Train Acc:  {train_acc:.4f} | Val Acc:  {val_acc:.4f}\n"
            f"  Train IoU:  {train_iou:.4f} | Val IoU:  {val_iou:.4f}\n"
        )

        # Atualiza scheduler com a métrica escolhida
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
