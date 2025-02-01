import numpy as np
from sklearn.model_selection import KFold
from utils.train import train_model
from utils.data_loader import CloudDataset
from torch.utils.data import DataLoader

def cross_validate(
    model_class,
    images_dir,
    masks_dir,
    all_files,
    k=5,
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
    Função para rodar cross-validation usando a train_model.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

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

        train_files = [all_files[i] for i in train_idx]
        val_files   = [all_files[i] for i in val_idx]

        # Datasets
        train_dataset = CloudDataset(images_dir, masks_dir, file_list=train_files)
        val_dataset   = CloudDataset(images_dir, masks_dir, file_list=val_files)

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

        # Instancia um novo modelo
        model = model_class()

        # Diretório de checkpoint específico para este fold
        fold_checkpoint_dir = f"{checkpoint_dir_base}_{fold_idx+1}"
        
        # Treinar
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            lr=lr,
            device=device,
            checkpoint_dir=fold_checkpoint_dir,
            resume_checkpoint=None,
            save_best=True,
            metric_to_monitor=metric_to_monitor,
            mode=mode,
            patience=patience,
            min_delta=min_delta,
            use_early_stopping=use_early_stopping
        )

        # Salva métricas da última época
        for key in all_metrics:
            all_metrics[key].append(history[key][-1])

    # Média das métricas após k folds
    avg_metrics = {key: np.mean(vals) for key, vals in all_metrics.items()}

    print("\n===== RESULTADOS FINAIS (K-FOLD) =====")
    for key, value in avg_metrics.items():
        print(f"{key}: {value:.4f}")

    return avg_metrics
