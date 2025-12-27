"""Experiment runner for cloud segmentation."""

import os
import pickle
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

import pandas as pd
import torch
import torch.nn as nn

from utils.constants import CLASS_NAMES
from utils.dataset import create_dataloaders
from utils.losses import get_loss
from utils.metrics import compute_metrics, evaluate_model
from utils.boa_metrics import evaluate_test_dataset
from utils.training import train_model


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    
    name: str
    model_class: Type[nn.Module]
    encoder_name: str = "tu-regnetz_d8"
    encoder_weights: Optional[str] = None
    in_channels: int = 13
    num_classes: int = 4
    batch_size: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 50
    patience: int = 10
    loss_name: str = "ce"
    loss_kwargs: Dict = None
    
    def __post_init__(self):
        if self.loss_kwargs is None:
            self.loss_kwargs = {}


class ExperimentRunner:
    """Runs and manages experiments."""

    def __init__(
        self,
        parts: List[str],
        base_dir: str = "/content/drive/MyDrive/pibic",
        device: str = "cuda",
    ):
        self.parts = parts
        self.base_dir = base_dir
        self.device = device
        self.results: List[Dict] = []

        self.checkpoints_dir = os.path.join(base_dir, "checkpoints")
        self.results_dir = os.path.join(base_dir, "results")
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def run(self, config: ExperimentConfig, resume: bool = True) -> Dict:
        """Run a single experiment."""
        
        print(f"\n{'='*60}")
        print(f"Experiment: {config.name}")
        print(f"Loss: {config.loss_name} | LR: {config.learning_rate} | BS: {config.batch_size}")
        print(f"{'='*60}")

        checkpoint_dir = os.path.join(self.checkpoints_dir, config.name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # DataLoaders
        train_loader, val_loader, test_loader = create_dataloaders(
            self.parts,
            batch_size=config.batch_size,
            normalize=True,
        )

        # Model
        model = config.model_class(
            encoder_name=config.encoder_name,
            encoder_weights=config.encoder_weights,
            in_channels=config.in_channels,
            num_classes=config.num_classes,
        )

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")

        # Loss
        loss_fn = get_loss(config.loss_name, **config.loss_kwargs)
        print(f"Loss function: {loss_fn.__class__.__name__}")

        # Resume
        resume_path = os.path.join(checkpoint_dir, "checkpoint.pth")
        if not resume or not os.path.isfile(resume_path):
            resume_path = None

        # Train
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.num_epochs,
            lr=config.learning_rate,
            device=self.device,
            checkpoint_dir=checkpoint_dir,
            resume_checkpoint=resume_path,
            metric_to_monitor="val_loss",
            mode="min",
            patience=config.patience,
            loss_fn=loss_fn,
        )

        # Load best model
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        checkpoint = torch.load(best_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device).eval()

        # Evaluate
        df_boa = evaluate_test_dataset(test_loader, model, device=self.device, normalize_imgs=False)
        conf_matrix = evaluate_model(test_loader, model, device=self.device, normalize_imgs=False)
        metrics = compute_metrics(conf_matrix)

        # Extract BOA values
        cloud_boa = float(df_boa[df_boa["Experiment"] == "cloud/no cloud"]["Median BOA"].values[0])
        shadow_boa = float(df_boa[df_boa["Experiment"] == "cloud shadow"]["Median BOA"].values[0])
        valid_boa = float(df_boa[df_boa["Experiment"] == "valid/invalid"]["Median BOA"].values[0])

        result = {
            "name": config.name,
            "loss": config.loss_name,
            "lr": config.learning_rate,
            "batch_size": config.batch_size,
            "cloud_boa": cloud_boa,
            "shadow_boa": shadow_boa,
            "valid_boa": valid_boa,
            "accuracy": metrics["Overall"]["Accuracy"],
            "best_epoch": checkpoint.get("best_epoch", -1),
            "n_params": n_params,
            "timestamp": datetime.now().isoformat(),
        }

        self.results.append(result)
        self._save_results()

        print(f"\nResults:")
        print(f"  Cloud BOA:  {cloud_boa:.4f}")
        print(f"  Shadow BOA: {shadow_boa:.4f}")
        print(f"  Valid BOA:  {valid_boa:.4f}")
        print(f"  Accuracy:   {metrics['Overall']['Accuracy']:.4f}")

        return result

    def run_loss_comparison(self, model_class: Type[nn.Module]) -> pd.DataFrame:
        """Etapa 1: Compare loss functions with fixed hyperparameters."""
        
        losses = ["ce", "dice_ce", "focal", "focal_tversky", "dice_focal"]
        
        for loss_name in losses:
            config = ExperimentConfig(
                name=f"loss_comparison_{loss_name}",
                model_class=model_class,
                loss_name=loss_name,
                learning_rate=1e-4,
                batch_size=4,
            )
            self.run(config, resume=True)
        
        return self.get_summary()

    def run_hyperparameter_tuning(
        self,
        model_class: Type[nn.Module],
        best_loss: str,
    ) -> pd.DataFrame:
        """Etapa 2: Tune hyperparameters with best loss."""
        
        learning_rates = [5e-4, 1e-4, 5e-5]
        batch_sizes = [4, 8]
        
        for lr in learning_rates:
            for bs in batch_sizes:
                config = ExperimentConfig(
                    name=f"hp_tuning_{best_loss}_lr{lr}_bs{bs}",
                    model_class=model_class,
                    loss_name=best_loss,
                    learning_rate=lr,
                    batch_size=bs,
                )
                self.run(config, resume=True)
        
        return self.get_summary()

    def get_summary(self) -> pd.DataFrame:
        """Get summary DataFrame of all results."""
        df = pd.DataFrame(self.results)
        if not df.empty:
            df = df.sort_values("cloud_boa", ascending=False)
        return df

    def _save_results(self) -> None:
        """Save results to disk."""
        df = self.get_summary()
        df.to_csv(os.path.join(self.results_dir, "experiments_summary.csv"), index=False)
        
        with open(os.path.join(self.results_dir, "experiments_full.pkl"), "wb") as f:
            pickle.dump(self.results, f)