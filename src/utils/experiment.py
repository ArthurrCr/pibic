"""Experiment runner for cloud segmentation."""

import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

import pandas as pd
import torch
import torch.nn as nn

from utils.constants import CLASS_NAMES
from utils.dataset import create_dataloaders
from utils.losses import get_loss, LOSS_REGISTRY
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
    loss_kwargs: Optional[Dict] = None
    scheduler_factor: float = 0.1
    scheduler_patience: int = 3

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

        # Cache dataloaders
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None

    def _get_dataloaders(self, batch_size: int):
        """Get or create dataloaders."""
        if (
            self._train_loader is None
            or self._train_loader.batch_size != batch_size
        ):
            self._train_loader, self._val_loader, self._test_loader = create_dataloaders(
                self.parts,
                batch_size=batch_size,
                normalize=True,
            )
        return self._train_loader, self._val_loader, self._test_loader
    def print_setup(
        self,
        model_class: Type[nn.Module],
        losses: List[str] = None,
        learning_rates: List[float] = None,
        batch_sizes: List[int] = None,
        scheduler_factors: List[float] = None,
        scheduler_patiences: List[int] = None,
    ) -> None:
        """Print experiment setup information."""

        if losses is None:
            losses = ["ce", "dice_ce", "focal", "focal_tversky", "dice_focal"]
        if learning_rates is None:
            learning_rates = [1e-4]
        if batch_sizes is None:
            batch_sizes = [4]
        if scheduler_factors is None:
            scheduler_factors = [0.1]
        if scheduler_patiences is None:
            scheduler_patiences = [3]

        train_loader, val_loader, test_loader = self._get_dataloaders(batch_sizes[0])

        print("\n" + "=" * 60)
        print("EXPERIMENT SETUP")
        print("=" * 60)

        print("\n[Model]")
        print(f"  Class:           {model_class.__name__}")
        print(f"  Encoder:         tu-regnetz_d8")
        print(f"  Encoder weights: None (13 bands)")
        print(f"  Input channels:  13")
        print(f"  Num classes:     4 (Clear, Thick Cloud, Thin Cloud, Shadow)")

        print("\n[Dataset]")
        print(f"  Train samples:   {len(train_loader.dataset)}")
        print(f"  Val samples:     {len(val_loader.dataset)}")
        print(f"  Test samples:    {len(test_loader.dataset)}")
        print(f"  Normalization:   /10000")

        print("\n[Training]")
        print(f"  Device:          {self.device}")
        print(f"  Epochs:          50")
        print(f"  Early stopping:  patience=10")
        print(f"  Optimizer:       Adam (weight_decay=1e-4)")
        print(f"  Scheduler:       ReduceLROnPlateau")

        print("\n[Hyperparameters to test]")
        print(f"  Learning rates:      {learning_rates}")
        print(f"  Batch sizes:         {batch_sizes}")
        print(f"  Scheduler factors:   {scheduler_factors}")
        print(f"  Scheduler patiences: {scheduler_patiences}")

        print("\n[Losses to test]")
        loss_descriptions = {
            "ce": "CrossEntropy (baseline)",
            "dice": "Dice Loss",
            "focal": "Focal Loss (gamma=2.0, alpha=0.25)",
            "tversky": "Tversky Loss (alpha=0.3, beta=0.7)",
            "focal_tversky": "Focal Tversky (alpha=0.3, beta=0.7, gamma=0.75)",
            "dice_ce": "Dice + CE (0.5/0.5)",
            "dice_focal": "Dice + Focal (0.5/0.5)",
        }
        for i, loss in enumerate(losses, 1):
            desc = loss_descriptions.get(loss, loss)
            print(f"  {i}. {loss:15} -> {desc}")

        print("\n[Output]")
        print(f"  Checkpoints:     {self.checkpoints_dir}")
        print(f"  Results:         {self.results_dir}")

        total = len(losses) * len(learning_rates) * len(batch_sizes)
        print("\n" + "-" * 60)
        print(f"Total experiments: {total}")
        print("=" * 60 + "\n")

    def run(self, config: ExperimentConfig, resume: bool = True) -> Dict:
        """Run a single experiment."""

        print(f"\n{'='*60}")
        print(f"RUNNING: {config.name}")
        print(f"{'='*60}")
        print(f"  Loss:             {config.loss_name}")
        print(f"  LR:               {config.learning_rate}")
        print(f"  Batch size:       {config.batch_size}")
        print(f"  Scheduler factor: {config.scheduler_factor}")
        print(f"  Scheduler patience: {config.scheduler_patience}")
        print("-" * 60)

        checkpoint_dir = os.path.join(self.checkpoints_dir, config.name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # DataLoaders
        train_loader, val_loader, test_loader = self._get_dataloaders(config.batch_size)

        # Model
        model = config.model_class(
            encoder_name=config.encoder_name,
            encoder_weights=config.encoder_weights,
            in_channels=config.in_channels,
            num_classes=config.num_classes,
        )

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        # Loss
        loss_fn = get_loss(config.loss_name, **config.loss_kwargs)
        print(f"  Loss fn:    {loss_fn.__class__.__name__}")

        # Resume
        resume_path = os.path.join(checkpoint_dir, "checkpoint.pth")
        if not resume or not os.path.isfile(resume_path):
            resume_path = None
            print(f"  Resume:     Starting from scratch")
        else:
            print(f"  Resume:     From checkpoint")

        print("-" * 60)

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
            scheduler_factor=config.scheduler_factor,
            scheduler_patience=config.scheduler_patience,
        )

        # Load best model
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        checkpoint = torch.load(best_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device).eval()

        # Evaluate
        print("\nEvaluating on test set...")
        df_boa = evaluate_test_dataset(
            test_loader, model, device=self.device, normalize_imgs=False
        )
        conf_matrix = evaluate_model(
            test_loader, model, device=self.device, normalize_imgs=False
        )
        metrics = compute_metrics(conf_matrix)

        # Extract BOA values
        cloud_boa = float(
            df_boa[df_boa["Experiment"] == "cloud/no cloud"]["Median BOA"].values[0]
        )
        shadow_boa = float(
            df_boa[df_boa["Experiment"] == "cloud shadow"]["Median BOA"].values[0]
        )
        valid_boa = float(
            df_boa[df_boa["Experiment"] == "valid/invalid"]["Median BOA"].values[0]
        )

        result = {
            "name": config.name,
            "loss": config.loss_name,
            "lr": config.learning_rate,
            "batch_size": config.batch_size,
            "sched_factor": config.scheduler_factor,
            "sched_patience": config.scheduler_patience,
            "cloud_boa": cloud_boa,
            "shadow_boa": shadow_boa,
            "valid_boa": valid_boa,
            "accuracy": metrics["Overall"]["Accuracy"],
            "best_epoch": checkpoint.get("best_epoch", -1),
            "best_val_loss": checkpoint.get("best_metric", -1),
            "n_params": n_params,
            "timestamp": datetime.now().isoformat(),
        }

        self.results.append(result)
        self._save_results()

        print(f"\n{'='*60}")
        print(f"RESULTS: {config.name}")
        print(f"{'='*60}")
        print(f"  Cloud BOA:     {cloud_boa:.4f}")
        print(f"  Shadow BOA:    {shadow_boa:.4f}")
        print(f"  Valid BOA:     {valid_boa:.4f}")
        print(f"  Accuracy:      {metrics['Overall']['Accuracy']:.4f}")
        print(f"  Best epoch:    {result['best_epoch']}")
        print(f"{'='*60}\n")

        return result

    def run_loss_comparison(
        self,
        model_class: Type[nn.Module],
        losses: List[str] = None,
    ) -> pd.DataFrame:
        """Stage 1: Compare loss functions with fixed hyperparameters."""

        if losses is None:
            losses = ["ce", "dice_ce", "focal", "focal_tversky", "dice_focal"]

        self.print_setup(model_class, losses=losses)

        print("\n" + "#" * 60)
        print("STAGE 1: LOSS FUNCTION COMPARISON")
        print("#" * 60)

        for i, loss_name in enumerate(losses, 1):
            print(f"\n[{i}/{len(losses)}] Testing: {loss_name}")

            config = ExperimentConfig(
                name=f"loss_{loss_name}",
                model_class=model_class,
                loss_name=loss_name,
                learning_rate=1e-4,
                batch_size=4,
            )
            self.run(config, resume=True)

        print("\n" + "#" * 60)
        print("STAGE 1 COMPLETE")
        print("#" * 60)

        return self.get_summary()

    def run_hyperparameter_tuning(
        self,
        model_class: Type[nn.Module],
        best_loss: str,
        learning_rates: List[float] = None,
        batch_sizes: List[int] = None,
    ) -> pd.DataFrame:
        """Stage 2: Tune hyperparameters with best loss."""

        if learning_rates is None:
            learning_rates = [5e-4, 1e-4, 5e-5]
        if batch_sizes is None:
            batch_sizes = [4, 8]

        self.print_setup(
            model_class,
            losses=[best_loss],
            learning_rates=learning_rates,
            batch_sizes=batch_sizes,
        )

        print("\n" + "#" * 60)
        print(f"STAGE 2: HYPERPARAMETER TUNING (loss={best_loss})")
        print("#" * 60)

        total = len(learning_rates) * len(batch_sizes)
        current = 0

        for lr in learning_rates:
            for bs in batch_sizes:
                current += 1
                print(f"\n[{current}/{total}] Testing: lr={lr}, batch_size={bs}")

                config = ExperimentConfig(
                    name=f"hp_{best_loss}_lr{lr}_bs{bs}",
                    model_class=model_class,
                    loss_name=best_loss,
                    learning_rate=lr,
                    batch_size=bs,
                )
                self.run(config, resume=True)

        print("\n" + "#" * 60)
        print("STAGE 2 COMPLETE")
        print("#" * 60)

        return self.get_summary()
    
    def run_scheduler_tuning(
        self,
        model_class: Type[nn.Module],
        best_loss: str,
        best_lr: float,
        best_batch_size: int,
        factors: List[float] = None,
        patiences: List[int] = None,
    ) -> pd.DataFrame:
        """Stage 3: Tune ReduceLROnPlateau hyperparameters."""

        if factors is None:
            factors = [0.1, 0.5]
        if patiences is None:
            patiences = [2, 3, 5]

        print("\n" + "#" * 60)
        print("STAGE 3: SCHEDULER TUNING (ReduceLROnPlateau)")
        print(f"  Loss: {best_loss}, LR: {best_lr}, BS: {best_batch_size}")
        print(f"  Factors:   {factors}")
        print(f"  Patiences: {patiences}")
        print("#" * 60)

        total = len(factors) * len(patiences)
        current = 0

        for factor in factors:
            for sched_patience in patiences:
                current += 1
                print(f"\n[{current}/{total}] Testing: factor={factor}, patience={sched_patience}")

                config = ExperimentConfig(
                    name=f"stage3_sched_f{factor}_p{sched_patience}",
                    model_class=model_class,
                    loss_name=best_loss,
                    learning_rate=best_lr,
                    batch_size=best_batch_size,
                    scheduler_factor=factor,
                    scheduler_patience=sched_patience,
                )
                self.run(config, resume=True)

        print("\n" + "#" * 60)
        print("STAGE 3 COMPLETE")
        print("#" * 60)
        self.print_summary()

        return self.get_summary()

    def get_summary(self) -> pd.DataFrame:
        """Get summary DataFrame of all results."""
        df = pd.DataFrame(self.results)
        if not df.empty:
            df = df.sort_values("cloud_boa", ascending=False)
        return df

    def print_summary(self) -> None:
        """Print formatted summary of results."""
        df = self.get_summary()

        if df.empty:
            print("No results yet.")
            return

        print("\n" + "=" * 80)
        print("EXPERIMENT RESULTS (sorted by Cloud BOA)")
        print("=" * 80)

        cols = ["name", "loss", "lr", "batch_size", "cloud_boa", "shadow_boa", "valid_boa", "accuracy", "best_epoch"]
        print(df[cols].to_string(index=False))

        print("\n" + "-" * 80)
        best = df.iloc[0]
        print(f"BEST: {best['name']}")
        print(f"  Loss: {best['loss']}, LR: {best['lr']}, BS: {best['batch_size']}")
        print(f"  Cloud BOA: {best['cloud_boa']:.4f}, Accuracy: {best['accuracy']:.4f}")
        print("=" * 80 + "\n")

    def _save_results(self) -> None:
        """Save results to disk."""
        df = self.get_summary()
        df.to_csv(
            os.path.join(self.results_dir, "experiments_summary.csv"), index=False
        )

        with open(os.path.join(self.results_dir, "experiments_full.pkl"), "wb") as f:
            pickle.dump(self.results, f)