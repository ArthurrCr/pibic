"""Experiment runner for cloud segmentation models."""

import os
import pickle
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type

import pandas as pd
import torch
import torch.nn as nn

from utils.constants import CLASS_NAMES
from utils.dataset import create_dataloaders
from utils.metrics import compute_metrics, evaluate_model
from utils.boa_metrics import evaluate_test_dataset
from utils.training import train_model


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    
    name: str
    model_class: Type[nn.Module]
    encoder_name: str
    encoder_weights: Optional[str] = None
    in_channels: int = 13
    num_classes: int = 4
    batch_size: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 50
    patience: int = 10
    metric_to_monitor: str = "val_loss"
    mode: str = "min"
    loss_fn: Optional[Callable] = None
    loss_name: str = "CrossEntropyLoss"


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    
    config: Dict
    boa_metrics: pd.DataFrame
    confusion_matrix: Any
    per_class_metrics: Dict
    overall_accuracy: float
    best_epoch: int
    best_metric: float
    history: Dict
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ExperimentRunner:
    """Runs and manages experiments."""

    def __init__(
        self,
        parts: List[str],
        base_dir: str = "/content/drive/MyDrive/pibic",
        device: str = "cuda",
    ):
        """
        Initialize the experiment runner.

        Args:
            parts: Dataset parts from download_cloudsen12.
            base_dir: Base directory for saving checkpoints and results.
            device: Device for training and evaluation.
        """
        self.parts = parts
        self.base_dir = base_dir
        self.device = device
        self.results: List[ExperimentResult] = []

        self.checkpoints_dir = os.path.join(base_dir, "checkpoints")
        self.results_dir = os.path.join(base_dir, "results")
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def run(
        self,
        config: ExperimentConfig,
        resume: bool = True,
    ) -> ExperimentResult:
        """
        Run a single experiment.

        Args:
            config: Experiment configuration.
            resume: If True, resumes from checkpoint if available.

        Returns:
            ExperimentResult with all metrics.
        """
        print(f"\n{'='*60}")
        print(f"Experiment: {config.name}")
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

        # Resume checkpoint
        resume_path = os.path.join(checkpoint_dir, "checkpoint.pth")
        if not resume or not os.path.isfile(resume_path):
            resume_path = None
            print("Starting from scratch")
        else:
            print(f"Resuming from: {resume_path}")

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
            metric_to_monitor=config.metric_to_monitor,
            mode=config.mode,
            patience=config.patience,
            loss_fn=config.loss_fn,
        )

        # Load best model
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        checkpoint = torch.load(best_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device).eval()

        # Evaluate
        df_boa = evaluate_test_dataset(
            test_loader, model, device=self.device, normalize_imgs=False
        )
        conf_matrix = evaluate_model(
            test_loader, model, device=self.device, normalize_imgs=False
        )
        metrics = compute_metrics(conf_matrix)

        # Build result
        result = ExperimentResult(
            config=asdict(config) | {"n_parameters": n_params},
            boa_metrics=df_boa,
            confusion_matrix=conf_matrix,
            per_class_metrics={c: metrics[c] for c in CLASS_NAMES},
            overall_accuracy=metrics["Overall"]["Accuracy"],
            best_epoch=checkpoint.get("best_epoch", -1),
            best_metric=checkpoint.get("best_metric", 0.0),
            history=history,
        )

        self.results.append(result)
        self._save_result(result)

        print(f"\nBOA Metrics:")
        print(df_boa.to_string(index=False))
        print(f"\nOverall Accuracy: {result.overall_accuracy:.4f}")

        return result

    def run_grid(
        self,
        configs: List[ExperimentConfig],
        resume: bool = True,
    ) -> pd.DataFrame:
        """
        Run multiple experiments.

        Args:
            configs: List of experiment configurations.
            resume: If True, resumes from checkpoints.

        Returns:
            DataFrame summarizing all results.
        """
        for config in configs:
            try:
                self.run(config, resume=resume)
            except Exception as e:
                print(f"Error in {config.name}: {e}")
                continue

        return self.get_summary()

    def get_summary(self) -> pd.DataFrame:
        """Get summary DataFrame of all results."""
        rows = []
        for r in self.results:
            cloud_boa = float(
                r.boa_metrics[r.boa_metrics["Experiment"] == "cloud/no cloud"]["Median BOA"].values[0]
            )
            shadow_boa = float(
                r.boa_metrics[r.boa_metrics["Experiment"] == "cloud shadow"]["Median BOA"].values[0]
            )
            valid_boa = float(
                r.boa_metrics[r.boa_metrics["Experiment"] == "valid/invalid"]["Median BOA"].values[0]
            )

            rows.append({
                "name": r.config["name"],
                "encoder": r.config["encoder_name"],
                "loss": r.config["loss_name"],
                "lr": r.config["learning_rate"],
                "batch_size": r.config["batch_size"],
                "cloud_boa": cloud_boa,
                "shadow_boa": shadow_boa,
                "valid_boa": valid_boa,
                "accuracy": r.overall_accuracy,
                "best_epoch": r.best_epoch,
                "n_params": r.config["n_parameters"],
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("cloud_boa", ascending=False)
        return df

    def _save_result(self, result: ExperimentResult) -> None:
        """Save result to disk."""
        name = result.config["name"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.results_dir, f"{name}_{timestamp}.pkl")

        with open(path, "wb") as f:
            pickle.dump(result, f)

        # Also save summary CSV
        summary = self.get_summary()
        summary.to_csv(os.path.join(self.results_dir, "summary.csv"), index=False)