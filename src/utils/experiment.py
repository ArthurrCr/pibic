"""Experiment runner for cloud segmentation.

This module provides a systematic framework for running and comparing
machine learning experiments for cloud segmentation, following best
practices for reproducibility and scientific rigor.

Methodology:
- Training set: optimize model weights
- Validation set: hyperparameter selection, early stopping, model selection
- Test set: final evaluation only (reported in paper)
"""

import os
import pickle
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils.constants import CLASS_NAMES
from utils.dataset import create_dataloaders
from utils.losses import get_loss
from utils.metrics import compute_metrics, evaluate_model
from utils.boa_metrics import evaluate_test_dataset
from utils.training import train_model


def set_seed(seed: int = 42) -> None:
    """Set seed for reproducibility across all random number generators.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment.
    
    Attributes:
        name: Unique identifier for the experiment.
        model_class: PyTorch model class to instantiate.
        encoder_name: Name of the encoder architecture.
        encoder_weights: Pretrained weights to use.
        in_channels: Number of input channels (13 for Sentinel-2).
        num_classes: Number of output classes (4 for CloudSEN12).
        batch_size: Training batch size.
        learning_rate: Initial learning rate.
        num_epochs: Maximum number of training epochs.
        patience: Early stopping patience.
        loss_name: Name of loss function from LOSS_REGISTRY.
        loss_kwargs: Additional arguments for loss function.
        scheduler_factor: Factor for ReduceLROnPlateau.
        scheduler_patience: Patience for ReduceLROnPlateau.
        seed: Random seed for reproducibility.
    """

    name: str
    model_class: Type[nn.Module]
    encoder_name: str = "tu-regnetz_d8"
    encoder_weights: Optional[str] = "imagenet"
    in_channels: int = 13
    num_classes: int = 4
    batch_size: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 50
    patience: int = 5
    loss_name: str = "ce"
    loss_kwargs: Dict = field(default_factory=dict)
    scheduler_factor: float = 0.1
    scheduler_patience: int = 3
    seed: int = 42


class ExperimentRunner:
    """Runs and manages machine learning experiments.
    
    This class provides methods for systematic experimentation including:
    - Stage 1: Loss function comparison
    - Stage 2: Hyperparameter tuning
    - Stage 3: Scheduler tuning
    
    Results are automatically saved and can be loaded for continuation.
    
    IMPORTANT: Model selection is based on VALIDATION metrics.
    Test metrics are computed for final reporting only.
    
    Attributes:
        parts: Dataset parts for tacoreader.
        base_dir: Base directory for checkpoints and results.
        device: PyTorch device for training.
        seed: Global random seed.
        results: List of experiment result dictionaries.
    """

    def __init__(
        self,
        parts: List[str],
        base_dir: str = "/content/drive/MyDrive/pibic",
        device: str = "cuda",
        seed: int = 42,
    ):
        """Initialize the experiment runner.
        
        Args:
            parts: Dataset parts for tacoreader.
            base_dir: Base directory for saving results.
            device: Device for training ("cuda" or "cpu").
            seed: Random seed for reproducibility.
        """
        self.parts = parts
        self.base_dir = base_dir
        self.device = device
        self.seed = seed
        self.results: List[Dict] = []

        self.checkpoints_dir = os.path.join(base_dir, "checkpoints")
        self.results_dir = os.path.join(base_dir, "results")
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        self._train_loader = None
        self._val_loader = None
        self._test_loader = None
        self._current_batch_size = None

    def _get_dataloaders(self, batch_size: int):
        """Get or create dataloaders with caching."""
        if self._train_loader is None or self._current_batch_size != batch_size:
            self._train_loader, self._val_loader, self._test_loader = create_dataloaders(
                self.parts,
                batch_size=batch_size,
                normalize=True,
                seed=self.seed,
            )
            self._current_batch_size = batch_size
        return self._train_loader, self._val_loader, self._test_loader

    def load_results(self, path: Optional[str] = None) -> int:
        """Load previous results from CSV file.
        
        Args:
            path: Path to CSV file. If None, uses default location.
            
        Returns:
            Number of results loaded.
        """
        if path is None:
            path = os.path.join(self.results_dir, "experiments_summary.csv")
        
        if os.path.exists(path):
            df = pd.read_csv(path)
            self.results = df.to_dict("records")
            print(f"Loaded {len(self.results)} previous results from {path}")
            return len(self.results)
        
        print(f"No previous results found at {path}")
        return 0

    def get_completed_experiments(self) -> set:
        """Get set of completed experiment names."""
        return {r["name"] for r in self.results}

    def is_completed(self, name: str) -> bool:
        """Check if an experiment has already been completed."""
        return name in self.get_completed_experiments()

    def print_setup(
        self,
        model_class: Type[nn.Module],
        losses: Optional[List[str]] = None,
        learning_rates: Optional[List[float]] = None,
        batch_sizes: Optional[List[int]] = None,
        scheduler_factors: Optional[List[float]] = None,
        scheduler_patiences: Optional[List[int]] = None,
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

        print("\n" + "=" * 70)
        print("EXPERIMENT SETUP")
        print("=" * 70)

        print("\n[Model]")
        print(f"  Class:           {model_class.__name__}")
        print(f"  Encoder:         tu-regnetz_d8")
        print(f"  Encoder weights: imagenet")
        print(f"  Input channels:  13 (Sentinel-2 bands)")
        print(f"  Num classes:     4 (Clear, Thick Cloud, Thin Cloud, Shadow)")

        print("\n[Dataset - CloudSEN12]")
        print(f"  Train samples:   {len(train_loader.dataset)}")
        print(f"  Val samples:     {len(val_loader.dataset)}")
        print(f"  Test samples:    {len(test_loader.dataset)}")
        print(f"  Normalization:   /10000")

        print("\n[Training]")
        print(f"  Device:          {self.device}")
        print(f"  Max epochs:      50")
        print(f"  Early stopping:  patience=5, monitor=val_loss")
        print(f"  Optimizer:       Adam (weight_decay=1e-4)")
        print(f"  Scheduler:       ReduceLROnPlateau")
        print(f"  Seed:            {self.seed}")

        print("\n[Evaluation Strategy]")
        print("  Selection:       Based on VALIDATION BOA metrics")
        print("  Test metrics:    Computed for final reporting only")

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
        print("\n" + "-" * 70)
        print(f"Total experiments planned: {total}")
        print("=" * 70 + "\n")

    def _extract_boa_metrics(self, df_boa: pd.DataFrame, prefix: str = "") -> Dict:
        """Extract BOA metrics from evaluation DataFrame.
        
        Args:
            df_boa: DataFrame from evaluate_test_dataset.
            prefix: Prefix for metric names (e.g., "val_" or "test_").
            
        Returns:
            Dictionary with BOA metrics.
        """
        cloud_boa = float(
            df_boa[df_boa["Experiment"] == "cloud/no cloud"]["Median BOA"].values[0]
        )
        shadow_boa = float(
            df_boa[df_boa["Experiment"] == "cloud shadow"]["Median BOA"].values[0]
        )
        valid_boa = float(
            df_boa[df_boa["Experiment"] == "valid/invalid"]["Median BOA"].values[0]
        )
        mean_boa = (cloud_boa + shadow_boa) / 2
        
        return {
            f"{prefix}cloud_boa": cloud_boa,
            f"{prefix}shadow_boa": shadow_boa,
            f"{prefix}valid_boa": valid_boa,
            f"{prefix}mean_boa": mean_boa,
        }

    def run(self, config: ExperimentConfig, resume: bool = True) -> Optional[Dict]:
        """Run a single experiment.
        
        Trains model, then evaluates on BOTH validation and test sets.
        Validation metrics are used for model selection.
        Test metrics are for final reporting only.
        
        Args:
            config: Experiment configuration.
            resume: If True, resumes from checkpoint if available.
            
        Returns:
            Result dictionary, or None if experiment was skipped.
        """
        # Check if already completed
        if self.is_completed(config.name):
            print(f"\n[SKIP] {config.name} already completed")
            return None

        # Set seed for reproducibility
        set_seed(config.seed)

        print(f"\n{'='*70}")
        print(f"RUNNING: {config.name}")
        print(f"{'='*70}")
        print(f"  Loss:               {config.loss_name}")
        print(f"  LR:                 {config.learning_rate}")
        print(f"  Batch size:         {config.batch_size}")
        print(f"  Scheduler factor:   {config.scheduler_factor}")
        print(f"  Scheduler patience: {config.scheduler_patience}")
        print(f"  Seed:               {config.seed}")
        print("-" * 70)

        checkpoint_dir = os.path.join(self.checkpoints_dir, config.name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        train_loader, val_loader, test_loader = self._get_dataloaders(config.batch_size)

        model = config.model_class(
            encoder_name=config.encoder_name,
            encoder_weights=config.encoder_weights,
            in_channels=config.in_channels,
            num_classes=config.num_classes,
        )

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters:         {n_params:,}")

        loss_fn = get_loss(config.loss_name, **config.loss_kwargs)
        print(f"  Loss fn:            {loss_fn.__class__.__name__}")

        resume_path = os.path.join(checkpoint_dir, "checkpoint.pth")
        if not resume or not os.path.isfile(resume_path):
            resume_path = None
            print(f"  Resume:             Starting from scratch")
        else:
            print(f"  Resume:             From checkpoint")

        print("-" * 70)

        # Training
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

        # Load best model for evaluation
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device).eval()

        # =====================================================================
        # VALIDATION SET EVALUATION (for model selection)
        # =====================================================================
        print("\nEvaluating on VALIDATION set (for model selection)...")
        df_boa_val = evaluate_test_dataset(
            val_loader, model, device=self.device, normalize_imgs=False
        )
        val_metrics = self._extract_boa_metrics(df_boa_val, prefix="val_")
        
        conf_matrix_val = evaluate_model(
            val_loader, model, device=self.device, normalize_imgs=False
        )
        metrics_val = compute_metrics(conf_matrix_val)
        val_metrics["val_accuracy"] = metrics_val["Overall"]["Accuracy"]

        # =====================================================================
        # TEST SET EVALUATION (for final reporting only)
        # =====================================================================
        print("Evaluating on TEST set (for final reporting)...")
        df_boa_test = evaluate_test_dataset(
            test_loader, model, device=self.device, normalize_imgs=False
        )
        test_metrics = self._extract_boa_metrics(df_boa_test, prefix="test_")
        
        conf_matrix_test = evaluate_model(
            test_loader, model, device=self.device, normalize_imgs=False
        )
        metrics_test = compute_metrics(conf_matrix_test)
        test_metrics["test_accuracy"] = metrics_test["Overall"]["Accuracy"]

        # Build result dictionary
        result = {
            "name": config.name,
            "loss": config.loss_name,
            "lr": config.learning_rate,
            "batch_size": config.batch_size,
            "sched_factor": config.scheduler_factor,
            "sched_patience": config.scheduler_patience,
            "seed": config.seed,
            "best_val_loss": checkpoint.get("best_metric", -1),
            "best_epoch": checkpoint.get("best_epoch", -1),
            # Validation metrics (USE FOR SELECTION)
            **val_metrics,
            # Test metrics (USE FOR REPORTING)
            **test_metrics,
            "n_params": n_params,
            "timestamp": datetime.now().isoformat(),
        }

        self.results.append(result)
        self._save_results()

        print(f"\n{'='*70}")
        print(f"RESULTS: {config.name}")
        print(f"{'='*70}")
        print(f"  Best val_loss: {result['best_val_loss']:.4f} (epoch {result['best_epoch']})")
        print(f"\n  [VALIDATION - for selection]")
        print(f"    Cloud BOA:  {val_metrics['val_cloud_boa']:.4f}")
        print(f"    Shadow BOA: {val_metrics['val_shadow_boa']:.4f}")
        print(f"    Valid BOA:  {val_metrics['val_valid_boa']:.4f}")
        print(f"    Mean BOA:   {val_metrics['val_mean_boa']:.4f}")
        print(f"    Accuracy:   {val_metrics['val_accuracy']:.4f}")
        print(f"\n  [TEST - for reporting]")
        print(f"    Cloud BOA:  {test_metrics['test_cloud_boa']:.4f}")
        print(f"    Shadow BOA: {test_metrics['test_shadow_boa']:.4f}")
        print(f"    Valid BOA:  {test_metrics['test_valid_boa']:.4f}")
        print(f"    Mean BOA:   {test_metrics['test_mean_boa']:.4f}")
        print(f"    Accuracy:   {test_metrics['test_accuracy']:.4f}")
        print(f"{'='*70}\n")

        return result

    def run_loss_comparison(
        self,
        model_class: Type[nn.Module],
        losses: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Stage 1: Compare loss functions with fixed hyperparameters.
        
        Args:
            model_class: Model class to use.
            losses: List of loss names to compare.
            
        Returns:
            Summary DataFrame of results.
        """
        if losses is None:
            losses = ["ce", "dice_ce", "focal", "focal_tversky", "dice_focal"]

        self.print_setup(model_class, losses=losses)

        print("\n" + "#" * 70)
        print("STAGE 1: LOSS FUNCTION COMPARISON")
        print("#" * 70)

        for i, loss_name in enumerate(losses, 1):
            print(f"\n[{i}/{len(losses)}] Testing: {loss_name}")

            config = ExperimentConfig(
                name=f"stage1_loss_{loss_name}",
                model_class=model_class,
                loss_name=loss_name,
                learning_rate=1e-4,
                batch_size=4,
                scheduler_factor=0.1,
                scheduler_patience=3,
                seed=self.seed,
            )
            self.run(config, resume=True)

        print("\n" + "#" * 70)
        print("STAGE 1 COMPLETE")
        print("#" * 70)
        self.print_summary()

        return self.get_summary()

    def run_hyperparameter_tuning(
        self,
        model_class: Type[nn.Module],
        losses: Union[str, List[str]],
        learning_rates: Optional[List[float]] = None,
        batch_sizes: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Stage 2: Tune LR and batch size for one or more losses.
        
        Args:
            model_class: Model class to use.
            losses: Single loss name or list of loss names to tune.
            learning_rates: List of learning rates to test.
            batch_sizes: List of batch sizes to test.
            
        Returns:
            Summary DataFrame of results.
        """
        if isinstance(losses, str):
            losses = [losses]
            
        if learning_rates is None:
            learning_rates = [5e-4, 1e-4, 5e-5]
        if batch_sizes is None:
            batch_sizes = [4, 8]

        self.print_setup(
            model_class,
            losses=losses,
            learning_rates=learning_rates,
            batch_sizes=batch_sizes,
        )

        print("\n" + "#" * 70)
        print(f"STAGE 2: HYPERPARAMETER TUNING")
        print(f"  Losses: {losses}")
        print("#" * 70)

        total = len(losses) * len(learning_rates) * len(batch_sizes)
        current = 0

        for loss_name in losses:
            print(f"\n{'='*50}")
            print(f"TUNING: {loss_name}")
            print(f"{'='*50}")
            
            for lr in learning_rates:
                for bs in batch_sizes:
                    current += 1
                    print(f"\n[{current}/{total}] loss={loss_name}, lr={lr}, bs={bs}")

                    config = ExperimentConfig(
                        name=f"stage2_{loss_name}_lr{lr}_bs{bs}",
                        model_class=model_class,
                        loss_name=loss_name,
                        learning_rate=lr,
                        batch_size=bs,
                        scheduler_factor=0.1,
                        scheduler_patience=3,
                        seed=self.seed,
                    )
                    self.run(config, resume=True)

        print("\n" + "#" * 70)
        print("STAGE 2 COMPLETE")
        print("#" * 70)
        self.print_summary(stage="stage2")

        return self.get_summary()

    def run_scheduler_tuning(
        self,
        model_class: Type[nn.Module],
        best_loss: str,
        best_lr: float,
        best_batch_size: int,
        factors: Optional[List[float]] = None,
        patiences: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Stage 3: Tune ReduceLROnPlateau hyperparameters.
        
        Args:
            model_class: Model class to use.
            best_loss: Best loss function from Stage 2.
            best_lr: Best learning rate from Stage 2.
            best_batch_size: Best batch size from Stage 2.
            factors: List of scheduler factors to test.
            patiences: List of scheduler patience values to test.
            
        Returns:
            Summary DataFrame of results.
        """
        if factors is None:
            factors = [0.1, 0.5]
        if patiences is None:
            patiences = [2, 3, 5]

        print("\n" + "#" * 70)
        print("STAGE 3: SCHEDULER TUNING (ReduceLROnPlateau)")
        print(f"  Loss: {best_loss}, LR: {best_lr}, BS: {best_batch_size}")
        print(f"  Factors:   {factors}")
        print(f"  Patiences: {patiences}")
        print(f"  Seed:      {self.seed}")
        print("#" * 70)

        total = len(factors) * len(patiences)
        current = 0

        for factor in factors:
            for sched_patience in patiences:
                current += 1
                print(f"\n[{current}/{total}] factor={factor}, patience={sched_patience}")

                config = ExperimentConfig(
                    name=f"stage3_{best_loss}_f{factor}_p{sched_patience}",
                    model_class=model_class,
                    loss_name=best_loss,
                    learning_rate=best_lr,
                    batch_size=best_batch_size,
                    scheduler_factor=factor,
                    scheduler_patience=sched_patience,
                    seed=self.seed,
                )
                self.run(config, resume=True)

        print("\n" + "#" * 70)
        print("STAGE 3 COMPLETE")
        print("#" * 70)
        self.print_summary(stage="stage3")

        return self.get_summary()

    def get_summary(
        self, 
        stage: Optional[str] = None,
        sort_by: str = "val_mean_boa",
        ascending: bool = False,
    ) -> pd.DataFrame:
        """Get summary DataFrame of results.
        
        Args:
            stage: Filter by stage ("stage1", "stage2", "stage3"). 
                   If None, returns all results.
            sort_by: Column to sort by. Default is "val_mean_boa".
            ascending: Sort order. Default is False (higher is better).
                   
        Returns:
            Sorted DataFrame with experiment results.
        """
        df = pd.DataFrame(self.results)
        
        if df.empty:
            return df
            
        # Filter by stage if specified
        if stage is not None:
            df = df[df["name"].str.startswith(stage)]
        
        # Sort
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending)
        
        return df

    def get_best_from_stage(
        self, 
        stage: str, 
        sort_by: str = "val_mean_boa",
    ) -> Optional[Dict]:
        """Get best result from a specific stage based on VALIDATION metrics.
        
        Args:
            stage: Stage prefix ("stage1", "stage2", "stage3").
            sort_by: Validation metric to use for selection.
            
        Returns:
            Best result dictionary or None if no results for stage.
        """
        df = self.get_summary(stage=stage, sort_by=sort_by, ascending=False)
        if df.empty:
            return None
        return df.iloc[0].to_dict()

    def print_summary(
        self, 
        stage: Optional[str] = None,
        sort_by: str = "val_mean_boa",
    ) -> None:
        """Print formatted summary of results.
        
        Args:
            stage: Filter by stage. If None, shows all results.
            sort_by: Column to sort by.
        """
        df = self.get_summary(stage=stage, sort_by=sort_by, ascending=False)

        if df.empty:
            print("No results yet.")
            return

        stage_label = f" ({stage})" if stage else ""

        print("\n" + "=" * 130)
        print(f"EXPERIMENT RESULTS{stage_label} - Sorted by {sort_by} (VALIDATION)")
        print("=" * 130)

        # Columns for display
        cols = [
            "name", "loss", "lr", "batch_size",
            "val_cloud_boa", "val_shadow_boa", "val_valid_boa", "val_mean_boa", "val_accuracy",
            "test_mean_boa", "test_accuracy",
        ]
        available_cols = [c for c in cols if c in df.columns]
        
        # Format for display
        df_display = df[available_cols].copy()
        print(df_display.to_string(index=False))

        print("\n" + "-" * 130)
        best = df.iloc[0]
        print(f"BEST (by {sort_by}): {best['name']}")
        print(f"  Config: loss={best['loss']}, LR={best['lr']}, BS={best['batch_size']}")
        
        print(f"\n  [VALIDATION - used for selection]")
        if "val_cloud_boa" in best:
            print(f"    Cloud BOA:  {best['val_cloud_boa']:.4f}")
            print(f"    Shadow BOA: {best['val_shadow_boa']:.4f}")
            print(f"    Valid BOA:  {best['val_valid_boa']:.4f}")
            print(f"    Mean BOA:   {best['val_mean_boa']:.4f}")
            print(f"    Accuracy:   {best['val_accuracy']:.4f}")
        
        print(f"\n  [TEST - for paper reporting]")
        if "test_cloud_boa" in best:
            print(f"    Cloud BOA:  {best['test_cloud_boa']:.4f}")
            print(f"    Shadow BOA: {best['test_shadow_boa']:.4f}")
            print(f"    Valid BOA:  {best['test_valid_boa']:.4f}")
            print(f"    Mean BOA:   {best['test_mean_boa']:.4f}")
            print(f"    Accuracy:   {best['test_accuracy']:.4f}")
        
        print("=" * 130 + "\n")

    def print_stage_comparison(self) -> None:
        """Print comparison of best results from each stage."""
        print("\n" + "=" * 90)
        print("STAGE COMPARISON - Best from each stage (by val_mean_boa)")
        print("=" * 90)
        
        for stage in ["stage1", "stage2", "stage3"]:
            best = self.get_best_from_stage(stage)
            if best:
                print(f"\n{stage.upper()}:")
                print(f"  Name: {best['name']}")
                print(f"  Loss: {best['loss']}, LR: {best['lr']}, BS: {best['batch_size']}")
                print(f"  [VAL]  Mean BOA: {best.get('val_mean_boa', 'N/A'):.4f}, Acc: {best.get('val_accuracy', 'N/A'):.4f}")
                print(f"  [TEST] Mean BOA: {best.get('test_mean_boa', 'N/A'):.4f}, Acc: {best.get('test_accuracy', 'N/A'):.4f}")
        
        print("\n" + "=" * 90)

    def _save_results(self) -> None:
        """Save results to disk."""
        df = pd.DataFrame(self.results)
        
        # Save CSV
        csv_path = os.path.join(self.results_dir, "experiments_summary.csv")
        df.to_csv(csv_path, index=False)
        
        # Save pickle for full data
        pkl_path = os.path.join(self.results_dir, "experiments_full.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(self.results, f)

    def export_for_paper(
        self, 
        output_path: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> pd.DataFrame:
        """Export results in a format suitable for academic papers.
        
        Uses TEST metrics for reporting (as per ML best practices).
        
        Args:
            output_path: Path to save LaTeX table. If None, only returns DataFrame.
            stage: Filter by stage.
            
        Returns:
            Formatted DataFrame with results.
        """
        df = self.get_summary(stage=stage)
        
        if df.empty:
            print("No results to export.")
            return df
        
        # Select columns for paper (TEST metrics)
        paper_df = df[[
            "name", "loss", "lr", "batch_size", 
            "test_cloud_boa", "test_shadow_boa", "test_valid_boa", 
            "test_mean_boa", "test_accuracy"
        ]].copy()
        
        paper_df.columns = [
            "Experiment", "Loss", "LR", "BS",
            "Cloud BOA", "Shadow BOA", "Valid BOA",
            "Mean BOA", "Accuracy"
        ]
        
        # Format numeric columns
        for col in ["Cloud BOA", "Shadow BOA", "Valid BOA", "Mean BOA", "Accuracy"]:
            paper_df[col] = paper_df[col].apply(lambda x: f"{x:.4f}")
        
        if output_path:
            latex = paper_df.to_latex(index=False, caption="Experiment Results (Test Set)")
            with open(output_path, "w") as f:
                f.write(latex)
            print(f"LaTeX table saved to {output_path}")
        
        return paper_df