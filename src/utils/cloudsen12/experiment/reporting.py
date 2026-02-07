"""Reporting utilities: setup display, summaries, and paper export."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Type

import pandas as pd
import torch.nn as nn

if TYPE_CHECKING:
    from cloudsen12.experiment.runner import ExperimentRunner


def print_setup(
    runner: ExperimentRunner,
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

    train_loader, val_loader, test_loader = runner.get_dataloaders(
        batch_sizes[0]
    )

    separator = "=" * 70
    print(f"\n{separator}")
    print("EXPERIMENT SETUP")
    print(separator)

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
    print(f"  Device:          {runner.device}")
    print(f"  Max epochs:      50")
    print(f"  Early stopping:  patience=5, monitor=val_loss")
    print(f"  Optimizer:       Adam (weight_decay=1e-4)")
    print(f"  Scheduler:       ReduceLROnPlateau")
    print(f"  Seed:            {runner.seed}")

    print("\n[Evaluation Strategy]")
    print("  Selection:       Based on VALIDATION BOA metrics")
    print("  Test metrics:    Computed for final reporting only")

    print("\n[Hyperparameters to test]")
    print(f"  Learning rates:      {learning_rates}")
    print(f"  Batch sizes:         {batch_sizes}")
    print(f"  Scheduler factors:   {scheduler_factors}")
    print(f"  Scheduler patiences: {scheduler_patiences}")

    loss_descriptions = {
        "ce": "CrossEntropy (baseline)",
        "dice": "Dice Loss",
        "focal": "Focal Loss (gamma=2.0, alpha=0.25)",
        "tversky": "Tversky Loss (alpha=0.3, beta=0.7)",
        "focal_tversky": "Focal Tversky (alpha=0.3, beta=0.7, gamma=0.75)",
        "dice_ce": "Dice + CE (0.5/0.5)",
        "dice_focal": "Dice + Focal (0.5/0.5)",
    }
    print("\n[Losses to test]")
    for i, loss in enumerate(losses, 1):
        desc = loss_descriptions.get(loss, loss)
        print(f"  {i}. {loss:15} -> {desc}")

    print("\n[Output]")
    print(f"  Checkpoints:     {runner.checkpoints_dir}")
    print(f"  Results:         {runner.results_dir}")

    total = len(losses) * len(learning_rates) * len(batch_sizes)
    print(f"\n{'-' * 70}")
    print(f"Total experiments planned: {total}")
    print(f"{separator}\n")


def get_summary(
    runner: ExperimentRunner,
    stage: Optional[str] = None,
    sort_by: str = "val_mean_boa",
    ascending: bool = False,
) -> pd.DataFrame:
    """Get summary DataFrame, optionally filtered by stage."""
    df = pd.DataFrame(runner.results)
    if df.empty:
        return df
    if stage is not None:
        df = df[df["name"].str.startswith(stage)]
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)
    return df


def get_best_from_stage(
    runner: ExperimentRunner,
    stage: str,
    sort_by: str = "val_mean_boa",
) -> Optional[Dict]:
    """Get best result from a stage based on validation metrics."""
    df = get_summary(runner, stage=stage, sort_by=sort_by, ascending=False)
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def print_summary(
    runner: ExperimentRunner,
    stage: Optional[str] = None,
    sort_by: str = "val_mean_boa",
) -> None:
    """Print formatted summary of results."""
    df = get_summary(runner, stage=stage, sort_by=sort_by, ascending=False)

    if df.empty:
        print("No results yet.")
        return

    stage_label = f" ({stage})" if stage else ""
    separator = "=" * 130

    print(f"\n{separator}")
    print(
        f"EXPERIMENT RESULTS{stage_label} - Sorted by {sort_by} (VALIDATION)"
    )
    print(separator)

    cols = [
        "name", "loss", "lr", "batch_size",
        "val_cloud_boa", "val_shadow_boa", "val_valid_boa",
        "val_mean_boa", "val_accuracy",
        "test_mean_boa", "test_accuracy",
    ]
    available_cols = [c for c in cols if c in df.columns]
    print(df[available_cols].to_string(index=False))

    print(f"\n{'-' * 130}")
    best = df.iloc[0]
    print(f"BEST (by {sort_by}): {best['name']}")
    print(
        f"  Config: loss={best['loss']}, LR={best['lr']}, "
        f"BS={best['batch_size']}"
    )

    print("\n  [VALIDATION - used for selection]")
    if "val_cloud_boa" in best:
        print(f"    Cloud BOA:  {best['val_cloud_boa']:.4f}")
        print(f"    Shadow BOA: {best['val_shadow_boa']:.4f}")
        print(f"    Valid BOA:  {best['val_valid_boa']:.4f}")
        print(f"    Mean BOA:   {best['val_mean_boa']:.4f}")
        print(f"    Accuracy:   {best['val_accuracy']:.4f}")

    print("\n  [TEST - for paper reporting]")
    if "test_cloud_boa" in best:
        print(f"    Cloud BOA:  {best['test_cloud_boa']:.4f}")
        print(f"    Shadow BOA: {best['test_shadow_boa']:.4f}")
        print(f"    Valid BOA:  {best['test_valid_boa']:.4f}")
        print(f"    Mean BOA:   {best['test_mean_boa']:.4f}")
        print(f"    Accuracy:   {best['test_accuracy']:.4f}")

    print(separator + "\n")


def print_stage_comparison(runner: ExperimentRunner) -> None:
    """Print best result from each stage side by side."""
    separator = "=" * 90
    print(f"\n{separator}")
    print("STAGE COMPARISON - Best from each stage (by val_mean_boa)")
    print(separator)

    for stage in ("stage1", "stage2", "stage3"):
        best = get_best_from_stage(runner, stage)
        if best:
            print(f"\n{stage.upper()}:")
            print(f"  Name: {best['name']}")
            print(
                f"  Loss: {best['loss']}, LR: {best['lr']}, "
                f"BS: {best['batch_size']}"
            )
            print(
                f"  [VAL]  Mean BOA: {best.get('val_mean_boa', 'N/A'):.4f}, "
                f"Acc: {best.get('val_accuracy', 'N/A'):.4f}"
            )
            print(
                f"  [TEST] Mean BOA: {best.get('test_mean_boa', 'N/A'):.4f}, "
                f"Acc: {best.get('test_accuracy', 'N/A'):.4f}"
            )

    print(f"\n{separator}")


def export_for_paper(
    runner: ExperimentRunner,
    output_path: Optional[str] = None,
    stage: Optional[str] = None,
) -> pd.DataFrame:
    """Export results formatted for academic papers (uses TEST metrics).

    Args:
        runner: ExperimentRunner instance.
        output_path: Path to save LaTeX table.
        stage: Filter by stage.

    Returns:
        Formatted DataFrame.
    """
    df = get_summary(runner, stage=stage)
    if df.empty:
        print("No results to export.")
        return df

    paper_df = df[[
        "name", "loss", "lr", "batch_size",
        "test_cloud_boa", "test_shadow_boa", "test_valid_boa",
        "test_mean_boa", "test_accuracy",
    ]].copy()

    paper_df.columns = [
        "Experiment", "Loss", "LR", "BS",
        "Cloud BOA", "Shadow BOA", "Valid BOA",
        "Mean BOA", "Accuracy",
    ]

    for col in ("Cloud BOA", "Shadow BOA", "Valid BOA", "Mean BOA", "Accuracy"):
        paper_df[col] = paper_df[col].apply(lambda x: f"{x:.4f}")

    if output_path:
        latex = paper_df.to_latex(
            index=False, caption="Experiment Results (Test Set)"
        )
        with open(output_path, "w") as f:
            f.write(latex)
        print(f"LaTeX table saved to {output_path}")

    return paper_df