"""Experiment stages: loss comparison, hyperparameter tuning, scheduler tuning."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Type, Union

import pandas as pd
import torch.nn as nn

from cloudsen12.experiment.config import ExperimentConfig
from cloudsen12.experiment.reporting import get_summary, print_summary

if TYPE_CHECKING:
    from cloudsen12.experiment.runner import ExperimentRunner


def run_loss_comparison(
    runner: ExperimentRunner,
    model_class: Type[nn.Module],
    losses: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Stage 1: Compare loss functions with fixed hyperparameters.

    Args:
        runner: ExperimentRunner instance.
        model_class: Model class to instantiate.
        losses: Loss names to compare.

    Returns:
        Summary DataFrame of results.
    """
    if losses is None:
        losses = ["ce", "dice_ce", "focal", "focal_tversky", "dice_focal"]

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
            seed=runner.seed,
        )
        runner.run(config, resume=True)

    print("\n" + "#" * 70)
    print("STAGE 1 COMPLETE")
    print("#" * 70)
    print_summary(runner)

    return get_summary(runner)


def run_hyperparameter_tuning(
    runner: ExperimentRunner,
    model_class: Type[nn.Module],
    losses: Union[str, List[str]],
    learning_rates: Optional[List[float]] = None,
    batch_sizes: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Stage 2: Tune LR and batch size for selected losses.

    Args:
        runner: ExperimentRunner instance.
        model_class: Model class to instantiate.
        losses: Single loss name or list of loss names.
        learning_rates: Learning rates to test.
        batch_sizes: Batch sizes to test.

    Returns:
        Summary DataFrame of results.
    """
    if isinstance(losses, str):
        losses = [losses]
    if learning_rates is None:
        learning_rates = [5e-4, 1e-4, 5e-5]
    if batch_sizes is None:
        batch_sizes = [4, 8]

    print("\n" + "#" * 70)
    print("STAGE 2: HYPERPARAMETER TUNING")
    print(f"  Losses: {losses}")
    print("#" * 70)

    total = len(losses) * len(learning_rates) * len(batch_sizes)
    current = 0

    for loss_name in losses:
        print(f"\n{'=' * 50}")
        print(f"TUNING: {loss_name}")
        print("=" * 50)

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
                    seed=runner.seed,
                )
                runner.run(config, resume=True)

    print("\n" + "#" * 70)
    print("STAGE 2 COMPLETE")
    print("#" * 70)
    print_summary(runner, stage="stage2")

    return get_summary(runner)


def run_scheduler_tuning(
    runner: ExperimentRunner,
    model_class: Type[nn.Module],
    best_loss: str,
    best_lr: float,
    best_batch_size: int,
    factors: Optional[List[float]] = None,
    patiences: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Stage 3: Tune ReduceLROnPlateau hyperparameters.

    Args:
        runner: ExperimentRunner instance.
        model_class: Model class to instantiate.
        best_loss: Best loss from Stage 2.
        best_lr: Best learning rate from Stage 2.
        best_batch_size: Best batch size from Stage 2.
        factors: Scheduler factors to test.
        patiences: Scheduler patience values to test.

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
    print(f"  Seed:      {runner.seed}")
    print("#" * 70)

    total = len(factors) * len(patiences)
    current = 0

    for factor in factors:
        for sched_patience in patiences:
            current += 1
            print(
                f"\n[{current}/{total}] "
                f"factor={factor}, patience={sched_patience}"
            )

            config = ExperimentConfig(
                name=f"stage3_{best_loss}_f{factor}_p{sched_patience}",
                model_class=model_class,
                loss_name=best_loss,
                learning_rate=best_lr,
                batch_size=best_batch_size,
                scheduler_factor=factor,
                scheduler_patience=sched_patience,
                seed=runner.seed,
            )
            runner.run(config, resume=True)

    print("\n" + "#" * 70)
    print("STAGE 3 COMPLETE")
    print("#" * 70)
    print_summary(runner, stage="stage3")

    return get_summary(runner)