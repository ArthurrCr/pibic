"""Results management and visualization for model comparison."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from utils.constants import CLASS_NAMES, EXPERIMENTS, METRIC_NAMES


@dataclass
class ModelResult:
    """
    Data structure for storing model evaluation results.

    Attributes:
        metrics: Per-class metrics dictionary.
        confusion_matrix: Confusion matrix as numpy array.
        overall_accuracy: Overall accuracy score.
        timestamp: ISO format timestamp of when results were created.
        boa_baseline: BOA results per experiment.
        optimal_thresholds: Optimal threshold results per experiment.
        additional_info: Additional metadata (parameters, GFLOPs, etc.).
    """

    metrics: Dict
    confusion_matrix: np.ndarray
    overall_accuracy: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    boa_baseline: Dict = field(default_factory=dict)
    optimal_thresholds: Dict = field(default_factory=dict)
    additional_info: Dict = field(default_factory=dict)


class ResultsManager:
    """
    Manages model results and creates comparative visualizations.

    Provides methods for storing results, parsing metrics, and generating
    various comparison plots across multiple models.

    Attributes:
        results: Dictionary mapping model names to ModelResult objects.
    """

    def __init__(self):
        """Initialize the results manager."""
        self.results: Dict[str, ModelResult] = {}
        self._color_map: Dict[str, tuple] = {}
        self._palette = plt.cm.Set3(np.linspace(0, 1, 12))

    def _get_color(self, model_name: str) -> tuple:
        """Get a consistent color for a model (creates one if needed)."""
        if model_name not in self._color_map:
            idx = len(self._color_map) % len(self._palette)
            self._color_map[model_name] = self._palette[idx]
        return self._color_map[model_name]

    def save_model_results(
        self,
        model_name: str,
        metrics: Dict,
        conf_matrix: np.ndarray,
        overall_accuracy: float,
        additional_info: Optional[Dict] = None,
    ) -> None:
        """
        Save results for a model.

        Args:
            model_name: Name identifier for the model.
            metrics: Per-class metrics dictionary.
            conf_matrix: Confusion matrix.
            overall_accuracy: Overall accuracy score.
            additional_info: Optional additional metadata.
        """
        self.results[model_name] = ModelResult(
            metrics=metrics,
            confusion_matrix=conf_matrix,
            overall_accuracy=overall_accuracy,
            additional_info=additional_info or {},
        )

    def parse_metrics_from_output(
        self,
        model_name: str,
        metrics_dict: Dict,
        conf_matrix: np.ndarray,
    ) -> None:
        """
        Convert evaluation output dictionary to internal structure.

        Args:
            model_name: Name identifier for the model.
            metrics_dict: Raw metrics dictionary from evaluation.
            conf_matrix: Confusion matrix.
        """
        parsed = {}
        for c in CLASS_NAMES:
            if c in metrics_dict:
                parsed[c] = {
                    k: metrics_dict[c][k]
                    for k in METRIC_NAMES + ["Support"]
                    if k in metrics_dict[c]
                }

        self.save_model_results(
            model_name,
            parsed,
            conf_matrix,
            metrics_dict["Overall"]["Accuracy"],
        )

    def save_boa_results(
        self,
        model_name: str,
        df_results: Optional[pd.DataFrame] = None,
        threshold_results: Optional[Dict] = None,
        experiment: Optional[str] = None,
    ) -> None:
        """
        Save BOA baseline and/or optimal threshold results.

        Args:
            model_name: Name identifier for the model.
            df_results: DataFrame with BOA baseline results.
            threshold_results: Threshold optimization results.
            experiment: Experiment name for threshold results.
        """
        if model_name not in self.results:
            self.results[model_name] = ModelResult(
                metrics={},
                confusion_matrix=np.array([]),
                overall_accuracy=0.0,
            )

        if df_results is not None:
            for _, row in df_results.iterrows():
                exp = row["Experiment"]
                self.results[model_name].boa_baseline[exp] = float(row["Median BOA"])

        if threshold_results is not None and experiment is not None:
            self.results[model_name].optimal_thresholds[experiment] = threshold_results

    def plot_metric_comparison(
        self,
        metric: str,
        models: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (20, 12),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot a specific metric by class comparing multiple models.

        Highlights the best performing model for each class with a border.

        Args:
            metric: Name of the metric to plot.
            models: List of model names. If None, uses all models.
            figsize: Figure size.
            save_path: If provided, saves the figure to this path.
        """
        if models is None:
            models = sorted(self.results.keys())

        # Validate data
        for m in models:
            for cname in CLASS_NAMES:
                if (
                    cname not in self.results[m].metrics
                    or metric not in self.results[m].metrics[cname]
                ):
                    raise ValueError(
                        f"Metric '{metric}' missing for class '{cname}' "
                        f"in model '{m}'."
                    )

        # Build values matrix [model, class]
        values_mat = np.array([
            [self.results[m].metrics[c][metric] for c in CLASS_NAMES]
            for m in models
        ])

        # Find best index for each class
        if metric in ("Omission Error", "Commission Error"):
            best_idx = values_mat.argmin(axis=0)
        else:
            best_idx = values_mat.argmax(axis=0)

        # Plot
        n_models = len(models)
        width = 0.8 / n_models
        x = np.arange(len(CLASS_NAMES))

        fig, ax = plt.subplots(figsize=figsize)

        for i, model in enumerate(models):
            vals = values_mat[i]
            offset = (i - n_models / 2 + 0.5) * width
            color = self._get_color(model)

            bars = ax.bar(x + offset, vals, width, alpha=0.85, color=color)

            for j, bar in enumerate(bars):
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.005,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
                if i == best_idx[j]:
                    bar.set_edgecolor("k")
                    bar.set_linewidth(2)
                    bar.set_linestyle("--")

        legend_patches = [
            Patch(color=self._get_color(m), label=m, alpha=0.85) for m in models
        ]

        ax.set_xlabel("Classes", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"{metric} Comparison Across Models", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, fontsize=10)
        ax.legend(
            handles=legend_patches,
            title="Models",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_threshold_curve(
        self,
        model_name: str,
        experiment: str,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot median BOA vs threshold curve for an experiment.

        Args:
            model_name: Name of the model.
            experiment: Name of the experiment.
            figsize: Figure size.
            save_path: If provided, saves the figure to this path.
        """
        if experiment not in self.results[model_name].optimal_thresholds:
            print(f"Experiment '{experiment}' not found for {model_name}")
            return

        data = self.results[model_name].optimal_thresholds[experiment]

        plt.figure(figsize=figsize)
        plt.plot(data["thresholds"], data["median_boas"], linewidth=2)
        plt.scatter(
            data["best_threshold"],
            data["best_median_boa"],
            s=100,
            zorder=5,
            label=f"t* = {data['best_threshold']:.2f}",
        )

        plt.xlabel("Threshold", fontsize=12)
        plt.ylabel("Median BOA", fontsize=12)
        plt.title(
            f"Threshold Optimization - {experiment} - {model_name}",
            fontsize=14,
        )
        plt.grid(alpha=0.3)
        plt.legend(fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def get_summary_dataframe(
        self,
        models: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Generate a summary DataFrame with key metrics for all models.

        Args:
            models: List of model names. If None, uses all models.

        Returns:
            DataFrame with model metrics summary.
        """
        if models is None:
            models = sorted(self.results.keys())

        rows = []
        for model in models:
            result = self.results[model]
            row = {"Model": model, "Overall Accuracy": result.overall_accuracy}

            for class_name in CLASS_NAMES:
                if class_name in result.metrics:
                    for metric in ["F1-Score", "Precision", "Recall"]:
                        if metric in result.metrics[class_name]:
                            key = f"{class_name} {metric}"
                            row[key] = result.metrics[class_name][metric]

            rows.append(row)

        return pd.DataFrame(rows)