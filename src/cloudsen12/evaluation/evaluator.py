"""Model evaluation orchestrator for cloud segmentation."""

import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from cloudsen12.evaluation.boa import evaluate_test_dataset
from cloudsen12.evaluation.metrics import compute_metrics, evaluate_model
from cloudsen12.evaluation.results import ResultsManager


class ModelEvaluator:
    """Orchestrates model evaluation with caching.

    Runs confusion matrix computation and patch-level BOA evaluation,
    persisting results to disk for reproducibility. Supports paired
    statistical comparison between models via Wilcoxon signed-rank test.

    Attributes:
        manager: ResultsManager instance for storing results.
        device: PyTorch device for computation.
        cache_dir: Directory for caching evaluation results.
        patch_data: Per-patch BOA data keyed by model name.
    """

    DEFAULT_MODEL_CONFIGS: Dict[str, Dict[str, bool]] = {
        "CloudS2Mask ensemble": {
            "use_ensemble": True,
            "normalize_imgs": True,
        },
        "CloudS2Mask Dice_1 (single)": {
            "use_ensemble": False,
            "normalize_imgs": True,
        },
        "CloudS2Mask Dice_2 (single)": {
            "use_ensemble": False,
            "normalize_imgs": True,
        },
    }

    def __init__(
        self,
        manager: ResultsManager,
        device: Optional[torch.device] = None,
        cache_dir: str = "./evaluation_cache",
    ) -> None:
        self.manager = manager
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"Using device: {self.device}")
        print(f"Cache directory: {self.cache_dir}")

        self.model_configs = dict(self.DEFAULT_MODEL_CONFIGS)
        self.patch_data: Dict[str, pd.DataFrame] = {}
        self._model_index: Dict[str, str] = {}
        self._load_existing_results()

    def _load_existing_results(self) -> None:
        """Load previously cached evaluation results."""
        print("Checking evaluation cache...")
        loaded_count = 0

        index_file = self.cache_dir / "model_index.pkl"
        model_index: Dict[str, str] = {}

        if index_file.exists():
            try:
                with open(index_file, "rb") as f:
                    model_index = pickle.load(f)
            except Exception:
                model_index = {}

        for file in self.cache_dir.glob("*_evaluation_results.pkl"):
            try:
                with open(file, "rb") as f:
                    result = pickle.load(f)

                model_name = None
                for name, path in model_index.items():
                    if path == file.name:
                        model_name = name
                        break

                if not model_name:
                    parts = file.stem.split("_")
                    if len(parts) > 2:
                        model_name = (
                            "_".join(parts[:-2])
                            .replace("_", " ")
                            .replace("-", "/")
                        )

                if model_name:
                    self.manager.results[model_name] = result
                    loaded_count += 1
                    print(f"  Loaded: {model_name}")

            except Exception as e:
                print(f"  Error loading {file.name}: {e}")

        # Load cached per-patch data.
        for file in self.cache_dir.glob("*_patch_data.pkl"):
            try:
                with open(file, "rb") as f:
                    data = pickle.load(f)
                model_name = data.get("model_name")
                if model_name and "patch_df" in data:
                    self.patch_data[model_name] = data["patch_df"]
                    print(f"  Loaded patch data: {model_name}")
            except Exception as e:
                print(f"  Error loading {file.name}: {e}")

        if loaded_count > 0:
            print(f"{loaded_count} evaluation(s) loaded from cache.")
        else:
            print("No previous evaluations found.")

        self._model_index = model_index

    def _get_cache_path(self, model_name: str, eval_type: str) -> Path:
        """Return a filesystem-safe cache path for the given model and type."""
        name_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]
        safe_name = model_name.replace("/", "-").replace(" ", "_")
        if len(safe_name) > 50:
            safe_name = safe_name[:50]
        return self.cache_dir / f"{safe_name}_{name_hash}_{eval_type}.pkl"

    def _save_to_cache(self, model_name: str, eval_type: str) -> None:
        """Persist current results for model_name to disk."""
        if model_name not in self.manager.results:
            return

        cache_path = self._get_cache_path(model_name, eval_type)
        with open(cache_path, "wb") as f:
            pickle.dump(self.manager.results[model_name], f)

        index_file = self.cache_dir / "model_index.pkl"
        self._model_index[model_name] = cache_path.name
        with open(index_file, "wb") as f:
            pickle.dump(self._model_index, f)

    def _save_patch_data(self, model_name: str) -> None:
        """Persist per-patch BOA data to disk."""
        if model_name not in self.patch_data:
            return
        cache_path = self._get_cache_path(model_name, "patch_data")
        with open(cache_path, "wb") as f:
            pickle.dump(
                {"model_name": model_name, "patch_df": self.patch_data[model_name]},
                f,
            )

    def evaluate_confusion_matrix(
        self,
        model_name: str,
        models: Union[torch.nn.Module, List[torch.nn.Module]],
        test_loader: torch.utils.data.DataLoader,
        use_ensemble: bool = False,
        normalize_imgs: bool = False,
    ) -> None:
        """Evaluate model and compute confusion matrix metrics."""
        print(f"Evaluating confusion matrix for: {model_name}")

        conf_matrix = evaluate_model(
            test_loader, models,
            device=str(self.device),
            use_ensemble=use_ensemble,
            normalize_imgs=normalize_imgs,
        )
        metrics = compute_metrics(conf_matrix)
        overall_acc = metrics["Overall"]["Accuracy"]

        self.manager.save_model_results(
            model_name, metrics, conf_matrix, overall_acc
        )
        self._save_to_cache(model_name, "evaluation_results")
        print(f"Overall Accuracy: {overall_acc:.4f}")

    def evaluate_boa(
        self,
        model_name: str,
        models: Union[torch.nn.Module, List[torch.nn.Module]],
        test_loader: torch.utils.data.DataLoader,
        use_ensemble: bool = False,
        normalize_imgs: bool = False,
    ) -> None:
        """Evaluate patch-level BOA metrics with argmax predictions."""
        print(f"Evaluating BOA for: {model_name}")

        summary_df, patch_df = evaluate_test_dataset(
            test_loader, models,
            device=str(self.device),
            use_ensemble=use_ensemble,
            normalize_imgs=normalize_imgs,
        )

        self.manager.save_boa_results(model_name, df_results=summary_df)
        self.patch_data[model_name] = patch_df
        self._save_to_cache(model_name, "evaluation_results")
        self._save_patch_data(model_name)
        print(summary_df.to_string(index=False))

    def full_evaluation(
        self,
        model_name: str,
        models: Union[torch.nn.Module, List[torch.nn.Module]],
        test_loader: torch.utils.data.DataLoader,
        use_ensemble: Optional[bool] = None,
        normalize_imgs: Optional[bool] = None,
    ) -> None:
        """Run complete evaluation: confusion matrix + BOA.

        If the model_name matches a known configuration in model_configs,
        the ensemble and normalization flags are applied automatically
        unless explicitly overridden.
        """
        if model_name in self.model_configs:
            config = self.model_configs[model_name]
            if use_ensemble is None:
                use_ensemble = config.get("use_ensemble", False)
            if normalize_imgs is None:
                normalize_imgs = config.get("normalize_imgs", False)
        else:
            use_ensemble = use_ensemble if use_ensemble is not None else False
            normalize_imgs = normalize_imgs if normalize_imgs is not None else False

        separator = "=" * 60
        print(f"\n{separator}")
        print(f"FULL EVALUATION: {model_name}")
        print(separator)
        print(f"use_ensemble: {use_ensemble}")
        print(f"normalize_imgs: {normalize_imgs}")

        self.evaluate_confusion_matrix(
            model_name, models, test_loader, use_ensemble, normalize_imgs
        )
        self.evaluate_boa(
            model_name, models, test_loader, use_ensemble, normalize_imgs
        )

        print(f"\n{separator}")
        print(f"EVALUATION COMPLETE: {model_name}")
        print(separator)

    # ------------------------------------------------------------------
    # Statistical comparison
    # ------------------------------------------------------------------

    def wilcoxon_test(
        self,
        model_a: str,
        model_b: str,
        experiments: Optional[List[str]] = None,
        alternative: str = "two-sided",
    ) -> pd.DataFrame:
        """Run paired Wilcoxon signed-rank test on per-patch BOA.

        Compares model_a vs model_b for each binary experiment. Patches
        where either model has NaN BOA are excluded from the test.

        Args:
            model_a: Name of the first model (expected to be better).
            model_b: Name of the second model (baseline).
            experiments: Experiment names to compare. If None, uses all
                experiments present in the patch data.
            alternative: 'two-sided', 'greater', or 'less'. Use 'greater'
                to test H1: model_a > model_b.

        Returns:
            DataFrame with columns: experiment, n_patches, median_a,
            median_b, median_diff, statistic, p_value, significant.

        Raises:
            ValueError: If patch data is missing for either model.
        """
        for name in (model_a, model_b):
            if name not in self.patch_data:
                raise ValueError(
                    f"No patch data for '{name}'. "
                    f"Run full_evaluation first."
                )

        df_a = self.patch_data[model_a]
        df_b = self.patch_data[model_b]

        if experiments is None:
            experiments = sorted(df_a["experiment"].unique())

        rows = []
        for exp in experiments:
            boa_a = df_a.loc[df_a["experiment"] == exp, "BOA"].values
            boa_b = df_b.loc[df_b["experiment"] == exp, "BOA"].values

            if len(boa_a) != len(boa_b):
                print(
                    f"  WARNING: patch count mismatch for {exp} "
                    f"({len(boa_a)} vs {len(boa_b)}). Skipping."
                )
                continue

            # Drop patches where either model has NaN.
            valid = ~(np.isnan(boa_a) | np.isnan(boa_b))
            boa_a_valid = boa_a[valid]
            boa_b_valid = boa_b[valid]
            n_valid = int(valid.sum())

            diff = boa_a_valid - boa_b_valid

            # Wilcoxon requires at least one non-zero difference.
            if np.all(diff == 0) or n_valid < 10:
                rows.append({
                    "experiment": exp,
                    "n_patches": n_valid,
                    "median_a": np.nanmedian(boa_a_valid),
                    "median_b": np.nanmedian(boa_b_valid),
                    "median_diff": np.nanmedian(diff),
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "significant": False,
                })
                continue

            stat, p_val = wilcoxon(
                boa_a_valid, boa_b_valid, alternative=alternative
            )

            rows.append({
                "experiment": exp,
                "n_patches": n_valid,
                "median_a": np.nanmedian(boa_a_valid),
                "median_b": np.nanmedian(boa_b_valid),
                "median_diff": np.nanmedian(diff),
                "statistic": stat,
                "p_value": p_val,
                "significant": p_val < 0.05,
            })

        result_df = pd.DataFrame(rows)

        # Print results.
        separator = "=" * 80
        print(f"\n{separator}")
        print(f"WILCOXON SIGNED-RANK TEST (alternative='{alternative}')")
        print(f"  Model A: {model_a}")
        print(f"  Model B: {model_b}")
        print(separator)

        for _, row in result_df.iterrows():
            sig = "***" if row["p_value"] < 0.001 else (
                "**" if row["p_value"] < 0.01 else (
                    "*" if row["p_value"] < 0.05 else "n.s."
                )
            )
            print(
                f"  {row['experiment']:20s}  "
                f"n={row['n_patches']:4d}  "
                f"median_A={row['median_a']:.4f}  "
                f"median_B={row['median_b']:.4f}  "
                f"diff={row['median_diff']:+.4f}  "
                f"W={row['statistic']:10.1f}  "
                f"p={row['p_value']:.2e}  {sig}"
            )

        print(separator)
        return result_df