"""Model evaluation orchestrator for cloud segmentation."""

import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

from cloudsen12.evaluation.boa import evaluate_test_dataset
from cloudsen12.evaluation.metrics import compute_metrics, evaluate_model
from cloudsen12.evaluation.results import ResultsManager


class ModelEvaluator:
    """Orchestrates model evaluation with caching.

    Runs confusion matrix computation and patch-level BOA evaluation,
    persisting results to disk for reproducibility.

    Attributes:
        manager: ResultsManager instance for storing results.
        device: PyTorch device for computation.
        cache_dir: Directory for caching evaluation results.
    """

    # Models that require special inference settings.
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

        df_results = evaluate_test_dataset(
            test_loader, models,
            device=str(self.device),
            use_ensemble=use_ensemble,
            normalize_imgs=normalize_imgs,
        )

        self.manager.save_boa_results(model_name, df_results=df_results)
        self._save_to_cache(model_name, "evaluation_results")
        print(df_results.to_string(index=False))

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