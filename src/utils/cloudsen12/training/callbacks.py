"""Training callbacks."""

from typing import Dict, Optional


class EarlyStopping:
    """Stops training when a monitored metric stops improving.

    Args:
        patience: Number of epochs to wait before stopping.
        mode: "max" if higher is better, "min" if lower is better.
        min_delta: Minimum change to qualify as improvement.
    """

    def __init__(
        self,
        patience: int = 5,
        mode: str = "max",
        min_delta: float = 1e-4,
    ) -> None:
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric: Optional[float] = None
        self.early_stop = False

    def __call__(self, current_metric: float) -> None:
        if self.best_metric is None:
            self.best_metric = current_metric
            return

        if self.mode == "max":
            improvement = current_metric - self.best_metric
        else:
            improvement = self.best_metric - current_metric

        if improvement < self.min_delta:
            self.counter += 1
        else:
            self.best_metric = current_metric
            self.counter = 0

        if self.counter >= self.patience:
            self.early_stop = True

    def state_dict(self) -> Dict:
        return {
            "counter": self.counter,
            "best_metric": self.best_metric,
            "early_stop": self.early_stop,
        }

    def load_state_dict(self, state: Dict) -> None:
        self.counter = state.get("counter", 0)
        self.best_metric = state.get("best_metric", None)
        self.early_stop = state.get("early_stop", False)