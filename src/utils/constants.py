"""Global constants for CloudSEN12 cloud segmentation."""

from typing import Dict, List

SENTINEL_BANDS: List[str] = [
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B10", "B11", "B12"
]

CLASS_NAMES: List[str] = ["Clear", "Thick Cloud", "Thin Cloud", "Cloud Shadow"]

METRIC_NAMES: List[str] = [
    "F1-Score",
    "Precision",
    "Recall",
    "Omission Error",
    "Commission Error"
]

EXPERIMENTS: Dict[str, Dict[str, List[int]]] = {
    "cloud/no cloud": {"pos": [1, 2]},
    "cloud shadow": {"pos": [3]},
    "valid/invalid": {"pos": [0]},
}

NUM_CLASSES: int = 4