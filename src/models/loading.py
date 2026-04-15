import os
from typing import List

from src.config.config import ROOT


def load_baseline_models(dataset_name: str, num_epochs: int) -> List[str]:
    """Loads the baseline models."""
    models_dir = os.path.join(ROOT, "Models/")
    full_dataset_dir = os.path.join(models_dir, "none", dataset_name)
    model_paths = []
    if os.path.exists(full_dataset_dir):
        for file in os.listdir(full_dataset_dir):
            if file.endswith(".pth") and f"_epoch_{num_epochs}" in file:
                model_paths.append(os.path.join(full_dataset_dir, file))
    return model_paths
