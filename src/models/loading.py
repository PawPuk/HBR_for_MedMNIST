import os

from src.config.config import ROOT


def load_baseline_models(dataset_name: str, num_epochs: int):
    """Loads the baseline models."""
    models_dir = os.path.join(ROOT, "Models/", dataset_name)
    model_paths = []
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith(".pth") and f"_epoch_{num_epochs}" in file:
                model_path = os.path.join(models_dir, file)
                model_paths.append(model_path)

    return model_paths
