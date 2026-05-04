import os
from typing import List

from src.config.config import ROOT
from src.utils.io import load_previous_hardness_estimates


def get_latest_model_index(dataset_name: str, split: str, suffix: str) -> List[int]:
    """
    Find the latest trained model index by reading the hardness estimates pickle file.
    This makes it possible to resume training without retraining existing models.

    Args:
        dataset_name: Name of the dataset (used to locate the results' folder)
        split: 'training', 'validation' or 'test'
        suffix: differentiates between hardness estimates on real and synthetic data
                (and between different synthetic data)

    Returns:
        The index of the latest model trained for the ensemble. Returns -1 for a dataset if no models exist yet.
    """
    hardness_save_dir = os.path.join(ROOT, "Results", dataset_name)
    path = os.path.join(hardness_save_dir, f'{split}_hardness_estimates_{suffix}.pkl')

    # Load existing hardness estimates (empty dict if file doesn't exist)
    hardness_estimates = load_previous_hardness_estimates(path)

    # Find the maximum model index
    max_index = -1
    for model_idx in hardness_estimates.keys():
        if model_idx > max_index:
            max_index = model_idx

    return max_index
