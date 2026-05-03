from collections import defaultdict
import os
from typing import List

from src.config.config import ROOT
from src.utils.io import load_previous_hardness_estimates


def get_latest_model_index(dataset_name: str, split: str, max_dataset_count: int, suffix: str) -> List[int]:
    """
    Find the latest trained model index for each dataset version by reading the hardness estimates pickle file.
    This makes it possible to resume training without retraining existing models.

    Args:
        dataset_name: Name of the dataset (used to locate the results' folder)
        split: 'training', 'validation' or 'test'
        max_dataset_count: Number of dataset versions to return indices for
                          (only dataset 0 is used in the current module)
        suffix: differentiates between hardness estimates on real and synthetic data
                (and between different synthetic data)

    Returns:
        List of the latest model indices for dataset indices 0..max_dataset_count-1.
        Returns -1 for a dataset if no models exist yet.
    """
    hardness_save_dir = os.path.join(ROOT, "Results", dataset_name)
    path = os.path.join(hardness_save_dir, f'{split}_hardness_estimates_{suffix}.pkl')

    # Load existing hardness estimates (empty dict if file doesn't exist)
    hardness_estimates = load_previous_hardness_estimates(path)

    # Find the maximum model index for each dataset index
    max_indices = defaultdict(lambda: -1)
    for (dataset_idx, model_idx) in hardness_estimates.keys():
        if model_idx > max_indices[dataset_idx]:
            max_indices[dataset_idx] = model_idx

    return [max_indices[i] for i in range(max_dataset_count)]
