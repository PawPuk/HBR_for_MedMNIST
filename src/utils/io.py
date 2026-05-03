import os
import pickle
from typing import Dict, List, Tuple, Union

from src.config.config import ROOT


def load_results(path: str):
    """Load results."""
    with open(path, 'rb') as file:
        return pickle.load(file)


def load_previous_hardness_estimates(path: str) -> Union[Dict, Dict[Tuple[int, int], Dict[str, List[float]]]]:
    """Loads the hardness estimates, if they have been computed before, or return an empty Dictionary otherwise."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        prior_hardness_estimates = load_results(path)
        print(f'{path} exists - extended hardness estimates.')
        return prior_hardness_estimates
    else:
        print(f"{path} does not exist or is empty. Initializing new data.")
        return {}


def save_results(hardness_estimates: Dict[Tuple[int, int], Dict[str, List[float]]], dataset_model_id: Tuple[int, int],
                 dataset_name: str, split: str, suffix: str = ""):
    """
    The purpose of this function is to enable easier generation of results. If we already spent a lot of
    resources on training an ensemble, we don't want it to go to waste just because the ensemble is not large
    enough. We want to add more models to the ensemble rather than have to retrain it from scratch.
    """
    hardness_save_dir = os.path.join(ROOT, "Results", dataset_name)
    os.makedirs(hardness_save_dir, exist_ok=True)

    # Build filename with suffix if provided
    filename = f'{split}_hardness_estimates_{suffix}.pkl'
    path = os.path.join(hardness_save_dir, filename)

    old_hardness_estimates = load_previous_hardness_estimates(path)
    old_hardness_estimates[dataset_model_id] = hardness_estimates[dataset_model_id]

    with open(path, "wb") as file:
        print(f'Saving updated hardness estimates to {path}')
        pickle.dump(old_hardness_estimates, file)
