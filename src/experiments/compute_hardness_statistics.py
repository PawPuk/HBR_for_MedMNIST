import argparse
import os
import pickle
from typing import List

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.config import ROOT, get_config
from src.data.loading import load_dataset
from src.utils.io import load_results


def get_class_cardinalities(loader: DataLoader, num_classes: int) -> List[int]:
    class_counts = [0] * num_classes
    for _, labels, _ in loader:
        for label in labels:
            class_counts[label] += 1
    return class_counts


def group_hardness_by_class(loader: DataLoader, hardness_estimates: list, num_classes: int) -> List[List[float]]:
    hardness_by_class = [[] for _ in range(num_classes)]
    for _, labels, indices in loader:
        for label, idx in zip(labels, indices):
            hardness_by_class[label].append(hardness_estimates[idx])
    return hardness_by_class


def compute_all_hardness_data(dataset_name: str):
    """Compute all hardness data for all ensemble sizes and save to disk."""
    hardness_estimates = load_results(f'{ROOT}/Results/{dataset_name}/hardness_estimates.pkl')
    config = get_config(dataset_name)
    num_classes = config['num_classes']
    training_loader, _, _, _ = load_dataset(dataset_name)

    class_cardinalities = get_class_cardinalities(training_loader, num_classes)

    thresholds = np.arange(5, 35, 10)
    n_models_total = len(hardness_estimates)
    model_counts = list(range(1, n_models_total + 1))

    # Storage structures
    all_hardness_by_class = {}          # {estimator: {num_models: hardness_by_class}}
    all_hard_samples = {}               # {estimator: {num_models: {threshold: hard_indices}}}
    all_hard_samples_by_class = {}      # {estimator: {num_models: {threshold: per_class_hard_indices}}}

    estimators = ['AUM', 'DataIQ', 'Forgetting']

    for est in estimators:
        # Pre‑load hardness over models (list of lists)
        hardness_over_models = [hardness_estimates[(0, model_id)][est] for model_id in range(n_models_total)]

        all_hardness_by_class[est] = {}
        all_hard_samples[est] = {}
        all_hard_samples_by_class[est] = {}

        for num_models in tqdm(model_counts, desc=f'Computing {est} over model counts'):
            # Average over `num_models` models
            final_hardness = list(np.mean(np.array(hardness_over_models[:num_models]), axis=0))
            hardness_by_class = group_hardness_by_class(training_loader, final_hardness, num_classes)
            all_hardness_by_class[est][num_models] = hardness_by_class

            hard_samples_for_threshold = {}
            hard_samples_by_class_for_threshold = {}

            for thr in thresholds:
                # For AUM high values correspond to easy samples (unlike for Forgetting or DataIQ)
                if est == 'AUM':
                    percentile_val = np.percentile(final_hardness, thr)
                    hard_indices = [i for i, h in enumerate(final_hardness) if h <= percentile_val]
                else:
                    percentile_val = np.percentile(final_hardness, 100 - thr)
                    hard_indices = [i for i, h in enumerate(final_hardness) if h >= percentile_val]
                hard_samples_for_threshold[thr] = hard_indices

                # Group by class
                per_class_hard_indices = [[] for _ in range(num_classes)]
                hard_idx_set = set(hard_indices)  # For faster lookup
                for _, labels, indices in training_loader:
                    for label, idx in zip(labels, indices):
                        if idx.item() in hard_idx_set:
                            per_class_hard_indices[label].append(idx.item())
                hard_samples_by_class_for_threshold[thr] = per_class_hard_indices

            all_hard_samples[est][num_models] = hard_samples_for_threshold
            all_hard_samples_by_class[est][num_models] = hard_samples_by_class_for_threshold

    # Save everything
    data = {
        'class_cardinalities': class_cardinalities,
        'thresholds': thresholds,
        'model_counts': model_counts,
        'estimators': estimators,
        'all_hardness_by_class': all_hardness_by_class,
        'all_hard_samples': all_hard_samples,
        'all_hard_samples_by_class': all_hard_samples_by_class,
    }

    save_path = os.path.join(ROOT, f'Results/{dataset_name}/hardness_data.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved hardness data to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['dermamnist', 'bloodmnist', 'pneumoniamnist'])
    args = parser.parse_args()
    compute_all_hardness_data(args.dataset_name)
