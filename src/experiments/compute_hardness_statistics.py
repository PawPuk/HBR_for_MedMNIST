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


def compute_hardness_data_for_split(dataset_name: str, split: str, loader: DataLoader,
                                    hardness_estimates: dict) -> dict:
    """Compute all hardness data for a single split (train/val/test)."""
    config = get_config(dataset_name)
    num_classes = config['num_classes']

    class_cardinalities = get_class_cardinalities(loader, num_classes)

    thresholds = np.arange(5, 35, 10)
    n_models_total = len(hardness_estimates)
    model_counts = list(range(1, n_models_total + 1))

    # Storage structures
    all_hardness_by_class = {}          # {estimator: {num_models: hardness_by_class}}
    all_hard_samples = {}               # {estimator: {num_models: {threshold: hard_indices}}}
    all_hard_samples_by_class = {}      # {estimator: {num_models: {threshold: per_class_hard_indices}}}
    all_final_hardness = {}             # {estimator: {num_models: {final_hardness}}}

    estimators = ['AUM', 'DataIQ', 'Forgetting']

    for est in estimators:
        # Pre‑load hardness over models (list of lists)
        hardness_over_models = [hardness_estimates[(0, model_id)][est] for model_id in range(n_models_total)]

        all_hardness_by_class[est] = {}
        all_hard_samples[est] = {}
        all_hard_samples_by_class[est] = {}
        all_final_hardness[est] = {}

        for num_models in tqdm(model_counts, desc=f'{split} - {est} over model counts'):
            # Average over `num_models` models
            final_hardness = list(np.mean(np.array(hardness_over_models[:num_models]), axis=0))
            hardness_by_class = group_hardness_by_class(loader, final_hardness, num_classes)
            all_hardness_by_class[est][num_models] = hardness_by_class

            hard_samples_for_threshold = {}
            hard_samples_by_class_for_threshold = {}

            for thr in thresholds:
                total_n_samples = len(final_hardness)
                target_n_hard_samples = int(total_n_samples * thr / 100)
                if est == 'AUM':
                    # AUM: high values = easy, low values = hard
                    sorted_indices = sorted(range(total_n_samples), key=lambda i: final_hardness[i])
                else:
                    sorted_indices = sorted(range(total_n_samples), key=lambda i: final_hardness[i], reverse=True)
                hard_indices = sorted_indices[:target_n_hard_samples]
                hard_samples_for_threshold[thr] = hard_indices

                # Group by class
                per_class_hard_indices = [[] for _ in range(num_classes)]
                hard_idx_set = set(hard_indices)  # For faster lookup
                for _, labels, indices in loader:
                    for label, idx in zip(labels, indices):
                        if idx.item() in hard_idx_set:
                            per_class_hard_indices[label].append(idx.item())
                hard_samples_by_class_for_threshold[thr] = per_class_hard_indices

            all_hard_samples[est][num_models] = hard_samples_for_threshold
            all_hard_samples_by_class[est][num_models] = hard_samples_by_class_for_threshold
            all_final_hardness[est][num_models] = final_hardness

    # Assemble data dictionary for this split
    data = {
        'class_cardinalities': class_cardinalities,
        'thresholds': thresholds,
        'model_counts': model_counts,
        'estimators': estimators,
        'all_hardness_by_class': all_hardness_by_class,
        'all_hard_samples': all_hard_samples,
        'all_hard_samples_by_class': all_hard_samples_by_class,
        'all_final_hardness': all_final_hardness,
    }
    return data


def compute_all_hardness_data(dataset_names: List[str] = None):
    """Compute hardness statistics for all specified datasets (or all MedMNIST datasets if None).
    For each dataset, processes train, validation and test splits separately,
    skipping any split for which the output file already exists."""

    # All available MedMNIST datasets
    all_datasets = [
        'bloodmnist', 'pneumoniamnist', 'dermamnist', 'pathmnist', 'octmnist', 'tissuemnist',
        'organamnist', 'organcmnist', 'organsmnist', 'breastmnist', 'retinamnist'
    ]

    # If no specific datasets provided, process all
    if dataset_names is None:
        dataset_names = all_datasets

    print(f"\n{'=' * 60}")
    print(f"Processing {len(dataset_names)} datasets: {', '.join(dataset_names)}")
    print(f"{'=' * 60}\n")

    for dataset_idx, dataset_name in enumerate(dataset_names, 1):
        print(f"\n{'=' * 60}")
        print(f"Dataset {dataset_idx}/{len(dataset_names)}: {dataset_name}")
        print(f"{'=' * 60}")

        # Load the dataset once to get the three loaders
        train_loader, _, val_loader, _, test_loader, _ = load_dataset(dataset_name)

        splits = [
            ('training', train_loader, 'training_hardness_data.pkl'),
            ('validation', val_loader, 'validation_hardness_data.pkl'),
            ('test', test_loader, 'test_hardness_data.pkl')
        ]

        for split_name, loader, out_filename in splits:
            save_path = os.path.join(ROOT, f'Results/{dataset_name}/{out_filename}')

            # Skip if output already exists
            if os.path.exists(save_path):
                print(f"✓ {split_name} split already processed (found {save_path})")
                continue

            # Load the corresponding hardness estimates file
            hardness_path = os.path.join(ROOT, f'Results/{dataset_name}/{split_name}_hardness_estimates.pkl')
            if not os.path.exists(hardness_path):
                print(f"⚠ Warning: {hardness_path} not found, skipping {split_name} split.")
                continue

            hardness_estimates = load_results(hardness_path)

            print(f"\n🔄 Processing {split_name} split...")
            data = compute_hardness_data_for_split(dataset_name, split_name, loader, hardness_estimates)

            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"✓ Saved hardness data to {save_path}")

        print(f"\n✓ Completed dataset: {dataset_name}")

    print(f"\n{'=' * 60}")
    print("Processing complete!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process hardness data for MedMNIST datasets')
    parser.add_argument('--dataset_name', type=str, nargs='+', required=False,
                        help='List of dataset names to process. If not provided, processes all MedMNIST datasets.',
                        choices=['bloodmnist', 'pneumoniamnist', 'dermamnist', 'pathmnist', 'chestmnist',
                                 'octmnist', 'tissuemnist', 'organamnist', 'organcmnist', 'organsmnist',
                                 'breastmnist', 'retinamnist'])
    args = parser.parse_args()

    # If datasets are provided, use them; otherwise process all
    compute_all_hardness_data(args.dataset_name if args.dataset_name else None)
