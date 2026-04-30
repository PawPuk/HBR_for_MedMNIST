import argparse
import os
import pickle
from typing import List

import numpy as np

from src.config.config import ROOT
from src.utils.io import load_results


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
        splits = [
            ('training', 'training_hardness_estimates.pkl'),
            ('validation', 'validation_hardness_estimates.pkl'),
            ('test', 'test_hardness_estimates.pkl')
        ]

        for split_name, out_filename in splits:
            save_path = os.path.join(ROOT, f'Results/final/{dataset_name}')
            os.makedirs(save_path, exist_ok=True)
            # Load the corresponding hardness estimates file
            hardness_path = os.path.join(ROOT, f'Results/{dataset_name}/{split_name}_hardness_estimates.pkl')
            if not os.path.exists(hardness_path):
                print(f"⚠ Warning: {hardness_path} not found, skipping {split_name} split.")
                continue

            hardness_estimates = load_results(hardness_path)
            n_models_total = len(hardness_estimates)
            hardness_over_models = [hardness_estimates[(0, model_id)]['DataIQ'] for model_id in range(n_models_total)]
            final_hardness = np.mean(np.array(hardness_over_models[:n_models_total]), axis=0)

            with open(os.path.join(save_path, out_filename), 'wb') as f:
                pickle.dump(final_hardness, f)
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
