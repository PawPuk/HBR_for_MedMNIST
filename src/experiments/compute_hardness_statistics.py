import argparse
import glob
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
    class_counts = [0 for _ in range(num_classes)]
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
        hardness_over_models = [hardness_estimates[model_id][est] for model_id in range(n_models_total)]

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


def process_real_data(dataset_names: List[str], skip_existing: bool = True):
    """Original real‑data processing."""
    for dataset_name in dataset_names:
        base_dir = os.path.join(ROOT, f'Results/{dataset_name}')
        if not os.path.isdir(base_dir):
            print(f"⚠ Directory {base_dir} not found, skipping {dataset_name}")
            continue

        hardness_path = os.path.join(base_dir, f"test_hardness_estimates_real.pkl")
        if not os.path.exists(hardness_path):
            print(f"⚠ No real hardness estimate files found for {dataset_name}")
            continue

        print(f"\n{'=' * 60}\nDataset: {dataset_name} (real)\n{'=' * 60}")

        out_filename = f"test_hardness_data_real.pkl"
        save_path = os.path.join(base_dir, out_filename)
        if skip_existing and os.path.exists(save_path):
            print(f"✓ Real data already processed (skipping)")
            continue

        hardness_path = os.path.join(ROOT, f'Results/{dataset_name}/test_hardness_estimates_real.pkl')
        loader, _ = load_dataset(dataset_name, 'test')
        hardness_estimates = load_results(hardness_path)
        data = compute_hardness_data_for_split(dataset_name, 'test', loader, hardness_estimates)
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)


def process_synthetic_data(dataset_names: List[str], skip_existing: bool = True):
    """Process synthetic hardness estimates for all discovered mask percentages."""
    for dataset_name in dataset_names:
        base_dir = os.path.join(ROOT, f'Results/{dataset_name}')
        if not os.path.isdir(base_dir):
            print(f"⚠ Directory {base_dir} not found, skipping {dataset_name}")
            continue

        # Find all hardness estimate files for this dataset.
        # Pattern: {split}_hardness_estimates_syn*.pkl
        all_est_files = glob.glob(os.path.join(base_dir, '*_hardness_estimates_syn*.pkl'))

        # Group by split and extract masking percentage
        split_to_suffixes = {}
        for fpath in all_est_files:
            basename = os.path.basename(fpath)
            # format: {split}_hardness_estimates_syn{masking_percentage}.pkl
            parts = basename.split('_hardness_estimates_syn')
            split = parts[0]  # 'train', 'val', or 'test'
            masking_percentage = parts[1].replace('.pkl', '')  # '0.25', '0.5', '0.75', '1.0'
            split_to_suffixes.setdefault(split, set()).add(masking_percentage)

        if not split_to_suffixes:
            print(f"⚠ No synthetic hardness estimate files found for {dataset_name}")
            continue

        # For each mask percentage (same across splits), process all three splits
        all_suffixes = set()
        for suffixes in split_to_suffixes.values():
            all_suffixes.update(suffixes)

        for mask_suffix in sorted(all_suffixes):
            print(f"\n{'=' * 60}")
            print(f"Dataset: {dataset_name} | Synthetic mask: {mask_suffix}")
            print(f"{'=' * 60}")

            masking_percentage = float(mask_suffix)

            # Process each split
            for split_name in ['train', 'val', 'test']:
                # Output file for hardness data
                out_filename = f"{split_name}_hardness_data_syn{mask_suffix}.pkl"
                save_path = os.path.join(base_dir, out_filename)
                if skip_existing and os.path.exists(save_path):
                    print(f"✓ {split_name} split already processed (skipping)")
                    continue

                # Input hardness estimates file
                hardness_path = os.path.join(base_dir, f"{split_name}_hardness_estimates_syn{mask_suffix}.pkl")
                if not os.path.exists(hardness_path):
                    print(f"⚠ Warning: {hardness_path} not found, skipping {split_name} split.")
                    continue

                print(f"\n🔄 Processing {split_name} split (synthetic, mask={mask_suffix})...")
                loader, _ = load_dataset(dataset_name, split_name, True, masking_percentage)
                hardness_estimates = load_results(hardness_path)
                data = compute_hardness_data_for_split(dataset_name, split_name, loader, hardness_estimates)
                with open(save_path, 'wb') as f:
                    pickle.dump(data, f)


def compute_all_hardness_data(dataset_names: List[str] = None, synthetic: bool = False):
    """Main dispatcher: process real or synthetic data."""
    all_datasets = [
        'bloodmnist', 'pneumoniamnist', 'dermamnist', 'pathmnist', 'chestmnist', 'octmnist',
        'tissuemnist', 'organamnist', 'organcmnist', 'organsmnist', 'breastmnist', 'retinamnist'
    ]
    if dataset_names is None:
        dataset_names = all_datasets

    print(f"\n{'=' * 60}")
    print(f"Processing {len(dataset_names)} datasets (mode: {'synthetic' if synthetic else 'real'})")
    print(f"{'=' * 60}\n")

    if synthetic:
        process_synthetic_data(dataset_names)
    else:
        process_real_data(dataset_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process hardness data for MedMNIST datasets')
    parser.add_argument('--dataset_name', type=str, nargs='+', required=False,
                        help='List of dataset names to process. If not provided, processes all MedMNIST datasets.',
                        choices=['bloodmnist', 'pneumoniamnist', 'dermamnist', 'pathmnist', 'chestmnist',
                                 'octmnist', 'tissuemnist', 'organamnist', 'organcmnist', 'organsmnist',
                                 'breastmnist', 'retinamnist'])
    parser.add_argument('--synthetic', action='store_true', default=False,
                        help='Process synthetic hardness estimates (auto‑discovers all mask percentages).')
    args = parser.parse_args()

    compute_all_hardness_data(args.dataset_name if args.dataset_name else None, synthetic=args.synthetic)
