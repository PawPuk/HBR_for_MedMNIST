import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.config.config import DEVICE, get_config, ROOT
from src.data.loading import load_dataset
from src.hardness.estimators import compute_margin_for_loader
from src.models.loading import load_baseline_models
from src.models.neural_networks import ResNet18


def main(dataset_name: str):
    config = get_config(dataset_name)

    model_paths = load_baseline_models(dataset_name, config['num_epochs'])
    real_val_loader, _ = load_dataset(dataset_name, 'val', shuffle=False)
    real_test_loader, _ = load_dataset(dataset_name, 'test', shuffle=False)

    mask_percentages = [0.00, 0.25, 0.50, 0.75, 1.00]
    results = {}   # mask -> {'val': margins, 'test': margins, 'syn': margins}

    for mask_percentage in tqdm(mask_percentages, desc='Iterating masks'):
        syn_test_loader, _ = load_dataset(dataset_name, 'test', True, mask_percentage, shuffle=False)

        margins_sum = {'val': None, 'test': None, 'syn': None}

        for model_path in model_paths:
            model_state = torch.load(model_path)
            model = ResNet18(in_channels=config['n_channels'], num_classes=config['num_classes']).to(DEVICE)
            model.load_state_dict(model_state)
            model.eval()

            val_margins = compute_margin_for_loader(model, real_val_loader, DEVICE)
            test_margins = compute_margin_for_loader(model, real_test_loader, DEVICE)
            syn_margins = compute_margin_for_loader(model, syn_test_loader, DEVICE)

            if margins_sum['val'] is None:
                margins_sum['val'] = val_margins
                margins_sum['test'] = test_margins
                margins_sum['syn'] = syn_margins
            else:
                margins_sum['val'] += val_margins
                margins_sum['test'] += test_margins
                margins_sum['syn'] += syn_margins

        margins_avg = {k: v / len(model_paths) for k, v in margins_sum.items()}
        results[mask_percentage] = margins_avg

    global_min = float('inf')
    global_max = float('-inf')
    for mask, margins_avg in results.items():
        for key in ['val', 'test', 'syn']:
            margins = margins_avg[key]
            global_min = min(global_min, np.min(margins))
            global_max = max(global_max, np.max(margins))
    padding = (global_max - global_min) * 0.05
    global_min -= padding
    global_max += padding

    n_masks = len(mask_percentages)
    n_cols = 3
    n_rows = (n_masks + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()

    for idx, mask in enumerate(mask_percentages):
        ax = axes[idx]
        margins_avg = results[mask]

        for key, label in [('val', 'Real val'), ('test', 'Real test'), ('syn', 'Syn test')]:
            margins = margins_avg[key]
            sorted_margins = np.sort(margins)
            quantiles = np.linspace(0, 1, len(sorted_margins))
            ax.plot(quantiles, sorted_margins, linewidth=2, label=label)

        ax.set_xlabel('Quantile')
        ax.set_ylabel('Margin')
        ax.set_ylim(global_min, global_max)
        ax.set_title(f'Synthetic mask = {int(mask*100)}%')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide any unused subplots (if n_masks < n_rows * n_cols)
    for idx in range(n_masks, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'Margin distribution – {dataset_name}', fontsize=14)
    plt.tight_layout()

    fig_dir = os.path.join(ROOT, f'Figures/{dataset_name}')
    os.makedirs(fig_dir, exist_ok=True)
    save_path = os.path.join(fig_dir, 'margins_all_masks.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load models for specified pruning strategy and dataset")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of the dataset (e.g., 'CIFAR10')")

    args = parser.parse_args()
    main(args.dataset_name)
