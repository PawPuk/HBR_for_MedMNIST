import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.config.config import DEVICE, get_config, ROOT
from src.data.loading import load_dataset
from src.hardness.estimators import accumulate_margins_with_labels
from src.models.loading import load_baseline_models
from src.models.neural_networks import ResNet18


def main(dataset_name: str):
    config = get_config(dataset_name)
    num_classes = config['num_classes']

    model_paths = load_baseline_models(dataset_name, config['num_epochs'])
    real_val_loader, _ = load_dataset(dataset_name, 'val', shuffle=False)
    real_test_loader, _ = load_dataset(dataset_name, 'test', shuffle=False)

    mask_percentages = [0.00, 0.25, 0.50, 0.75, 1.00]
    # Store for each mask:
    #   results[mask]['val'] = (margins_avg, labels)
    #   results[mask]['test'] = (...)
    #   results[mask]['syn'] = (...)
    results = {}

    for mask_percentage in tqdm(mask_percentages, desc='Iterating masks'):
        syn_test_loader, _ = load_dataset(dataset_name, 'test', True, mask_percentage, shuffle=False)

        sum_val, sum_test, sum_syn, labels_val, labels_test, labels_syn = None, None, None, None, None, None

        for model_path in model_paths:
            model = ResNet18(in_channels=config['n_channels'], num_classes=config['num_classes']).to(DEVICE)
            model.load_state_dict(torch.load(model_path))
            model.eval()

            val_m, val_l = accumulate_margins_with_labels(model, real_val_loader, DEVICE)
            test_m, test_l = accumulate_margins_with_labels(model, real_test_loader, DEVICE)
            syn_m, syn_l = accumulate_margins_with_labels(model, syn_test_loader, DEVICE)

            if sum_val is None:
                sum_val = val_m
                sum_test = test_m
                sum_syn = syn_m
                labels_val = val_l
                labels_test = test_l
                labels_syn = syn_l
            else:
                sum_val += val_m
                sum_test += test_m
                sum_syn += syn_m

        # Average over models
        num_models = len(model_paths)
        results[mask_percentage] = {
            'val': (sum_val / num_models, labels_val),
            'test': (sum_test / num_models, labels_test),
            'syn': (sum_syn / num_models, labels_syn),
        }

    # ------------------ Overall figure (all classes together) ------------------
    plt.figure(figsize=(10, 6))
    # Real validation (use any mask, e.g., first mask; they are all identical)
    val_margins, _ = results[mask_percentages[0]]['val']
    sorted_val = np.sort(val_margins)
    quantiles_val = np.linspace(0, 1, len(sorted_val))
    plt.plot(quantiles_val, sorted_val, linewidth=2, label='Real validation')

    # Real test (also the same for all masks)
    test_margins, _ = results[mask_percentages[0]]['test']
    sorted_test = np.sort(test_margins)
    quantiles_test = np.linspace(0, 1, len(sorted_test))
    plt.plot(quantiles_test, sorted_test, linewidth=2, label='Real test')

    # Synthetic masks
    for mask in mask_percentages:
        syn_margins, _ = results[mask]['syn']
        sorted_syn = np.sort(syn_margins)
        quantiles_syn = np.linspace(0, 1, len(sorted_syn))
        plt.plot(quantiles_syn, sorted_syn, linewidth=1.5, linestyle='--', label=f'Syn {int(mask*100)}%')

    plt.xlabel('Quantile')
    plt.ylabel('Margin')
    plt.title(f'Margin distributions – {dataset_name} (all classes)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig_dir = os.path.join(ROOT, f'Figures/{dataset_name}')
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, 'margin_overall.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ------------------ Per‑class figure ------------------
    class_data = {c: {'val': [], 'test': [], 'syn': {m: [] for m in mask_percentages}} for c in range(num_classes)}

    test_margins, test_labels = results[mask_percentages[0]]['test']
    val_margins, val_labels = results[mask_percentages[0]]['val']
    for c in range(num_classes):
        class_data[c]['test'] = test_margins[test_labels == c]
        class_data[c]['val'] = val_margins[val_labels == c]
        for mask in mask_percentages:
            syn_margins, syn_labels = results[mask]['syn']
            class_data[c]['syn'][mask] = syn_margins[syn_labels == c]

    # Determine subplot grid
    n_cols = min(3, num_classes)
    n_rows = (num_classes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten() if num_classes > 1 else [axes]

    # Compute global y‑limits across all classes to keep subplots comparable
    global_min, global_max = float('inf'), float('-inf')
    for c in range(num_classes):
        for key in ['val', 'test']:
            arr = class_data[c][key]
            if len(arr) > 0:
                global_min = min(global_min, np.min(arr))
                global_max = max(global_max, np.max(arr))
        for mask in class_data[c]['syn']:
            arr = class_data[c]['syn'][mask]
            if len(arr) > 0:
                global_min = min(global_min, np.min(arr))
                global_max = max(global_max, np.max(arr))
    padding = (global_max - global_min) * 0.05
    global_min -= padding
    global_max += padding

    for c in range(num_classes):
        ax = axes[c]
        # Real validation
        val_arr = class_data[c]['val']
        if len(val_arr) > 0:
            sorted_vals = np.sort(val_arr)
            quantiles = np.linspace(0, 1, len(sorted_vals))
            ax.plot(quantiles, sorted_vals, linewidth=2, label='Real val')
        # Real test
        test_arr = class_data[c]['test']
        if len(test_arr) > 0:
            sorted_test = np.sort(test_arr)
            quantiles = np.linspace(0, 1, len(sorted_test))
            ax.plot(quantiles, sorted_test, linewidth=2, label='Real test')
        # Synthetic masks
        for mask, arr in class_data[c]['syn'].items():
            if len(arr) > 0:
                sorted_syn = np.sort(arr)
                quantiles = np.linspace(0, 1, len(sorted_syn))
                ax.plot(quantiles, sorted_syn, linewidth=1, linestyle='--', label=f'Syn {int(mask*100)}%')
        ax.set_title(f'Class {c} (n={len(test_arr)})')
        ax.set_xlabel('Quantile')
        ax.set_ylabel('Margin')
        ax.set_ylim(global_min, global_max)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(num_classes, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'Margin distributions per class – {dataset_name}', fontsize=14)
    plt.tight_layout()
    fig_dir = os.path.join(ROOT, f'Figures/{dataset_name}')
    os.makedirs(fig_dir, exist_ok=True)
    per_class_path = os.path.join(fig_dir, 'margin_per_class.png')
    plt.savefig(per_class_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)

    args = parser.parse_args()
    main(args.dataset_name)
