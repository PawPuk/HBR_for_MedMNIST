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
    real_val_loader, _ = load_dataset(dataset_name, 'val')
    real_test_loader, _ = load_dataset(dataset_name, 'test')

    for mask_percentage in tqdm([0.00, 0.25, 0.50, 0.75, 1.00], desc='Iterating masks'):
        syn_test_loader, _ = load_dataset(dataset_name, 'test', True, mask_percentage)

        margins_sum = {'val': None, 'test': None, 'syn': None}

        for model_path in tqdm(model_paths, desc='Iterating models'):
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

        plt.figure(figsize=(8, 6))

        for key, label in [('val', 'Real val'), ('test', 'Real test'), ('syn', 'Syn test')]:
            margins = margins_avg[key]
            sorted_margins = np.sort(margins)
            # x = percentile (0 to 100) or quantile (0 to 1)
            quantiles = np.linspace(0, 1, len(sorted_margins))
            plt.plot(quantiles, sorted_margins, linewidth=2, label=label)

        plt.xlabel('Quantile (sorted by margin)')
        plt.ylabel('Margin (p_correct - max_other)')
        plt.title(f'Margin distribution – {dataset_name}\nSynthetic mask = {int(mask_percentage*100)}%')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save figure
        fig_dir = os.path.join(ROOT, f'Figures/{dataset_name}/margin_curves')
        os.makedirs(fig_dir, exist_ok=True)
        save_path = os.path.join(fig_dir, f'mask_{int(mask_percentage*100)}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load models for specified pruning strategy and dataset")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of the dataset (e.g., 'CIFAR10')")

    args = parser.parse_args()
    main(args.dataset_name)
