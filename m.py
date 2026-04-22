import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: str):
    """Load results."""
    with open(path, 'rb') as file:
        return pickle.load(file)


dataset_name = 'bloodmnist'

num_models_for_hardness = 16
hardness_estimates = load_results(f'Results/{dataset_name}/hardness_estimates.pkl')
class_level_performances = load_results(f'Results/{dataset_name}/class_level_performances.pkl')

for hardness_estimator in ['AUM', 'DataIQ', 'Forgetting']:
    # Load and preprocess hardness estimates
    hardness_over_models = [hardness_estimates[(0, model_id)][hardness_estimator]
                            for model_id in range(len(hardness_estimates))]
    n = min(len(hardness_estimates), num_models_for_hardness)
    final_hardness_estimates = list(np.mean(np.array(hardness_over_models[:n]), axis=0))

    # Load and preprocess performance estimates

    all_accuracies = np.array([class_level_performances[i]['accuracy'] for i in range(len(class_level_performances))])
    all_aucs = np.array([class_level_performances[i]['auc'] for i in range(len(class_level_performances))])
    avg_accuracy_per_class = np.mean(all_accuracies, axis=0)
    avg_auc_per_class = np.mean(all_aucs, axis=0)
    print([f'{x:.15f}' for x in avg_auc_per_class])
    print([f'{x:.15f}' for x in avg_accuracy_per_class])

    # Sort the hardness estimates in ascending order
    sorted_hardness = np.sort(final_hardness_estimates)

    # Create a new figure for this estimator
    plt.figure(figsize=(10, 6))

    plt.plot(range(len(sorted_hardness)), sorted_hardness, 'b-', linewidth=2)
    plt.scatter(range(len(sorted_hardness)), sorted_hardness, c='blue', s=20, alpha=0.6)

    # Customize the plot
    plt.xlabel('Sample Index (sorted by hardness)', fontsize=12)
    plt.ylabel('Hardness Score', fontsize=12)
    plt.title(f'Distribution of {hardness_estimator} Hardness Estimates', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Add some statistics as text on the plot
    stats_text = f'Mean: {np.mean(sorted_hardness):.3f}\nStd: {np.std(sorted_hardness):.3f}\n' \
                 f'Min: {np.min(sorted_hardness):.3f}\nMax: {np.max(sorted_hardness):.3f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Optional: Save the figures
    save_path = f'Figures/{dataset_name}/'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'hardness_distribution_{hardness_estimator}.png'),
                dpi=150, bbox_inches='tight')

    plt.tight_layout()
    # plt.show()
