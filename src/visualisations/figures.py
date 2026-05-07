import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def plot_hard_sample_pairwise_overlap(all_hard_samples: Dict[str, Dict[int, List[int]]],
                                      thresholds: List[int],
                                      dataset_name: str,
                                      save_path: str):
    """
    Creates a single plot with four lines showing pairwise and triple overlap of hard sample indices.
    Y-axis shows percentage overlap relative to threshold size.

    Args:
        all_hard_samples: dict mapping estimator -> {threshold_pct: list_of_indices}
        thresholds: list of percentile thresholds (e.g., [10,20,30,40])
        dataset_name: for title and filename
        save_path: directory to save figure
    """
    estimators = list(all_hard_samples.keys())

    # Calculate total number of samples per threshold (same for all estimators)
    total_samples_per_threshold = {}
    for threshold in thresholds:
        total_samples_per_threshold[threshold] = len(all_hard_samples[estimators[0]][threshold])

    # Prepare parameters for plotting.
    pairs = [(estimators[0], estimators[1]), (estimators[0], estimators[2]), (estimators[1], estimators[2])]
    pair_labels = [f"{a} ∩ {b}" for a, b in pairs]
    colors = ['blue', 'red', 'green', 'black']
    line_styles = ['-', '--', '-.', ':']

    # Plot pairwise overlaps
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, ((est1, est2), label, color, style) in enumerate(zip(pairs, pair_labels, colors[:-1], line_styles[:-1])):
        overlap_percentages = []
        for pct in thresholds:
            set1 = set(all_hard_samples[est1][pct])
            set2 = set(all_hard_samples[est2][pct])
            overlap_count = len(set1 & set2)
            # Calculate percentage relative to threshold size
            overlap_percentage = (overlap_count / total_samples_per_threshold[pct]) * 100
            overlap_percentages.append(overlap_percentage)

        ax.plot(thresholds, overlap_percentages, marker='o', linewidth=2, markersize=6,
                color=color, linestyle=style, label=label)

    # Plot triple overlap (common to all three estimators)
    triple_overlap_percentages = []
    for pct in thresholds:
        sets = [set(all_hard_samples[est][pct]) for est in estimators]
        triple_overlap_count = len(set.intersection(*sets))
        triple_overlap_percentage = (triple_overlap_count / total_samples_per_threshold[pct]) * 100
        triple_overlap_percentages.append(triple_overlap_percentage)

    ax.plot(thresholds, triple_overlap_percentages, marker='s', linewidth=2.5, markersize=7,
            color=colors[-1], linestyle=line_styles[-1], label='All three estimators (triple overlap)')

    ax.set_xlabel('Top x% hardest samples', fontsize=12)
    ax.set_ylabel('Overlap (%)', fontsize=12)
    ax.set_title(f'{dataset_name}: Overlap of Hardest Samples across Estimators', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    save_file = os.path.join(save_path, f'hard_sample_pairwise_overlap_{dataset_name}.png')
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved pairwise overlap plot to {save_file}")


def plot_class_hard_samples_simple(class_cardinalities: List[int],
                                   hard_samples_by_class_for_estimator: Dict[int, List[List[int]]],
                                   estimator_name: str,
                                   dataset_name: str,
                                   save_path: str,
                                   thresholds: List[int]):
    """
    Creates a simple line plot showing number of hard samples per class for each threshold.
    One figure per estimator.

    Args:
        class_cardinalities: list of ints, total samples per class (for reference)
        hard_samples_by_class_for_estimator: dict mapping estimator -> {threshold_pct: per_class_indices}
        estimator_name: which estimator to plot (e.g., 'AUM', 'DataIQ', 'Forgetting')
        dataset_name: for title and filename
        save_path: directory to save figure
        thresholds: list of percentile thresholds
    """
    num_classes = len(class_cardinalities)
    classes = list(range(num_classes))

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot a line for each threshold
    for threshold in thresholds:
        hard_counts = []
        hard_by_class = hard_samples_by_class_for_estimator[threshold]
        for c in range(num_classes):
            hard_counts.append(len(hard_by_class[c]))

        ax.plot(classes, hard_counts, marker='o', linewidth=2, markersize=6,
                label=f'Top {threshold}% hardest')

    # Add total class cardinalities as a reference line
    ax.plot(classes, class_cardinalities, 'k--', linewidth=2, marker='s',
            label='Total samples (reference)', alpha=0.7)

    # Customize plot
    ax.set_xlabel('Class Label', fontsize=12)
    ax.set_ylabel('Number of Hard Samples', fontsize=12)
    ax.set_title(f'{dataset_name} - {estimator_name}: Hard Samples per Class', fontsize=14)
    ax.set_xticks(classes)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    save_file = os.path.join(save_path, f'class_hard_samples_{estimator_name}.png')
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to {save_file}")


def plot_class_cardinalities_and_hardness(class_cardinalities: List[int],
                                          all_hardness_by_class: Dict[str, List[List[float]]],
                                          dataset_name: str,
                                          save_path: str):
    """
    Creates a figure with three subplots (one per hardness estimator).
    Each subplot shows class cardinalities (bars) and per‑class hardness mean ± std (error bars).
    A single shared legend is placed below the figure.

    Args:
        class_cardinalities: list of ints, length = number of classes.
        all_hardness_by_class: dict mapping estimator name -> list of lists of hardness values per class.
        dataset_name: used for title and filename.
        save_path: directory to save the figure.
    """
    estimators = list(all_hardness_by_class.keys())
    num_estimators = len(estimators)
    num_classes = len(class_cardinalities)
    classes = list(range(num_classes))

    # Prepare data: for each estimator, compute mean and std per class
    means, stds = {}, {}
    for est in estimators:
        hardness_lists = all_hardness_by_class[est]
        est_means, est_stds = [], []
        for c in range(num_classes):
            vals = hardness_lists[c]
            est_means.append(np.mean(vals) if vals else 0.0)
            est_stds.append(np.std(vals) if vals else 0.0)
        means[est] = est_means
        stds[est] = est_stds

    # Create subplots: one row, three columns
    fig, axes = plt.subplots(1, num_estimators, figsize=(5 * num_estimators, 5))

    x = np.arange(num_classes)
    width = 0.35

    # We'll collect handles and labels for a single legend
    legend_handles, legend_labels = [], []

    for idx, est in enumerate(estimators):
        ax = axes[idx]
        # Bar chart: cardinalities
        bars = ax.bar(x - width/2, class_cardinalities, width, color='skyblue', alpha=0.7, label='Cardinality')
        if idx == 0:
            legend_handles.append(bars)
            legend_labels.append('Cardinality')

        ax.set_xlabel('Class Label')
        ax.set_ylabel('Number of Samples', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')

        # Twin axis for hardness
        ax2 = ax.twinx()
        errorbar = ax2.errorbar(x, means[est], yerr=stds[est], fmt='o-', color='red',
                                capsize=5, label='Hardness (mean ± std)', linewidth=2, markersize=6)
        if idx == 0:
            legend_handles.append(errorbar)
            legend_labels.append('Hardness (mean ± std)')

        ax2.set_ylabel('Hardness Estimate', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        ax.set_title(f'{est}')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)

    # Single legend placed below the figure
    fig.legend(legend_handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.1),
               ncol=2, fontsize=10)

    fig.suptitle(f'{dataset_name}: Class Cardinalities and Hardness per Estimator', fontsize=14)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Make room for the legend
    save_file = os.path.join(save_path, f'class_cardinalities_hardness_all_estimators.png')
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved combined figure to {save_file}")


def plot_relative_stability_of_estimates(all_hard_samples: Dict[str, Dict[int, Dict[int, List[int]]]],
                                         thresholds: List[int], model_counts: List[int], estimator: str,
                                         dataset_name: str, save_path: str):
    """
    Plot how the hard sample set changes with ensemble size.
    For each threshold, shows the overlap (as percentage) with the final set (using all models).
    """
    final_models = max(model_counts)
    colors = ['green', 'blue', 'brown']

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, thr in enumerate(thresholds):
        final_set = set(all_hard_samples[estimator][final_models][thr])
        overlaps = []
        for k in model_counts:
            cur_set = set(all_hard_samples[estimator][k][thr])
            overlap = len(cur_set & final_set) / len(final_set) * 100
            overlaps.append(overlap)

        ax.plot(model_counts, overlaps, marker='o', linewidth=2, markersize=6,
                color=colors[idx], label=f'Top {thr}% hardest')

    ax.set_xlabel('Number of models in ensemble', fontsize=12)
    ax.set_ylabel('Overlap with final set (%)', fontsize=12)
    ax.set_title(f'{dataset_name} – {estimator}\nStability of hard sample set with ensemble size', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    save_file = os.path.join(save_path, f'relative_stability_{estimator}.png')
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved relative stability plot to {save_file}")


def plot_consecutive_stability(all_hard_samples, thresholds, model_counts, estimator, dataset_name, save_path):
    """
    Plot how the hard sample set changes between consecutive ensemble sizes.
    Shows overlap (as percentage) between hard samples from i models and i+1 models.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Define a colormap for the thresholds
    colors = ['green', 'blue', 'brown']
    x_ticks = None

    for idx, thr in enumerate(thresholds):
        consecutive_overlaps = []

        # Compare i models vs i+1 models for i = 1 to (max-1)
        for i in range(len(model_counts) - 1):
            set_i = set(all_hard_samples[estimator][model_counts[i]][thr])
            set_i_plus_1 = set(all_hard_samples[estimator][model_counts[i + 1]][thr])

            if set_i:
                overlap = len(set_i & set_i_plus_1) / len(set_i) * 100
            else:
                overlap = 0
            consecutive_overlaps.append(overlap)

        # Plot using the number of models as x-axis (first point corresponds to 1->2 models)
        x_ticks = [f"{model_counts[i]}→{model_counts[i + 1]}" for i in range(len(model_counts) - 1)]
        ax.plot(range(len(consecutive_overlaps)), consecutive_overlaps,
                marker='o', linewidth=2, markersize=6,
                color=colors[idx], label=f'Top {thr}% hardest')

    ax.set_xlabel('Ensemble size transition', fontsize=12)
    ax.set_ylabel('Overlap between consecutive ensembles (%)', fontsize=12)
    ax.set_title(f'{dataset_name} – {estimator}\nConsecutive stability of hard sample set', fontsize=14)
    ax.set_xticks(range(len(x_ticks)))
    ax.set_xticklabels(x_ticks, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    save_file = os.path.join(save_path, f'consecutive_stability_{estimator}.png')
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved consecutive stability plot to {save_file}")
