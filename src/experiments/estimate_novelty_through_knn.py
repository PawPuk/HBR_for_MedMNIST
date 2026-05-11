import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

from src.config.config import ROOT
from src.data.loading import load_dataset


def load_all_real_data(dataset_name: str):
    """Load train, val, test real images and flatten them."""
    splits = ['train', 'val', 'test']
    all_images, split_ids = [], []   # 0=train, 1=val, 2=test

    for split_id, split in enumerate(splits):
        loader, _ = load_dataset(dataset_name, split=split, shuffle=False)
        images = [batch_imgs.numpy() for batch_imgs, batch_labels, _ in loader]
        images = np.concatenate(images, axis=0)          # (N_split, C, H, W)
        all_images.append(images)
        split_ids.extend([split_id] * len(images))

    all_images = np.concatenate(all_images, axis=0)      # (N_total, C, H, W)
    split_ids = np.array(split_ids)

    # Flatten: (N_total, C*H*W)
    all_images_flat = all_images.reshape(all_images.shape[0], -1)
    return all_images_flat, split_ids


def load_synthetic_test_data(dataset_name: str, mask_percentage: float):
    """Load synthetic test images for a given masking percentage."""
    loader, _ = load_dataset(dataset_name, split='test', synthetic=True, masking_percentage=mask_percentage,
                             shuffle=False)
    images = [batch_imgs.numpy() for batch_imgs, _, _ in loader]
    images = np.concatenate(images, axis=0)          # (N_synth, C, H, W)
    images_flat = images.reshape(images.shape[0], -1)
    return images_flat


def get_paired_real_test_indices(real_split_ids):
    """
    Returns the indices in the combined real array that belong to the test split.
    The order matches the test loader order.
    """
    return np.where(real_split_ids == 2)[0]   # 2 = test split


def compute_matching_metrics(real_flat, synth_flat, paired_real_indices):
    """
    For each synthetic sample compute:
        - rank of paired real sample
        - distance to paired real
        - distance to the nearest real sample that is NOT the paired one
        - distance to the nearest other synthetic sample
        - whether nearest synthetic is closer than paired real
    """
    M = synth_flat.shape[0]

    # All pairwise distances (synthetic x real)
    print("Computing synthetic–real distance matrix...")
    dist_matrix = cdist(synth_flat, real_flat, metric='euclidean')   # (M, N_real)

    ranks = np.zeros(M, dtype=int)
    dist_to_paired = np.zeros(M)
    dist_to_nearest_other_real = np.zeros(M)

    for i in range(M):
        paired_idx = paired_real_indices[i]
        d_paired = dist_matrix[i, paired_idx]
        dist_to_paired[i] = d_paired
        # rank: count how many real samples are strictly closer
        n_closer = np.sum(dist_matrix[i] < d_paired)
        ranks[i] = n_closer + 1

        # nearest other real: set paired distance to inf and take min
        row = dist_matrix[i].copy()
        row[paired_idx] = np.inf
        dist_to_nearest_other_real[i] = np.min(row)

    # Intra‑synthetic nearest neighbours (excluding self)
    print("Computing intra‑synthetic nearest neighbours...")
    nn_synth = NearestNeighbors(n_neighbors=2, metric='euclidean')
    nn_synth.fit(synth_flat)
    distances_synth, _ = nn_synth.kneighbors(synth_flat)
    dist_to_nearest_synth = distances_synth[:, 1]

    synth_closer_than_paired = (dist_to_nearest_synth < dist_to_paired).astype(int)
    fraction_synth_closer = np.mean(synth_closer_than_paired) * 100

    return {
        'ranks': ranks,
        'dist_to_paired': dist_to_paired,
        'dist_to_nearest_other_real': dist_to_nearest_other_real,
        'dist_to_nearest_synth': dist_to_nearest_synth,
        'fraction_synth_closer_pct': fraction_synth_closer,
        'mean_rank': np.mean(ranks),
        'median_rank': np.median(ranks),
        'rank1_accuracy': np.mean(ranks == 1) * 100,
        'mean_dist_paired': np.mean(dist_to_paired),
        'median_dist_paired': np.median(dist_to_paired),
        'mean_dist_nearest_other_real': np.mean(dist_to_nearest_other_real),
        'median_dist_nearest_other_real': np.median(dist_to_nearest_other_real),
        'mean_dist_nearest_synth': np.mean(dist_to_nearest_synth),
        'median_dist_nearest_synth': np.median(dist_to_nearest_synth),
    }


def plot_summary(results, dataset_name):
    masks = sorted(results.keys())
    mask_pcts = [int(m*100) for m in masks]

    # Extract metrics
    rank1_acc = [results[m]['rank1_accuracy'] for m in masks]
    mean_rank = [results[m]['mean_rank'] for m in masks]
    median_rank = [results[m]['median_rank'] for m in masks]
    synth_closer = [results[m]['fraction_synth_closer_pct'] for m in masks]

    mean_dist_paired = [results[m]['mean_dist_paired'] for m in masks]
    median_dist_paired = [results[m]['median_dist_paired'] for m in masks]
    mean_dist_other = [results[m]['mean_dist_nearest_other_real'] for m in masks]
    median_dist_other = [results[m]['median_dist_nearest_other_real'] for m in masks]
    mean_dist_synth = [results[m]['mean_dist_nearest_synth'] for m in masks]
    median_dist_synth = [results[m]['median_dist_nearest_synth'] for m in masks]

    fig, (ax_dist, ax_table) = plt.subplots(1, 2, figsize=(15, 6))

    # ---------- Left: distance curves (mean solid, median dashed) ----------
    ax_dist.plot(mask_pcts, mean_dist_paired, 'o-', color='blue', linewidth=2, markersize=6, label='Mean dist to paired real')
    ax_dist.plot(mask_pcts, median_dist_paired, 'o--', color='blue', linewidth=1.5, markersize=4, alpha=0.7, label='Median dist to paired real')

    ax_dist.plot(mask_pcts, mean_dist_other, 's-', color='green', linewidth=2, markersize=6, label='Mean dist to nearest other real')
    ax_dist.plot(mask_pcts, median_dist_other, 's--', color='green', linewidth=1.5, markersize=4, alpha=0.7, label='Median dist to nearest other real')

    ax_dist.plot(mask_pcts, mean_dist_synth, '^-', color='red', linewidth=2, markersize=6, label='Mean dist to nearest other synthetic')
    ax_dist.plot(mask_pcts, median_dist_synth, '^--', color='red', linewidth=1.5, markersize=4, alpha=0.7, label='Median dist to nearest other synthetic')

    ax_dist.set_xlabel('Masking percentage (%)')
    ax_dist.set_ylabel('Euclidean distance')
    ax_dist.set_title('Distance metrics (mean = solid, median = dashed)')
    ax_dist.legend(loc='best', fontsize=9)
    ax_dist.grid(True, alpha=0.3)

    # ---------- Right: table with all statistics ----------
    ax_table.axis('tight')
    ax_table.axis('off')
    table_data = [
        ['Mask (%)'] + mask_pcts,
        ['Rank‑1 accuracy (%)'] + [f'{v:.1f}' for v in rank1_acc],
        ['Mean rank'] + [f'{v:.2f}' for v in mean_rank],
        ['Median rank'] + [f'{v:.2f}' for v in median_rank],
        ['Synthetic neighbour closer than paired (%)'] + [f'{v:.1f}' for v in synth_closer],
        ['Mean dist paired'] + [f'{v:.3f}' for v in mean_dist_paired],
        ['Median dist paired'] + [f'{v:.3f}' for v in median_dist_paired],
        ['Mean dist other real'] + [f'{v:.3f}' for v in mean_dist_other],
        ['Median dist other real'] + [f'{v:.3f}' for v in median_dist_other],
        ['Mean dist synth'] + [f'{v:.3f}' for v in mean_dist_synth],
        ['Median dist synth'] + [f'{v:.3f}' for v in median_dist_synth],
    ]
    table = ax_table.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    ax_table.set_title('Summary statistics', fontsize=10)

    plt.suptitle(f'kNN Identity Check – {dataset_name}', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(ROOT, f'Figures/{dataset_name}/knn_novelty_analysis.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved summary figure to {save_path}")
    plt.show()


def main(dataset_name: str, masking_percentages=None):
    if masking_percentages is None:
        masking_percentages = [0.00, 0.25, 0.5, 0.75, 1.0]

    print(f"Loading all real data (train+val+test) for {dataset_name}...")
    real_flat, split_ids = load_all_real_data(dataset_name)

    paired_real_indices = get_paired_real_test_indices(split_ids)

    results = {}

    for mask_pct in masking_percentages:
        print(f"\n💠 Processing synthetic mask {int(mask_pct*100)}%...")
        synth_flat = load_synthetic_test_data(dataset_name, mask_pct)

        assert synth_flat.shape[0] == len(paired_real_indices), \
            f"Count mismatch: synth={synth_flat.shape[0]} vs test real={len(paired_real_indices)}"

        metrics = compute_matching_metrics(real_flat, synth_flat, paired_real_indices)

        results[mask_pct] = {
            'rank1_accuracy': metrics['rank1_accuracy'],
            'mean_rank': metrics['mean_rank'],
            'median_rank': metrics['median_rank'],
            'mean_dist_paired': metrics['mean_dist_paired'],
            'median_dist_paired': metrics['median_dist_paired'],
            'mean_dist_nearest_other_real': metrics['mean_dist_nearest_other_real'],
            'median_dist_nearest_other_real': metrics['median_dist_nearest_other_real'],
            'mean_dist_nearest_synth': metrics['mean_dist_nearest_synth'],
            'median_dist_nearest_synth': metrics['median_dist_nearest_synth'],
            'fraction_synth_closer_pct': metrics['fraction_synth_closer_pct'],
        }

    # Create visualisation
    plot_summary(results, dataset_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='kNN identity check for synthetic MedMNIST data (with visualisation)')
    parser.add_argument('--dataset', required=True, help='Dataset name, e.g. bloodmnist')
    parser.add_argument('--masks', type=float, nargs='+', help='Masking percentages (e.g., 0.25 0.5 0.75 1.0)')
    args = parser.parse_args()
    main(args.dataset, args.masks)