import argparse
import glob
import pickle
from src.config.config import ROOT
from src.visualisations.figures import *


def main(dataset_name: str):
    # Find all hardness data files (assuming test split only)
    data_dir = os.path.join(ROOT, f'Results/{dataset_name}')
    pattern = os.path.join(data_dir, 'test_hardness_data_*.pkl')
    file_paths = glob.glob(pattern)
    if not file_paths:
        raise FileNotFoundError(f"No files matching {pattern} found.")

    # Process each file independently
    for file_path in sorted(file_paths):
        # Extract variant ID from filename, e.g. "test_hardness_data_syn0.5.pkl" -> "syn0.5"
        basename = os.path.basename(file_path)
        variant_id = basename.replace('test_hardness_data_', '').replace('.pkl', '')  # "real" or "syn0.5"

        # Determine output subdirectory
        figure_subdir = 'real' if variant_id == 'real' else os.path.join('synthetic', variant_id)
        figure_save_path = os.path.join(ROOT, f'Figures/{dataset_name}/{figure_subdir}/')
        os.makedirs(figure_save_path, exist_ok=True)

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        class_cardinalities = data['class_cardinalities']
        thresholds = data['thresholds']
        model_counts = data['model_counts']
        estimators = data['estimators']
        all_hardness_by_class = data['all_hardness_by_class']
        all_hard_samples = data['all_hard_samples']
        all_hard_samples_by_class = data['all_hard_samples_by_class']

        # Use the largest ensemble for all plots (apart from those for analysis of stability of hardness estimates)
        n_models_max = max(model_counts)
        hardness_by_class_max = {est: all_hardness_by_class[est][n_models_max] for est in estimators}
        hard_samples_max = {est: all_hard_samples[est][n_models_max] for est in estimators}
        hard_samples_by_class_max = {est: all_hard_samples_by_class[est][n_models_max] for est in estimators}

        # --- Plots using largest ensemble statistics ---
        for est in estimators:
            plot_class_hard_samples_simple(class_cardinalities, hard_samples_by_class_max[est],
                                           est, dataset_name, figure_save_path, thresholds)
        plot_hard_sample_pairwise_overlap(hard_samples_max, thresholds, dataset_name, figure_save_path)
        plot_class_cardinalities_and_hardness(class_cardinalities, hardness_by_class_max,
                                              dataset_name, figure_save_path)

        # --- Stability plots ---
        for est in estimators:
            plot_relative_stability_of_estimates(all_hard_samples, thresholds, model_counts,
                                                 est, dataset_name, figure_save_path)
            plot_consecutive_stability(all_hard_samples, thresholds, model_counts,
                                       est, dataset_name, figure_save_path)

        plot_hardness_with_std(dataset_name)
        plot_hardness_with_std(dataset_name, shared_sorting=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['bloodmnist', 'pneumoniamnist', 'dermamnist', 'pathmnist',
                                 'octmnist', 'tissuemnist', 'organamnist', 'organcmnist',
                                 'chestmnist', 'organsmnist', 'breastmnist', 'retinamnist'])
    args = parser.parse_args()
    main(args.dataset_name)
