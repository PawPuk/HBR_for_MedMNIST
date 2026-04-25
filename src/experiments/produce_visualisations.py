import argparse
import pickle

from src.config.config import ROOT
from src.visualisations.terminal import report_performance_metrics
from src.visualisations.figures import *


def main(dataset_name: str):
    # Report performance metrics once (not split‑dependent)
    report_performance_metrics(dataset_name)

    # Splits and corresponding file suffixes
    splits = [
        ('training', 'training_hardness_data.pkl'),
        ('validation', 'validation_hardness_data.pkl'),
        ('test', 'test_hardness_data.pkl')
    ]

    for split_name, data_filename in splits:
        data_path = os.path.join(ROOT, f'Results/{dataset_name}/{data_filename}')
        if not os.path.exists(data_path):
            print(f"Warning: {data_path} not found, skipping {split_name} split.")
            continue

        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        class_cardinalities = data['class_cardinalities']
        thresholds = data['thresholds']
        model_counts = data['model_counts']
        estimators = data['estimators']
        all_hardness_by_class = data['all_hardness_by_class']
        all_hard_samples = data['all_hard_samples']
        all_hard_samples_by_class = data['all_hard_samples_by_class']
        all_final_hardness = data['all_final_hardness']

        # Create split‑specific figure directory
        figure_save_path = os.path.join(ROOT, f'Figures/{dataset_name}/{split_name}/')
        os.makedirs(figure_save_path, exist_ok=True)

        # Use the largest ensemble for all plots (apart from those for analysis of stability of hardness estimates)
        n_models_max = max(model_counts)
        hardness_by_class_max = {est: all_hardness_by_class[est][n_models_max] for est in estimators}
        hard_samples_max = {est: all_hard_samples[est][n_models_max] for est in estimators}
        hard_samples_by_class_max = {est: all_hard_samples_by_class[est][n_models_max] for est in estimators}
        final_hardness = {est: all_final_hardness[est][n_models_max] for est in estimators}

        # --- Plots using largest ensemble statistics ---
        for est in estimators:
            plot_dataset_level_hardness_distribution(final_hardness[est], est, figure_save_path)
            plot_class_hard_samples_simple(class_cardinalities, hard_samples_by_class_max[est], est, dataset_name,
                                           figure_save_path, thresholds)
        plot_hard_sample_pairwise_overlap(hard_samples_max, thresholds, dataset_name, figure_save_path)
        plot_class_cardinalities_and_hardness(class_cardinalities, hardness_by_class_max, dataset_name, figure_save_path)

        # --- Stability plots ---
        for est in estimators:
            plot_relative_stability_of_estimates(all_hard_samples, thresholds, model_counts, est, dataset_name,
                                                 figure_save_path)
            plot_consecutive_stability(all_hard_samples, thresholds, model_counts, est, dataset_name, figure_save_path)

        print(f"Finished visualisations for {split_name} split (saved to {figure_save_path})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=False, default='bloodmnist',
                        choices=['bloodmnist', 'pneumoniamnist', 'dermamnist', 'pathmnist', 'chestmnist',
                                 'octmnist', 'tissuemnist', 'organamnist', 'organcmnist', 'organsmnist',
                                 'breastmnist', 'retinamnist'])
    args = parser.parse_args()
    main(args.dataset_name)
