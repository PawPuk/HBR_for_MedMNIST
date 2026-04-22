import os.path

import numpy as np

from src.config.config import ROOT
from src.utils.io import load_results


def report_performance_metrics(dataset_name: str):
    """Load performance measures (per-class AUCs and accuracies) and report them."""
    class_level_performances = load_results(os.path.join(ROOT, f'Results/{dataset_name}/class_level_performances.pkl'))

    all_accuracies = np.array([class_level_performances[i]['accuracy'] for i in range(len(class_level_performances))])
    all_aucs = np.array([class_level_performances[i]['auc'] for i in range(len(class_level_performances))])
    avg_accuracy_per_class = np.mean(all_accuracies, axis=0)
    avg_auc_per_class = np.mean(all_aucs, axis=0)

    print([f'{x:.15f}' for x in avg_auc_per_class])
    print([f'{x:.15f}' for x in avg_accuracy_per_class])
