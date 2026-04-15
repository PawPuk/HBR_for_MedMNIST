"""
Module for loading retrained models and measuring class-level performance metrics.
"""

import argparse
import os
import pickle

import torch

from src.config.config import DEVICE, ROOT, get_config
from src.data.loading import load_dataset
from src.models.loading import load_baseline_models
from src.models.neural_networks import ResNet18
from src.utils.evaluation import evaluate_model_class_level


def main(dataset_name: str):
    training_loader, _, _, _ = load_dataset(dataset_name)

    config = get_config(dataset_name)
    model_paths = load_baseline_models(dataset_name, config['num_epochs'])
    class_level_performances = [{} for _ in range(len(model_paths))]
    for model_idx, model_path in enumerate(model_paths):
        model_state = torch.load(model_path)
        model = ResNet18(in_channels=config['n_channels'], num_classes=config['num_classes']).to(DEVICE)
        model.load_state_dict(model_state)
        model.eval()
        per_class_accuracy, per_class_auc = evaluate_model_class_level(model, training_loader, config['num_classes'])
        class_level_performances[model_idx]["accuracy"] = per_class_accuracy
        class_level_performances[model_idx]["auc"] = per_class_auc

    path = os.path.join(ROOT, 'Results', f'class_level_performances_{dataset_name}.pkl')
    with open(path, "wb") as file:
        pickle.dump(class_level_performances, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ensemble of models on MedMNIST datasets.')
    parser.add_argument('--dataset_name', type=str, required=False, default='bloodmnist',
                        choices=['bloodmnist', 'pneumoniamnist'], help='Dataset name: bloodmnist or pneumoniamnist')

    args = parser.parse_args()
    main(args.dataset_name)
