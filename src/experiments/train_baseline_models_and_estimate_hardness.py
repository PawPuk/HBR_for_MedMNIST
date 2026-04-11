"""
This module trains an ensemble on the balanced, full-sized dataset and computes the hardness of each sample.
"""

import argparse

from src.data.loading import load_dataset
from src.training.train_ensembles import ModelTrainer


def main(dataset_name: str):
    training_loader, training_set, validation_loader, test_loader = load_dataset(dataset_name, True, True)
    training_set_size = len(training_set)
    training_loaders = [training_loader]

    trainer = ModelTrainer(training_set_size, training_loaders, validation_loader, test_loader, dataset_name,
                           estimate_hardness=True, for_baseline=True)

    trainer.train_ensemble()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ensemble of models on MedMNIST datasets.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['bloodmnist', 'pneumoniamnist'], help='Dataset name: bloodmnist or pneumoniamnist')

    args = parser.parse_args()
    main(args.dataset_name)
