"""
This module trains an ensemble on the balanced, full-sized dataset and computes the hardness of each sample.
"""

import argparse

from src.data.loading import load_dataset
from src.training.train_ensembles import ModelTrainer


def main(dataset_name: str, split: str):
    train_loader, train_set, val_loader, val_set, test_loader, test_set = load_dataset(dataset_name, True, True)
    if split == 'training':
        training_set_size = len(train_set)
    elif split == 'validation':
        training_set_size = len(val_set)
    else:
        training_set_size = len(test_set)

    trainer = ModelTrainer(training_set_size, train_loader, val_loader, test_loader, dataset_name, split,
                           for_baseline=True)

    trainer.train_ensemble()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ensemble of models on MedMNIST datasets.')
    parser.add_argument('--dataset_name', type=str, required=False, default='bloodmnist',
                        choices=['bloodmnist', 'pneumoniamnist', 'dermamnist', 'pathmnist', 'chestmnist',
                                 'octmnist', 'tissuemnist', 'organamnist', 'organcmnist', 'organsmnist',
                                 'breastmnist', 'retinamnist'],
                        help='MedMNIST dataset name.')
    parser.add_argument('--split', type=str, default='training', choices=['training', 'validation', 'test'],
                        help='Split on which hardness will be estimated.')

    args = parser.parse_args()
    main(args.dataset_name, args.split)
