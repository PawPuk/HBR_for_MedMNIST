"""
This module estimates hardness of each sample on the specified dataset. Works both with real and synthetic data.
"""

import argparse

from src.data.loading import load_dataset
from src.training.train_ensembles import ModelTrainer


def main(dataset_name: str, split: str, synthetic: bool, masking_percentage: float = None):
    dataloader, dataset = load_dataset(dataset_name, split, synthetic=synthetic, masking_percentage=masking_percentage)

    dataset_size = len(dataset)
    suffix = f"syn{masking_percentage}" if synthetic else "real"

    trainer = ModelTrainer(dataset_size, dataloader, dataset_name, split, run_suffix=suffix)
    trainer.train_ensemble()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate hardness on MedMNIST or synthetic data.')
    parser.add_argument('--dataset_name', type=str, default='bloodmnist',
                        choices=['bloodmnist', 'pneumoniamnist', 'dermamnist', 'pathmnist', 'chestmnist',
                                 'octmnist', 'tissuemnist', 'organamnist', 'organcmnist', 'organsmnist',
                                 'breastmnist', 'retinamnist'],
                        help='MedMNIST dataset name.')
    parser.add_argument('--split', type=str, default='test', choices=['training', 'validation', 'test'],
                        help='Split on which hardness will be estimated.')
    parser.add_argument('--synthetic', action='store_true', default=False,
                        help='Use synthetic JPG dataset instead of original MedMNIST.')
    parser.add_argument('--masking_percentage', type=float, choices=[0.25, 0.50, 0.75, 1.00],
                        help='Masking percentage for synthetic data (required if --synthetic).')

    args = parser.parse_args()
    if args.synthetic and args.masking_percentage is None:
        parser.error("--masking_percentage is required when --synthetic is set.")

    main(args.dataset_name, args.split, args.synthetic, args.masking_percentage)
