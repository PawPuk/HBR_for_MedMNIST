"""
This module estimates hardness of each sample on the specified dataset. Works both with real and synthetic data.
"""

import argparse

from src.data.loading import load_dataset
from src.training.train_ensembles import ModelTrainer


def main(dataset_name: str, split: str, data_type: bool, masking_percentage: float, save_models: bool):
    dataloader, dataset = load_dataset(dataset_name, split, data_type=data_type, masking_percentage=masking_percentage)

    dataset_size = len(dataset)
    suffix = f"{data_type}{masking_percentage}" if data_type in ['syn', 'both'] else "real"

    trainer = ModelTrainer(dataset_size, dataloader, dataset_name, split, save_models, run_suffix=suffix)
    trainer.train_ensemble()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate hardness on MedMNIST or synthetic data.')
    parser.add_argument('--dataset_name', type=str, default='bloodmnist',
                        choices=['bloodmnist', 'pneumoniamnist', 'dermamnist', 'pathmnist', 'chestmnist',
                                 'octmnist', 'tissuemnist', 'organamnist', 'organcmnist', 'organsmnist',
                                 'breastmnist', 'retinamnist'],
                        help='MedMNIST dataset name.')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Split on which training will be performed and hardness estimated.')
    parser.add_argument('--data_type', choices=['real', 'syn', 'both'],
                        help='The type of data that will be used for training and hardness estimation (type 1).')
    parser.add_argument('--masking_percentage', type=float, choices=[0.25, 0.50, 0.75, 1.00], default=0.00,
                        help='Masking percentage for synthetic data (required if --data_type syn/both).')
    parser.add_argument('--save_models', action='store_true', default=False,
                        help='Save the models used to estimate hardness.')

    args = parser.parse_args()
    if args.synthetic and args.masking_percentage is None:
        parser.error("--masking_percentage is required when --synthetic is set.")

    main(args.dataset_name, args.split, args.data_type, args.masking_percentage, args.save_models)
