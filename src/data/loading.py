"""The data module: Provides Dataset subclasses and methods for loading data from MedMNIST or synthetic JPG folders."""

import os
import random
from typing import Dict, Tuple, Union, Optional

import numpy as np
from torch.utils.data import ConcatDataset, DataLoader
import torchvision.transforms as transforms

from src.config.config import get_config, ROOT
from src.data.datasets import IndexedDataset, LocalDataset


def get_transform(config: Dict[str, Union[int, float, Tuple[float, float, float]]]) -> transforms.Compose:
    """For getting the transformation to the training and test sets."""
    size = config['size']
    train_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(config['mean'], config['std']),
    ])
    return train_transform


def worker_init_fn(worker_id):
    """Set the seed for workers"""
    np.random.seed(42 + worker_id)
    random.seed(42 + worker_id)


def get_dataloader(dataset: IndexedDataset, batch_size: int, shuffle: bool) -> DataLoader:
    """Create a DataLoader with deterministic worker initialization."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, worker_init_fn=worker_init_fn)


def load_dataset(dataset_name: str, split: str, data_type: str, masking_percentage: Optional[float] = 2.00,
                 shuffle: bool = True):
    """Load dataset from MedMNIST or synthetic JPG folders.

    :param dataset_name: Name of the dataset (e.g., 'pathmnist').
    :param split: Name of the split ('train', 'val' or 'test').
    :param data_type: The type of data to load
    :param masking_percentage: Required if synthetic=True, one of {0.25, 0.50, 0.75, 1.00}.
    :param shuffle: If true then DataLoader will shuffle the data.
    """
    config = get_config(dataset_name)
    transform = get_transform(config)
    as_rgb = False

    root_map = {
        'real': [f'real_{dataset_name}'],
        'syn': [f'synthetic_{dataset_name}'],
        'both': [f'real_{dataset_name}', f'synthetic_{dataset_name}']
    }

    datasets = [
        LocalDataset(
            os.path.join(ROOT, 'Data', root),
            masking_percentage,
            split,
            transform=transform,
            as_rgb=as_rgb
        )
        for root in root_map[data_type]
    ]

    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    indexed_dataset = IndexedDataset(dataset)
    dataloader = get_dataloader(indexed_dataset, config['batch_size'], shuffle)

    return dataloader, indexed_dataset
