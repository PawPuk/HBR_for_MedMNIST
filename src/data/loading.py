"""The data module: Provides Dataset subclasses and methods for loading data from MedMNIST or synthetic JPG folders."""

import os
import random
from typing import Dict, Tuple, Union, Optional

import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.config.config import get_config
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


def get_dataloader(dataset: IndexedDataset, batch_size: int) -> DataLoader:
    """Create a DataLoader with deterministic worker initialization."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, worker_init_fn=worker_init_fn)


def load_dataset(dataset_name: str, split: str, synthetic: bool = False, masking_percentage: Optional[float] = None):
    """Load dataset from MedMNIST or synthetic JPG folders.

    :param dataset_name: Name of the dataset (e.g., 'pathmnist').
    :param split: Name of the split ('train', 'val' or 'test').
    :param synthetic: Load from synthetic JPG images.
    :param masking_percentage: Required if synthetic=True, one of {0.25, 0.50, 0.75, 1.00}.
    """
    config = get_config(dataset_name)
    transform = get_transform(config)
    as_rgb = False
    root = os.path.join('Data', f'{["real", "synthetic"][synthetic]}_{dataset_name}')

    dataset = LocalDataset(root, masking_percentage, split, transform=transform, as_rgb=as_rgb)
    indexed_dataset = IndexedDataset(dataset)
    dataloader = get_dataloader(indexed_dataset, config['batch_size'])

    return dataloader, indexed_dataset
