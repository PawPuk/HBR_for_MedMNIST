"""The data module: Provides Dataset subclasses and methods for loading data from MedMNIST or synthetic JPG folders."""

import os
import random
from typing import Dict, List, Tuple, Union, Optional

import medmnist
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.config.config import get_config
from src.data.datasets import IndexedDataset, SyntheticDataset


def get_transform(
        config: Dict[str, Union[int, float, List[int], List[float], List[str], Tuple[float, float, float]]]
) -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    """For getting the transformation to the training and test sets."""
    train_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(config['mean'], config['std']),
    ])
    return train_transform


def worker_init_fn(worker_id):
    """Set the seed for workers"""
    np.random.seed(42 + worker_id)
    random.seed(42 + worker_id)


def get_dataloader(
        dataset: IndexedDataset,
        batch_size: int
):
    """Create a DataLoader with deterministic worker initialization."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, worker_init_fn=worker_init_fn)


def load_dataset(
        dataset_name: str,
        synthetic: bool = False,
        masking_percentage: Optional[float] = None,
):
    """Load dataset from MedMNIST, local NPZ, or synthetic JPG folders.

    :param dataset_name: Name of the dataset (e.g., 'pathmnist').
    :param synthetic: Load from synthetic JPG images + CSV.
    :param masking_percentage: Required if synthetic=True, one of {0.25, 0.50, 0.75, 1.00}.
    """
    config = get_config(dataset_name)
    size = config['size']
    training_transform = get_transform(config)
    as_rgb = False
    synthetic_root = os.path.join('Data', f'synthetic_{dataset_name}')

    if synthetic:
        # Create datasets for each split using the synthetic folder structure
        train_set = SyntheticDataset(synthetic_root, dataset_name, masking_percentage,
                                     transform=training_transform, size=size)
        val_set = SyntheticDataset(synthetic_root, dataset_name, masking_percentage,
                                   transform=training_transform, size=size)
        test_set = SyntheticDataset(synthetic_root, dataset_name, masking_percentage,
                                    transform=training_transform, size=size)
    else:
        # Original MedMNIST library loading
        DataClass = getattr(medmnist, medmnist.INFO[dataset_name]['python_class'])
        train_set = DataClass(split='train', transform=training_transform, download=True,
                              as_rgb=as_rgb, size=size)
        val_set = DataClass(split='val', transform=training_transform, download=True,
                            as_rgb=as_rgb, size=size)
        test_set = DataClass(split='test', transform=training_transform, download=True,
                             as_rgb=as_rgb, size=size)

    train_set = IndexedDataset(train_set)
    val_set = IndexedDataset(val_set)
    test_set = IndexedDataset(test_set)

    train_loader = get_dataloader(train_set, config['batch_size'])
    val_loader = get_dataloader(val_set, config['batch_size'])
    test_loader = get_dataloader(test_set, config['batch_size'])

    return train_loader, train_set, val_loader, val_set, test_loader, test_set
