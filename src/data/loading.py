"""The data module: Provides two core Dataset subclasses, and the method for loading the data."""

import random
from typing import Dict, List, Tuple, Union

import medmnist
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.config.config import get_config
from src.data.datasets import IndexedDataset


def get_transform(
        apply_augmentation: bool,
        config: Dict[str, Union[int, float, List[int], List[float], List[str], Tuple[float, float, float]]]
) -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    """For getting the transformation to the training and test sets."""
    if apply_augmentation:
        train_transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(config['mean'], config['std']),
        ])
    else:
        train_transform = transforms.ToTensor()

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config['mean'], config['std']),
    ])
    return train_transform, test_transform, test_transform


def worker_init_fn(worker_id):
    """Set the seed for workers"""
    np.random.seed(42 + worker_id)
    random.seed(42 + worker_id)


def get_dataloader(
        dataset: IndexedDataset,
        batch_size: int,
        shuffle: bool = False
):
    """Create a DataLoader with deterministic worker initialization."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, worker_init_fn=worker_init_fn)


def load_dataset(
        dataset_name: str,
        shuffle: bool = False,
        apply_augmentation: bool = False
) -> Tuple[DataLoader[IndexedDataset], IndexedDataset, DataLoader[IndexedDataset], DataLoader[IndexedDataset]]:
    """Load the dataset giving control over shuffling and augmentation.

    :param dataset_name: Name of the dataset to load (only accepts MedMNIST datasets).
    :param shuffle: Raise this flag to shuffle the training dataset.
    :param apply_augmentation: Raise this flag to apply data augmentation to the training set.

    :return: Tuple containing DataLoader for the training set, training set, as well as DataLoaders for validation and
    test sets.
    """
    config = get_config(dataset_name)
    size = config['size']
    DataClass = getattr(medmnist, medmnist.INFO[dataset_name]['python_class'])

    training_transform, validation_transform, test_transform = get_transform(apply_augmentation, config)

    as_rgb = False
    training_set = DataClass(split='train', transform=training_transform, download=True, as_rgb=as_rgb, size=size)
    validation_set = DataClass(split='val', transform=validation_transform, download=True, as_rgb=as_rgb, size=size)
    test_set = DataClass(split='test', transform=test_transform, download=True, as_rgb=as_rgb, size=size)

    training_set = IndexedDataset(training_set, apply_augmentation is False)
    validation_set = IndexedDataset(validation_set, True)
    test_set = IndexedDataset(test_set, True)

    training_loader = get_dataloader(training_set, config['batch_size'], shuffle)
    validation_loader = get_dataloader(validation_set, config['batch_size'])
    test_loader = get_dataloader(test_set, config['batch_size'])

    return training_loader, training_set, validation_loader, test_loader
