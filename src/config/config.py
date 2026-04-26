"""
Configuration module.

This module defines global experiment parameters used across the repository.

Experimental design
-------------------

The experiments are designed to measure whether hardness-based resampling
reduces performance disparities across classes.

To ensure statistical robustness, we account for two sources of randomness:

1. Model initialization randomness
2. Dataset generation randomness (pruning or resampling)

To control for these factors, experiments are performed using a two-level design:

    number of dataset variants  ×  number of models per dataset

For example:

    4 datasets × 4 models per dataset = 16 trained models

Each dataset variant is generated using different random seeds during
pruning or resampling (see utils/reproducibility.py for seed generation).
For each dataset variant an ensemble of models is trained, allowing us
to compute stable performance estimates.

Important parameters
--------------------

num_datasets
    Number of independently generated dataset variants.

num_models_per_dataset
    Number of models trained for each dataset variant.

num_models_for_hardness
    Number of models used to estimate sample hardness during
    baseline ensemble training.

Runtime considerations
---------------------

Larger configurations increase statistical robustness but significantly
increase training time.

Typical settings:

    4×4  – used in the paper experiments
    2×2  – recommended for faster experimentation
    1×1  – quick but non-robust experimentation
"""

import os
from typing import Any, Dict

import medmnist
import torch


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = '.'

_base_config = {
    # Training hyperparameters
    'batch_size': 128,
    'lr': 0.001,
    'gamma': 0.1,
    'num_epochs': 100,
    'size': 28,  # the image size of the dataset, 28 or 64 or 128 or 224 are possible, but we use 28.

    # Experimental robustness parameters
    'num_datasets': 3,              # number of dataset variants
    'num_models_per_dataset': 3,    # number of models trained on each dataset variant

    # Hardness estimation
    'num_models_for_hardness': 10,   # ensemble size used to compute hardness in train_baseline_models.py
}

_dataset_specific_config = {
    # 2D MedMNIST datasets
    'pathmnist': {
        'num_classes': 9,
        'mean': (0.5,),
        'std': (0.5,),
        'n_channels': medmnist.INFO['pathmnist']['n_channels']
    },
    'chestmnist': {
        'num_classes': 2,
        'mean': (0.5,),
        'std': (0.5,),
        'n_channels': medmnist.INFO['chestmnist']['n_channels']
    },
    'dermamnist': {
        'num_classes': 7,
        'mean': (0.5,),
        'std': (0.5,),
        'n_channels': medmnist.INFO['dermamnist']['n_channels']
    },
    'octmnist': {
        'num_classes': 4,
        'mean': (0.5,),
        'std': (0.5,),
        'n_channels': medmnist.INFO['octmnist']['n_channels']
    },
    'pneumoniamnist': {
        'num_classes': 2,
        'mean': (0.5,),
        'std': (0.5,),
        'n_channels': medmnist.INFO['pneumoniamnist']['n_channels']
    },
    'retinamnist': {
        'num_classes': 5,
        'mean': (0.5,),
        'std': (0.5,),
        'n_channels': medmnist.INFO['retinamnist']['n_channels']
    },
    'breastmnist': {
        'num_classes': 2,
        'mean': (0.5,),
        'std': (0.5,),
        'n_channels': medmnist.INFO['breastmnist']['n_channels']
    },
    'bloodmnist': {
        'num_classes': 8,
        'mean': (0.5,),
        'std': (0.5,),
        'n_channels': medmnist.INFO['bloodmnist']['n_channels']
    },
    'tissuemnist': {
        'num_classes': 8,
        'mean': (0.5,),
        'std': (0.5,),
        'n_channels': medmnist.INFO['tissuemnist']['n_channels']
    },
    'organamnist': {
        'num_classes': 11,
        'mean': (0.5,),
        'std': (0.5,),
        'n_channels': medmnist.INFO['organamnist']['n_channels']
    },
    'organcmnist': {
        'num_classes': 11,
        'mean': (0.5,),
        'std': (0.5,),
        'n_channels': medmnist.INFO['organcmnist']['n_channels']
    },
    'organsmnist': {
        'num_classes': 11,
        'mean': (0.5,),
        'std': (0.5,),
        'n_channels': medmnist.INFO['organsmnist']['n_channels']
    }
}


def build_complete_configurations() -> Dict[str, Any]:
    dataset_configs = {}

    num_epochs = _base_config['num_epochs']
    milestones = [int(0.5 * num_epochs), int(0.75 * num_epochs)]

    for dataset_name, specific_config in _dataset_specific_config.items():
        config = _base_config.copy()
        config.update(specific_config)
        config['milestones'] = milestones
        dataset_configs[dataset_name] = config

    return dataset_configs


DATASET_CONFIGS = build_complete_configurations()


def get_config(dataset_name: str) -> Dict[str, Any]:
    if dataset_name in DATASET_CONFIGS:
        config = DATASET_CONFIGS[dataset_name].copy()
        config['probe_base_seed'] = 42
        config['probe_seed_step'] = 42
        config['probe_dataset_step'] = 420_000
        return config
    else:
        raise ValueError(f"Configuration for dataset {dataset_name} not found!")
