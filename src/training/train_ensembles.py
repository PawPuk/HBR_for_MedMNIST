"""Core module that allows for training ensembles of models as well as estimating hardness."""

import os
from typing import cast, Dict, Sized, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.config.config import DEVICE, get_config
from src.hardness.estimators import estimate_instance_hardness_via_learning_dynamics
from src.models.neural_networks import ResNet18
from src.utils.io import save_results
from src.utils.reproducibility import compute_current_seed, set_reproducibility
from src.utils.structures import get_latest_model_index


class ModelTrainer:
    """Allows training ensembles of models as well as estimating hardness."""
    def __init__(self, dataset_size: int, dataloader: DataLoader, dataset_name: str, split: str, save_models: bool,
                 run_suffix: str = ""):
        """
        Initialize the ModelTrainer class with configuration specific to the dataset.

        :param dataset_size: Specified the size of the dataset. This is required for initializing hardness estimates.
        :param dataloader: DataLoader wrapping the dataset on which hardness will be estimated.
        :param dataset_name: The name of the dataset. Used for saving
        :param split: Name of the split on which hardness estimation will be performed
        :param save_models: Indicates if the models trained during hardness estimation are to be saved for later use.
        :param run_suffix: Optional suffix added to save path (e.g., masking percentage) to avoid overwrites.
        """
        self.dataset_size = dataset_size
        self.split = split
        self.loader = dataloader
        self.dataset_name = dataset_name
        self.save_models = save_models
        self.run_suffix = run_suffix

        self.config = get_config(self.dataset_name)

        self.num_epochs = self.config['num_epochs']
        self.num_models_to_train = self.config['num_models']

        self.save_dir = os.path.join(self.config['save_dir'], dataset_name)
        os.makedirs(self.save_dir, exist_ok=True)

    def train_model(self, current_model_index: int, hardness_estimates: Dict[Tuple[int, int], Dict]):
        """Train a single model."""
        seed = compute_current_seed(self.config, current_model_index)
        set_reproducibility(seed)

        model = ResNet18(in_channels=self.config['n_channels'], num_classes=self.config['num_classes']).to(DEVICE)
        if self.dataset_name == "chestmnist":
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config['lr'])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.config['milestones'],
                                                   gamma=self.config['gamma'])

        for estimator in ['AUM', 'DataIQ']:
            # hardness_estimates[dataset_model_id][estimator][epoch_index][sample_index]: float
            hardness_estimates[current_model_index][estimator] = [[0.0 for _ in range(self.num_epochs)]
                                                                  for _ in range(self.dataset_size)]
        # hardness_estimates[dataset_model_id]['Forgetting'][sample_index]: int
        hardness_estimates[current_model_index]['Forgetting'] = [0 for _ in range(self.dataset_size)]
        remembering = [False for _ in range(self.dataset_size)]  # Required to computing Forgetting

        for epoch in tqdm(range(self.config['num_epochs']), desc='Iterating through epochs.'):
            model.train()

            for inputs, labels, indices in self.loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                estimate_instance_hardness_via_learning_dynamics(
                    indices, inputs, outputs, labels, predicted, hardness_estimates, epoch, remembering,
                    current_model_index
                )
            scheduler.step()

        if self.save_models:
            final_save_path = os.path.join(self.save_dir, f'model_{current_model_index}'
                                                          f'_epoch_{self.config["num_epochs"]}.pth')
            torch.save(model.state_dict(), final_save_path)

    def train_ensemble(self):
        """Train an ensemble of models."""

        latest_model_index = get_latest_model_index(self.dataset_name, self.split, self.run_suffix)

        print(f"Starting training ensemble of {self.num_models_to_train} models on the {self.split} split of "
              f"{self.dataset_name}.")
        print(f"Number of samples in the used DataLoader: {len(cast(Sized, self.loader.dataset))}")
        print('-'*20)

        for model_id in tqdm(range(latest_model_index + 1, self.num_models_to_train)):
            hardness_estimates = {model_id: {}}
            self.train_model(model_id, hardness_estimates)
            for estimator in ['AUM', 'DataIQ']:
                # Average hardness estimates (the ones that used learning dynamics) over all epochs.
                hardness_estimates[model_id][estimator] = np.mean(
                    hardness_estimates[model_id][estimator], axis=1)
            save_results(hardness_estimates, model_id, self.dataset_name, self.split, self.run_suffix)
