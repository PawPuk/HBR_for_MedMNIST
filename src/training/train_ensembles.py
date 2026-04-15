"""Core module that allows for training ensembles of models as well as estimating hardness."""

import os
from typing import cast, Dict, List, Sized, Tuple, Union

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.config.config import DEVICE, get_config
from src.hardness.estimators import estimate_instance_hardness
from src.models.neural_networks import ResNet18
from src.utils.evaluation import evaluate_model_dataset_level
from src.utils.io import save_results
from src.utils.reproducibility import compute_current_seed, set_reproducibility
from src.utils.structures import get_latest_model_index


class ModelTrainer:
    """Allows training ensembles of models as well as estimating hardness."""
    def __init__(
            self,
            training_set_size: int,
            training_loaders: List[DataLoader],
            validation_loader: Union[DataLoader, None],
            test_loader: Union[DataLoader, None],
            dataset_name: str,
            estimate_hardness: bool = False,
            for_baseline: bool = False
    ):
        """
        Initialize the ModelTrainer class with configuration specific to the dataset.

        :param training_set_size: Specified the size of the training set. This is only useful for measuring hardness.
        :param training_loaders: List of DataLoaders for the training datasets. For baseline training, where only one
        dataset is used, pass the single DataLoader in a List.
        :param validation_loader: Dataloader for the validation set.
        :param test_loader: DataLoader for the test set.
        :param dataset_name: The name of the dataset. Used for saving
        :param estimate_hardness: Specify if the hardness should be saved and stored during training (default False). We
        set this to True only for experiment1.py as we do not currently estimate hardness on pruned or resampled
        datasets.
        :param for_baseline: A flag used to indicate whether the training is performed for baseline models (in which
        case we train only one ensemble with more models) or for resampling (where we train multiple smaller ensembles)
        """
        self.training_set_size = training_set_size
        self.training_loaders = training_loaders
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.dataset_name = dataset_name
        self.estimate_hardness = estimate_hardness

        self.config = get_config(self.dataset_name)

        self.num_epochs = self.config['num_epochs']
        # For baseline training we train single ensemble as there is only one dataset (unlike with resampling
        # experiments where we train on multiple versions of a dataset to account for variability in resampling)
        if for_baseline:
            self.num_models_to_train_per_dataset = self.config['num_datasets'] * self.config['num_models_per_dataset']
            self.dataset_count = 1
        else:
            self.num_models_to_train_per_dataset = self.config['num_models_per_dataset']
            self.dataset_count = self.config['num_datasets']

        self.save_dir = os.path.join(self.config['save_dir'], dataset_name)
        os.makedirs(self.save_dir, exist_ok=True)

    @staticmethod
    def print_performance(split: str, avg_loss: float, accuracy: float, auc_macro: float, auc_weighted: float):
        print(f'{split} Loss: {avg_loss:.4f}, {split} Acc: {accuracy:.2f}%, '
              f'{split} AUC (macro, weighted): ({auc_macro:.2f}, {auc_weighted:.2f})')

    def train_model(
            self,
            current_dataset_index: int,
            current_model_index: int,
            hardness_estimates: Union[Dict[Tuple[int, int], Dict], None]
    ):
        """Train a single model."""
        dataset_model_id = (current_dataset_index, current_model_index)
        seed = compute_current_seed(self.config, current_dataset_index, current_model_index)
        set_reproducibility(seed)

        model = ResNet18(in_channels=self.config['n_channels'], num_classes=self.config['num_classes']).to(DEVICE)
        if self.dataset_name == "chestmnist":
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config['lr'])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.config['milestones'],
                                                   gamma=self.config['gamma'])

        if self.estimate_hardness:
            for estimator in ['AUM', 'DataIQ']:
                # hardness_estimates[dataset_model_id][estimator][epoch_index][sample_index]: float
                hardness_estimates[dataset_model_id][estimator] = [[0.0 for _ in range(self.num_epochs)]
                                                                   for _ in range(self.training_set_size)]
            # hardness_estimates[dataset_model_id]['Forgetting'][sample_index]: int
            hardness_estimates[dataset_model_id]['Forgetting'] = [0 for _ in range(self.training_set_size)]
        remembering = [False for _ in range(self.training_set_size)]  # Required to computing Forgetting

        best_val_auc, best_model_state = 0.0, None

        for epoch in range(self.config['num_epochs']):
            model.train()
            running_loss, correct_train, total_train, all_probs, all_labels = 0.0, 0, 0, [], []

            for inputs, labels, indices in self.training_loaders[current_dataset_index]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += predicted.eq(labels).sum().item()

                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.detach().cpu())
                all_labels.append(labels.cpu())

                if self.estimate_hardness:
                    estimate_instance_hardness(indices, inputs, outputs, labels, predicted, hardness_estimates, epoch,
                                               remembering, dataset_model_id)
            scheduler.step()

            all_labels = torch.cat(all_labels).numpy()
            all_probs = torch.cat(all_probs).numpy()

            # Report progress (accuracy & loss on training & test sets)
            if self.test_loader is not None:
                avg_val_loss, val_accuracy, val_auc_macro, val_auc_weighted = evaluate_model_dataset_level(
                    model, criterion, self.validation_loader
                )
                if val_auc_weighted > best_val_auc:
                    best_val_auc = val_auc_weighted
                    best_model_state = model.state_dict().copy()

                avg_training_loss = running_loss / total_train
                training_accuracy = 100 * correct_train / total_train

                num_classes = all_probs.shape[1]
                if num_classes == 2:
                    training_auc_macro = roc_auc_score(all_labels, all_probs[:, 1])
                    training_auc_weighted = training_auc_macro
                else:
                    training_auc_macro = roc_auc_score(all_labels, all_probs, multi_class='ovr')
                    training_auc_weighted = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
                print(f'Model {current_model_index}, Epoch [{epoch + 1}/{self.config["num_epochs"]}] ')
                self.print_performance("Training", avg_training_loss, training_accuracy, training_auc_macro,
                                       training_auc_weighted)
                self.print_performance("Validation", avg_val_loss, val_accuracy, val_auc_macro, val_auc_weighted)

        best_model = ResNet18(in_channels=self.config['n_channels'], num_classes=self.config['num_classes']).to(DEVICE)
        best_model.load_state_dict(best_model_state)
        avg_val_loss, val_accuracy, val_auc_macro, val_auc_weighted = evaluate_model_dataset_level(
            best_model, criterion, self.validation_loader
        )
        avg_test_loss, test_accuracy, test_auc_macro, test_auc_weighted = evaluate_model_dataset_level(
            best_model, criterion, self.test_loader
        )

        print('-'*20)
        print(f'Best model performance:')
        self.print_performance("Validation", avg_val_loss, val_accuracy, val_auc_macro, val_auc_weighted)
        self.print_performance("Test", avg_test_loss, test_accuracy, test_auc_macro, test_auc_weighted)
        print('-'*20)

        # Save model after full training
        final_save_path = os.path.join(self.save_dir, f'dataset_{current_dataset_index}'
                                                      f'_model_{current_model_index}'
                                                      f'_epoch_{self.config["num_epochs"]}.pth')
        torch.save(best_model_state, final_save_path)

    def train_ensemble(
            self
    ):
        """Train an ensemble of models."""

        latest_model_indices = get_latest_model_index(self.save_dir, self.config['num_epochs'], self.dataset_count)

        print(f"Starting training {self.dataset_count} ensembles of {self.num_models_to_train_per_dataset} models each "
              f"on {self.dataset_name}.")
        print(f"Number of samples in the training loader: {len(cast(Sized, self.training_loaders[0].dataset))}")
        print(f"Number of samples in the test loader: {len(cast(Sized, self.test_loader.dataset))}")
        print('-'*20)

        for dataset_id in tqdm(range(self.dataset_count)):
            for model_id in tqdm(range(latest_model_indices[dataset_id] + 1, self.num_models_to_train_per_dataset)):
                hardness_estimates = {(dataset_id, model_id): {}}
                self.train_model(dataset_id, model_id, hardness_estimates)
                if self.estimate_hardness:
                    # Even though we computed multiple hardness estimates we only used AUM for our core experiments.
                    for estimator in ['AUM', 'DataIQ']:
                        # Average hardness estimates (the ones that used learning dynamics) over all epochs.
                        hardness_estimates[(dataset_id, model_id)][estimator] = np.mean(
                            hardness_estimates[(dataset_id, model_id)][estimator], axis=1)
                    save_results(hardness_estimates, (dataset_id, model_id), self.dataset_name)
