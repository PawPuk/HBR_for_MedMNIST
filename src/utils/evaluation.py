from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config.config import DEVICE
from src.models.neural_networks import ResNet18


def evaluate_model_dataset_level(
        model: ResNet18,
        criterion: nn.CrossEntropyLoss,
        test_loader: DataLoader
) -> Tuple[float, float, float, float]:
    """Evaluate the model on the test set."""
    model.eval()
    correct, total, running_loss, all_probs, all_labels = 0, 0, 0.0, [], []

    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    avg_loss = running_loss / total
    accuracy = 100 * correct / total

    num_classes = all_probs.shape[1]
    if num_classes == 2:
        auc_macro = roc_auc_score(all_labels, all_probs[:, 1])
        auc_weighted = auc_macro
    else:
        auc_macro = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        auc_weighted = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    return avg_loss, accuracy, auc_macro, auc_weighted


def evaluate_model_class_level(
        model: ResNet18,
        data_loader: DataLoader,
        num_classes: int
) -> Tuple[List[float], List[float]]:
    """
    Evaluate model performance at the class level.

    :param model: The model to evaluate
    :param data_loader: DataLoader for the dataset to evaluate on
    :param num_classes: Number of classes in the dataset
    :return: Dictionary containing per-class accuracy and per-class AUC scores
             Format: {'accuracy': {class_idx: accuracy, ...},
                     'auc': {class_idx: auc_score, ...}}
    """
    model.eval()
    all_labels, all_probs, per_class_accuracy = [], [], []

    with torch.no_grad():
        for inputs, labels, _ in data_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    # Calculate per-class accuracy
    predicted_classes = np.argmax(all_probs, axis=1)
    for class_idx in range(num_classes):
        class_mask = (all_labels == class_idx)
        per_class_accuracy.append(100 * np.mean(predicted_classes[class_mask] == class_idx))

    # Calculate per-class AUC
    if num_classes == 2:
        pos_prob = all_probs[:, 1]
        auc = roc_auc_score(all_labels, pos_prob)
        per_class_auc = [auc, auc]
    else:
        per_class_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average=None)

    return per_class_accuracy, per_class_auc
