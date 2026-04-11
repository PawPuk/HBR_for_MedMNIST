from typing import Tuple

from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config.config import DEVICE
from src.models.neural_networks import ResNet18


def evaluate_model(
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

    avg_loss = running_loss / total
    accuracy = 100 * correct / total
    auc_macro = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    auc_weighted = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    return avg_loss, accuracy, auc_macro, auc_weighted
