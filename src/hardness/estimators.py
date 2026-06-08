from typing import Dict, List, Tuple, Union

import numpy as np
import torch


def update_AUM(
        batch_indices: torch.Tensor,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        hardness_estimates: Dict[Tuple[int, int], Dict[str, List[Union[int, List[float]]]]],
        epoch: int,
        current_model_index: Tuple[int, int]
):
    """Estimate hardness through AUM (https://arxiv.org/pdf/2001.10528)"""
    for index_within_batch, (i, logits, correct_label) in enumerate(zip(batch_indices, outputs, labels)):
        i = i.item()
        correct_label = correct_label.item()

        logits = logits.detach()
        correct_logit = logits[correct_label].item()

        max_other_logit = torch.max(torch.cat((logits[:correct_label], logits[correct_label + 1:]))).item()

        hardness_estimates[current_model_index][i][epoch] = correct_logit - max_other_logit


def accumulate_margins_with_labels(model, dataloader, device):
    """Return (margins_array, labels_array) for all samples."""
    margins, labels_list = [], []
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)                     # (batch, num_classes)
            probs = torch.nn.functional.softmax(logits, dim=1)           # (batch, num_classes)
            # Get probability of true class
            true_probs = probs[range(len(labels)), labels]   # (batch,)
            # Mask out true class to find max other probability
            probs[range(len(labels)), labels] = 0.0
            max_other_probs, _ = probs.max(dim=1)      # (batch,)
            margins_batch = true_probs - max_other_probs   # (batch,)
            margins.append(margins_batch.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    return np.concatenate(margins), np.concatenate(labels_list)
