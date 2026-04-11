from typing import Dict, List, Tuple, Union

import torch


def estimate_instance_hardness(
        batch_indices: torch.Tensor,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        predicted: torch.Tensor,
        hardness_estimates: Dict[Tuple[int, int], Dict[str, List[Union[int, List[float]]]]],
        epoch: int,
        remembering: List[bool],
        dataset_model_id: Tuple[int, int]
):
    """Estimate hardness through AUM, DataIQ, and Forgetting. In our work we use AUM as the default estimator for
    resampling and pruning. This function is used in train_ensemble.py and is called only when running
    train_baseline_models.py."""

    for index_within_batch, (i, x, logits, correct_label) in enumerate(zip(batch_indices, inputs, outputs, labels)):
        i = i.item()
        correct_label = correct_label.item()
        predicted_label = predicted[index_within_batch].item()

        logits = logits.detach()
        correct_logit = logits[correct_label].item()
        probs = torch.nn.functional.softmax(logits, dim=0)
        # AUM
        max_other_logit = torch.max(torch.cat((logits[:correct_label], logits[correct_label + 1:]))).item()
        hardness_estimates[dataset_model_id]['AUM'][i][epoch] = correct_logit - max_other_logit
        # DataIQ (aleatoric uncertainty)
        p_y = probs[correct_label].item()
        hardness_estimates[dataset_model_id]['DataIQ'][i][epoch] = p_y * (1 - p_y)
        # Forgetting
        if predicted_label == correct_label:
            remembering[i] = True
        elif predicted_label != correct_label and remembering[i]:
            hardness_estimates[dataset_model_id]['Forgetting'][i] += 1
            remembering[i] = False
