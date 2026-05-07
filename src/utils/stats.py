import numpy as np


def compute_mean_std(hardness_estimates, num_models, estimator):
    """
    Compute per‑sample mean and std across the first `num_models` models.

    Args:
        hardness_estimates: dict, keys are model indices (int or tuple),
                            values are dict {estimator: list of hardness per sample}
        num_models: int, number of models to include
        estimator: str, which estimator to use ('AUM', 'DataIQ', 'Forgetting')

    Returns:
        mean: np.ndarray, shape (n_samples,)
        std: np.ndarray, shape (n_samples,)
    """
    all_hardness = []
    for model_id in range(num_models):
        # Try both possible key formats
        try:
            est_dict = hardness_estimates[(0, model_id)]
        except (KeyError, TypeError):
            est_dict = hardness_estimates[model_id]
        all_hardness.append(est_dict[estimator])
    arr = np.array(all_hardness)          # (num_models, n_samples)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0, ddof=1)     # sample std
    return mean, std