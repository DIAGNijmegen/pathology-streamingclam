import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import dataset


def weighted_sampler(dataset: dataset) -> WeightedRandomSampler:
    labels = np.array([int(label) for label in dataset.labels])

    # more generalized approach, should result in the same distribution
    # calculate inverse class frequency, then squash to [0,1] by dividing by max value
    _, class_counts = np.unique(labels, return_counts=True)
    inv_freq = len(labels) / class_counts
    norm_weights = inv_freq / np.max(inv_freq)

    # create weight array and replace labels by their weights
    weights = np.array(labels, dtype=np.float32)
    for i, weight in enumerate(norm_weights):
        weights[labels == i] = weight

    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
