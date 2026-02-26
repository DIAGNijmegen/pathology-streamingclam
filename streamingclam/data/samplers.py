import math
from collections.abc import Iterator
from typing import Optional, TypeVar, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import dataset
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import WeightedRandomSampler

_T_co = TypeVar("_T_co", covariant=True)
_int_classes = int

class DistributedWeightedRandomSampler(Sampler[_T_co]):
    """Sampler that applies weighted random sampling across multiple distributed processes.

    This sampler combines the behavior of `WeightedRandomSampler` (sampling based on weights)
    and `DistributedSampler` (splitting data across multiple processes).

    Args:
        weights (sequence): A sequence of weights, not necessarily summing up to one.
        num_samples (int): Total number of samples across all processes.
        num_replicas (int, optional): Number of processes in distributed training.
        rank (int, optional): Rank of the current process within num_replicas.
        replacement (bool, optional): If True, samples are drawn with replacement.
        shuffle (bool, optional): If True, shuffle indices before applying weights.
        seed (int, optional): Random seed for reproducibility.

    Example:
        >>> import torch
        >>> from torch.utils.data import DataLoader
        >>> from torch.utils.data.distributed import DistributedSampler
        >>> dataset = list(range(10))
        >>> weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        >>> sampler = DistributedWeightedRandomSampler(weights, num_samples=5, num_replicas=2, rank=0)
        >>> loader = DataLoader(dataset, sampler=sampler, batch_size=2)
        >>> for epoch in range(start_epoch, n_epochs):
        >>>     sampler.set_epoch(epoch)
        >>>     for batch in loader:
        >>>         print(batch)  # Each process gets a subset of the sampled indices
    """

    def __init__(
            self,
            weights: Sequence[float],
            num_samples: int,
            num_replicas: Optional[int] = None,
            rank: int = None,
            replacement: bool = True,
            shuffle: bool = True,
            seed: int = 0
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        if len(weights) == 0:
            raise ValueError("Weights must be a non-empty sequence.")
        if any(w < 0 for w in weights):
            raise ValueError("Weights must be non-negative.")

        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples  # total
        self.num_replicas = num_replicas
        self.num_samples_per_proc = int(math.ceil(self.num_samples * 1.0 / self.num_replicas))  # per process
        self.replacement = replacement
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Optional shuffling indices and weights before sampling
        indices = torch.arange(len(self.weights))
        if self.shuffle:
            perm = torch.randperm(len(indices), generator=g)  # Generate permutation
            indices = indices[perm]  # Shuffle indices
            weights = self.weights[perm]  # Shuffle weights in the same order
        else:
            weights = self.weights

        # Perform weighted sampling
        sampled_indices = torch.multinomial(
            weights, self.num_samples, self.replacement, generator=g
        )

        # Map sampled indices back to original dataset indices
        sampled_indices = indices[sampled_indices]

        # Distribute samples across processes. Add extra sample if samples are not perfectly divisible across ranks
        total_required = self.num_samples_per_proc * self.num_replicas
        if len(sampled_indices) < total_required:
            extra = total_required - len(sampled_indices)
            sampled_indices = torch.cat([sampled_indices, sampled_indices[:extra]])

        # Now split evenly per process
        sampled_indices = sampled_indices[self.rank:total_required:self.num_replicas]

        return iter(sampled_indices.tolist())

    def __len__(self) -> int:
        return self.num_samples_per_proc

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for deterministic shuffling."""
        self.epoch = epoch


def weighted_sampler(dataset: dataset, distributed: bool) -> WeightedRandomSampler | DistributedWeightedRandomSampler:
    """
    Weighted sampler to be used in pytorch dataloaders. Class weights are computed using inverse class frequency.
    These weights are then used to undersample majority classes, and oversample minority classes (within pytorch).

    Parameters
    ----------
    dataset : dataset
        A pytorch dataset class that contains a labels attribute, containing a list of labels.
    distributed: bool
        Indicates if DDP (distributed) training is used

    Returns
    -------
    WeightedRandomSampler class that performs balanced sampling based on the computed class weights

    """
    labels = np.array([int(label) for label in dataset.labels])

    # calculate inverse class frequency, then squash to [0,1] by dividing by max value
    _, class_counts = np.unique(labels, return_counts=True)  # List that counts how many times a unique label occurs
    inv_freq = len(labels) / class_counts
    norm_weights = inv_freq / np.max(inv_freq)

    # create weight array and replace labels by their weights
    weights = np.array(labels, dtype=np.float32)
    for i, weight in enumerate(norm_weights):
        weights[labels == i] = weight

    if distributed:
        print("Using distributed weighted random sampler")
        return DistributedWeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    else:
        print("Using weighted random sampler")
        return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)