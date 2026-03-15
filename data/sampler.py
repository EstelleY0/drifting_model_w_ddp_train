import torch
import numpy as np
from torch.utils.data import Sampler


class ClassBalancedSampler(Sampler):
    """
    Sampler ensuring balanced class distribution per batch
    Samples a subset of classes and fixed number of samples per class
    """

    def __init__(self, labels, n_classes_per_batch=10, n_samples_per_class=32):
        self.labels = torch.as_tensor(labels)
        self.n_classes = len(torch.unique(self.labels))
        self.n_classes_per_batch = min(n_classes_per_batch, self.n_classes)
        self.n_samples_per_class = n_samples_per_class

        # Pre-index samples for each class
        self.class_indices = {
            c: torch.where(self.labels == c)[0].tolist()
            for c in range(self.n_classes)
        }

    def __iter__(self):
        all_classes = list(range(self.n_classes))
        while True:
            # Select classes to participate in this batch
            batch_classes = np.random.choice(all_classes, self.n_classes_per_batch, replace=False)
            batch_indices = []
            for c in batch_classes:
                idx_list = self.class_indices[c]
                replace = len(idx_list) < self.n_samples_per_class
                sampled = np.random.choice(idx_list, self.n_samples_per_class, replace=replace)
                batch_indices.extend(sampled.tolist())
            yield from batch_indices

    def __len__(self):
        return len(self.labels)
