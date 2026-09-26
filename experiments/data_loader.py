"""PyTorch dataset adapters and experiment minibatch sampling."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from preprocessing.dataset_cache import PreparedDatasetSplit
from utils.random_seed import seed_dataloader_worker


class CachedEEGDataset(Dataset):
    """Convert a prepared cache's samples to training tensors."""

    def __init__(self, root, split):
        self.cache = PreparedDatasetSplit(root, split)
        self.root = self.cache.root
        self.split = self.cache.split
        self.manifest = self.cache.manifest
        self.samples = self.cache.samples
        self.labels = self.cache.labels
        self.groups = self.cache.groups
        self.pairs = self.cache.pairs

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, index):
        item = self.cache.read(index)
        item["sample"] = torch.from_numpy(item["sample"])
        item["alignment_pair"] = torch.from_numpy(item["alignment_pair"])
        return item


class ArraySplitBatchSampler:
    """Balanced-size minibatches from a NumPy permutation of training indices."""

    def __init__(self, size, batch_size):
        self.size = int(size)
        self.batch_size = int(batch_size)

    def __iter__(self):
        order = np.random.permutation(self.size)
        yield from (part.tolist() for part in np.array_split(order, len(self)))

    def __len__(self):
        return max(1, int(np.ceil(self.size / self.batch_size)))


def build_dataloader(dataset, settings, seed, *, training=False):
    batch_size = int(settings.get("batch_size", 64) if training else
                     settings.get("eval_batch_size", settings.get("batch_size", 64)))
    common = dict(num_workers=int(settings.get("num_workers", 0)),
                  pin_memory=torch.cuda.is_available(), worker_init_fn=seed_dataloader_worker)
    generator = torch.Generator().manual_seed(seed)
    if settings.get("batch_order") == "numpy_array_split":
        batches = ArraySplitBatchSampler(len(dataset), batch_size) if training else [
            part.tolist() for part in np.array_split(
                np.arange(len(dataset)), max(1, int(np.ceil(len(dataset) / batch_size))),
            )
        ]
        return DataLoader(dataset, batch_sampler=batches, generator=generator, **common)
    sampler = None
    if training and settings.get("balanced_sampler", False):
        labels = np.asarray(dataset.labels, dtype=np.int64)
        counts = np.bincount(labels)
        weights = counts[labels].astype(np.float64) ** -float(settings.get("sampler_power", 1))
        sampler = WeightedRandomSampler(torch.as_tensor(weights), len(labels), replacement=True,
                                        generator=generator)
    return DataLoader(dataset, batch_size=batch_size, shuffle=training and sampler is None,
                      sampler=sampler, generator=generator, drop_last=False, **common)
