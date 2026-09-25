"""Memory-mapped EEG datasets with group-specific alignment operators."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    def __init__(self, root, split):
        self.root = Path(root)
        self.split = split
        with (self.root / "manifest.json").open(encoding="utf-8") as stream:
            self.manifest = json.load(stream)
        self.samples = np.load(self.root / f"{split}_samples.npy", mmap_mode="r")
        self.labels = np.load(self.root / f"{split}_labels.npy", mmap_mode="r")
        self.groups = np.load(self.root / f"{split}_group_indices.npy", mmap_mode="r")
        split_pairs = self.root / f"{split}_alignment_pairs.npy"
        self.pairs = np.load(split_pairs if split_pairs.exists() else self.root / "alignment_pairs.npy")
        if self.samples.ndim != 3 or len(self.samples) != len(self.labels) or len(self.labels) != len(self.groups):
            raise ValueError(f"Inconsistent {split} sample, label, or group dimensions")
        channels = self.samples.shape[1]
        if self.pairs.shape[1:] != (2, channels, channels):
            raise ValueError("Alignment pairs must have shape [groups, 2, channels, channels]")
        if len(self.groups) and (self.groups.min() < 0 or self.groups.max() >= len(self.pairs)):
            raise ValueError("Sample group indices exceed the alignment table")
        self.aligned = bool(self.manifest.get("samples_aligned", True))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = np.array(self.samples[index], copy=True)
        pair = np.asarray(self.pairs[int(self.groups[index])], dtype=np.float64)
        if not self.aligned:
            sample = pair[0] @ sample.astype(np.float64)
        return {
            "sample": torch.from_numpy(sample),
            "label": int(self.labels[index]),
            "alignment_pair": torch.from_numpy(pair.copy()),
            "index": int(index),
        }


class ArraySplitSampler:
    """Balanced-size minibatches from a NumPy permutation of training indices."""

    def __init__(self, size, batch_size):
        self.size = int(size)
        self.batch_size = int(batch_size)

    def __iter__(self):
        order = np.random.permutation(self.size)
        yield from (part.tolist() for part in np.array_split(order, len(self)))

    def __len__(self):
        return max(1, int(np.ceil(self.size / self.batch_size)))
