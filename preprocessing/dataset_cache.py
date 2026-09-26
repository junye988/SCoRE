"""Read and write prepared EEG arrays and their alignment metadata."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.file_io import read_json, write_json
from .alignment import mirror_permutation

CACHE_FORMAT = "score_eeg_v1"
SPLITS = ("train", "val", "test")


class PreparedDatasetSplit:
    """Read prepared split arrays and apply their frozen alignment operators."""

    def __init__(self, root, split):
        self.root = Path(root)
        self.split = split
        self.manifest = read_json(self.root / "manifest.json")
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

    def read(self, index):
        sample = np.array(self.samples[index], copy=True)
        pair = np.asarray(self.pairs[int(self.groups[index])], dtype=np.float64)
        if not self.aligned:
            sample = pair[0] @ sample.astype(np.float64)
        return {
            "sample": sample,
            "label": int(self.labels[index]),
            "alignment_pair": pair.copy(),
            "index": int(index),
        }


def encode_groups_in_order(values):
    lookup, keys, indices = {}, [], []
    for value in values:
        key = tuple(value) if isinstance(value, (list, np.ndarray)) else value
        if key not in lookup:
            lookup[key] = len(keys)
            keys.append(key)
        indices.append(lookup[key])
    return keys, np.asarray(indices, dtype=np.int64)


class DatasetCacheWriter:
    """Write split arrays with explicit alignment state and source hashes."""

    def __init__(self, output, dataset, channels, sample_rate, class_names, *, samples_aligned,
                 permutation=None, preprocessing=None):
        self.root = Path(output)
        self.root.mkdir(parents=True, exist_ok=False)
        self.manifest = {"format": CACHE_FORMAT, "dataset": dataset, "status": "building",
                         "channels": list(channels), "num_channels": len(channels),
                         "sample_rate": sample_rate, "class_names": list(class_names),
                         "num_classes": len(class_names), "samples_aligned": bool(samples_aligned),
                         "mirror_permutation": list(map(int, mirror_permutation(channels) if permutation is None else permutation)),
                         "counts": {}, "splits": {}, "preprocessing": preprocessing or {}}
        write_json(self.root / "manifest.json", self.manifest)

    def add_split(self, split, samples, labels, groups, pairs, *, indices=None, source_hash=None):
        if split not in SPLITS:
            raise ValueError(f"Unknown partition {split}")
        labels, groups, pairs = np.asarray(labels), np.asarray(groups), np.asarray(pairs)
        if indices is None:
            indices = np.arange(len(samples))
        indices = np.asarray(indices, dtype=np.int64)
        n, channels, times = len(indices), samples.shape[1], samples.shape[2]
        if channels != self.manifest["num_channels"]:
            raise ValueError("Sample channels differ from the declared montage")
        if labels.shape != (n,) or groups.shape != (n,) or n == 0:
            raise ValueError("A split needs nonempty matching sample, label and group arrays")
        if labels.dtype.kind not in "iu" or not np.isin(labels, np.arange(self.manifest["num_classes"])).all():
            raise ValueError("Class indices are outside the dataset label range")
        if pairs.ndim != 4 or pairs.shape[1:] != (2, channels, channels):
            raise ValueError("Alignment pairs must have shape [groups,2,C,C]")
        if groups.min() < 0 or groups.max() >= len(pairs) or not np.isfinite(pairs).all():
            raise ValueError("Invalid alignment group assignments or operators")
        target = np.lib.format.open_memmap(self.root / f"{split}_samples.npy", mode="w+",
                                          dtype=samples.dtype, shape=(n, channels, times))
        for start in range(0, n, 256):
            block = np.asarray(samples[indices[start:start + 256]])
            if not np.isfinite(block).all():
                raise ValueError("EEG contains non-finite values")
            target[start:start + len(block)] = block
        target.flush()
        del target
        np.save(self.root / f"{split}_labels.npy", labels.astype(np.int64))
        np.save(self.root / f"{split}_group_indices.npy", groups.astype(np.int64))
        np.save(self.root / f"{split}_alignment_pairs.npy", pairs)
        self.manifest["counts"][split] = n
        self.manifest["num_samples"] = times
        self.manifest["splits"][split] = {"count": n, "num_groups": len(pairs),
                                           "sample_dtype": str(samples.dtype), "alignment_dtype": str(pairs.dtype),
                                           "class_counts": np.bincount(labels, minlength=self.manifest["num_classes"]).tolist()}
        if source_hash:
            self.manifest["splits"][split]["source_sha256"] = source_hash

    def finish(self):
        if set(self.manifest["counts"]) != set(SPLITS):
            raise ValueError("Cache must contain train, val and test partitions")
        self.manifest["status"] = "complete"
        self.manifest["total_samples"] = sum(self.manifest["counts"].values())
        write_json(self.root / "manifest.json", self.manifest)
        return self.root
