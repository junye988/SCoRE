"""Portable memory-mapped datasets and frozen Euclidean alignment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np

CACHE_FORMAT = "score_eeg_v1"
SPLITS = ("train", "val", "test")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mirror_permutation(channels):
    """Pair numbered electrodes, leaving midline/unpaired electrodes fixed."""
    names = [str(name).replace(".", "").lower() for name in channels]
    lookup = {name: i for i, name in enumerate(names)}
    if len(lookup) != len(names):
        raise ValueError("Channel names must be unique")
    indices = []
    for i, name in enumerate(names):
        match = re.fullmatch(r"([a-z]+)([0-9]+)", name)
        partner = name
        if match:
            number = int(match[2])
            partner = f"{match[1]}{number + 1 if number % 2 else number - 1}"
        indices.append(lookup.get(partner, i))
    indices = np.asarray(indices, dtype=np.int64)
    if not np.array_equal(indices[indices], np.arange(len(indices))):
        raise ValueError("Electrode mapping must be an involution")
    return indices


def alignment_pair(reference, eigenvalue_floor=1e-12):
    """Return [A, inverse(A)] for A = mean(XX^T)^(-1/2)."""
    reference = np.asarray(reference, dtype=np.float64)
    reference = .5 * (reference + reference.T)
    if reference.ndim != 2 or reference.shape[0] != reference.shape[1] or not np.isfinite(reference).all():
        raise ValueError("Alignment reference must be a finite square matrix")
    if not 0 < eigenvalue_floor < 1:
        raise ValueError("Relative eigenvalue floor must lie in (0,1)")
    values, vectors = np.linalg.eigh(reference)
    if values[-1] <= 0:
        raise ValueError("Alignment reference must have positive signal power")
    operator = (vectors / np.sqrt(np.maximum(values, values[-1] * eigenvalue_floor))) @ vectors.T
    return np.stack((operator, np.linalg.inv(operator)))


def fit_group_alignment(samples, groups, *, chunk_size=256, eigenvalue_floor=1e-12):
    """Fit each group independently using its complete unlabeled epoch set."""
    if samples.ndim != 3 or len(samples) != len(groups) or len(samples) == 0:
        raise ValueError("Expected nonempty [N,C,T] samples and N group indices")
    groups = np.asarray(groups, dtype=np.int64)
    if groups.min() < 0 or not np.array_equal(np.unique(groups), np.arange(groups.max() + 1)):
        raise ValueError("Group indices must be contiguous from zero")
    sums = np.zeros((groups.max() + 1, samples.shape[1], samples.shape[1]), dtype=np.float64)
    for start in range(0, len(samples), chunk_size):
        work = np.asarray(samples[start:start + chunk_size], dtype=np.float64)
        if not np.isfinite(work).all():
            raise ValueError("EEG contains non-finite values")
        np.add.at(sums, groups[start:start + len(work)], work @ work.transpose(0, 2, 1))
    references = sums / np.bincount(groups)[:, None, None]
    return np.stack([alignment_pair(reference, eigenvalue_floor) for reference in references])


def first_seen_groups(values):
    lookup, keys, indices = {}, [], []
    for value in values:
        key = tuple(value) if isinstance(value, (list, np.ndarray)) else value
        if key not in lookup:
            lookup[key] = len(keys)
            keys.append(key)
        indices.append(lookup[key])
    return keys, np.asarray(indices, dtype=np.int64)


class CacheWriter:
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
