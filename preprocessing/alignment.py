"""Frozen Euclidean alignment and electrode reflection transforms."""
from __future__ import annotations

import re

import numpy as np


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
