"""Convert anonymous HandMI and SSVEP-EEG windows to split-level arrays."""
from pathlib import Path
import numpy as np

from utils.io import sha256_file
from .cache import CacheWriter, SPLITS


def convert_windows(source, output, dataset):
    source = Path(source)
    if source.is_dir():
        candidates = ("mi_windows.npz", "windows.npz") if dataset == "HandMI" else ("ssvep_windows.npz", "fiveclass_windows.npz")
        source = next((source / name for name in candidates if (source / name).is_file()), None)
        if source is None:
            raise FileNotFoundError(f"Expected one of {candidates}")
    with np.load(source, allow_pickle=False) as archive:
        x, labels = archive["aligned_windows"], archive["y"]
        subjects, a, inverse = archive["subject_index"], archive["A"], archive["A_inv"]
        channels, permutation = archive["channel_names"].tolist(), archive["mirror_permutation"]
    if set(np.unique(subjects).tolist()) != set(range(12)):
        raise ValueError("Expected anonymous subjects S001 through S012")
    if x.dtype != np.float32 or x.shape[1:] != (32, 400):
        raise ValueError("Expected float32 [N,32,400] windows")
    names = ["left hand", "right hand"] if dataset == "HandMI" else ["5 Hz", "7.5 Hz", "12 Hz", "15 Hz", "rest"]
    writer = CacheWriter(output, dataset, channels, 200, names, samples_aligned=True, permutation=permutation,
                         preprocessing={"alignment_unit": "subject", "ridge_relative": 1e-4, "eigenvalue_floor_relative": 1e-6,
                                        "alignment_fit": "complete imagery trials" if dataset == "HandMI" else "disjoint stimulation/rest intervals",
                                        "window_seconds": 2, "bandpass_hz": [4, 40] if dataset == "HandMI" else [3, 45]})
    ranges = (range(8), range(8, 10), range(10, 12))
    source_hash = sha256_file(source)
    for split, selected in zip(SPLITS, ranges):
        selected = np.asarray(list(selected))
        indices = np.flatnonzero(np.isin(subjects, selected))
        groups = np.searchsorted(selected, subjects[indices])
        pairs = np.stack((a[selected], inverse[selected]), axis=1)
        writer.add_split(split, x, labels[indices], groups, pairs, indices=indices, source_hash=source_hash)
    return writer.finish()
