"""HMC EDF preprocessing for the 100-Hz sleep-staging experiment."""

from pathlib import Path

import mne
import numpy as np
import pandas as pd


CHANNELS = ["EEG F4-M1", "EEG C4-M1", "EEG O2-M1", "EEG C3-M2"]
LABELS = {
    "Sleep stage W": 0,
    "Sleep stage N1": 1,
    "Sleep stage N2": 2,
    "Sleep stage N3": 3,
    "Sleep stage R": 4,
}


def read_labeled_epochs(edf_path):
    """Return labeled 30-second epochs, preserving annotation order."""
    edf_path = Path(edf_path)
    annotation_path = edf_path.with_name(edf_path.stem + "_sleepscoring.txt")
    annotations = pd.read_csv(annotation_path)
    annotations.columns = [str(column).strip() for column in annotations.columns]
    required = {"Recording onset", "Duration", "Annotation"}
    if not required.issubset(annotations.columns):
        raise ValueError(f"Missing HMC sleep-scoring columns in {annotation_path.name}")

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    try:
        missing = set(CHANNELS).difference(raw.ch_names)
        if missing:
            raise ValueError(f"Missing HMC EEG channels: {sorted(missing)}")
        raw.pick(CHANNELS)
        raw.reorder_channels(CHANNELS)
        raw.filter(l_freq=0.3, h_freq=45.0, n_jobs=1, verbose="ERROR")
        raw.notch_filter(freqs=[50.0], n_jobs=1, verbose="ERROR")
        raw.resample(100.0, n_jobs=1, verbose="ERROR")
        signals = raw.get_data(units="uV").astype(np.float32)
        times = raw.times.astype(np.float64)
    finally:
        raw.close()

    epochs, labels = [], []
    for _, row in annotations.iterrows():
        label = str(row["Annotation"]).strip()
        if abs(float(row["Duration"]) - 30.0) > 1e-6 or label not in LABELS:
            continue
        start = int(np.searchsorted(times, float(row["Recording onset"]), side="left"))
        end = start + 3000
        if end > signals.shape[1]:
            continue
        epoch = signals[:, start:end]
        if epoch.shape == (4, 3000):
            epochs.append(epoch.astype(np.float32))
            labels.append(LABELS[label])
    if not epochs:
        raise ValueError(f"No labeled HMC epochs in {edf_path.name}")
    return np.stack(epochs), np.asarray(labels, dtype=np.int64)
