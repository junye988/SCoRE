"""Prepare ISRUC-SLEEP Group I recordings as 20-epoch sequences.

Adapted from CBraMod preprocessing/ISRUC/prepare_ISRUC_1.py.
Copyright (c) 2025 Jiquan Wang. Distributed under the MIT license
in licenses/CBraMod.txt.
"""
from pathlib import Path
import os
import shutil
import tempfile

import numpy as np

LABEL_MAP = {"0": 0, "1": 1, "2": 2, "3": 3, "5": 4}


def prepare_recording(source, subject, expected_channels=None):
    """Return the six protocol-selected channels and first-scorer labels at 200 Hz."""
    import mne
    source = Path(source)
    recording = source / str(subject) / f"{subject}.rec"
    label_path = source / str(subject) / f"{subject}_1.txt"
    with tempfile.TemporaryDirectory(prefix="score-isruc-") as work:
        # ISRUC's .rec files contain EDF data. Expose the unchanged bytes with
        # an .edf suffix for MNE's public reader.
        edf_path = Path(work) / "recording.edf"
        try:
            os.link(recording, edf_path)
        except OSError:
            shutil.copyfile(recording, edf_path)
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
        try:
            if float(raw.info["sfreq"]) != 200:
                raise ValueError("ISRUC Group I recordings must be sampled at 200 Hz")
            if len(raw.ch_names) < 8:
                raise ValueError("ISRUC recording must contain channel positions 2 through 7")
            if expected_channels is not None and raw.ch_names[2:8] != list(expected_channels):
                raise ValueError(f"ISRUC subject {subject} channel labels differ from the source inventory")
            raw.filter(.3, 35, fir_design="firwin", n_jobs=1, verbose="ERROR")
            raw.notch_filter(50, n_jobs=1, verbose="ERROR")
            values = raw.to_data_frame().values[:, 1:][:, 2:8]
        finally:
            raw.close()
    remainder_samples = len(values) % 6000
    if remainder_samples:
        values = values[:-remainder_samples]
    epochs = values.reshape(-1, 6000, 6)
    remainder_epochs = len(epochs) % 20
    if remainder_epochs:
        epochs = epochs[:-remainder_epochs]
    sequences = epochs.reshape(-1, 20, 6000, 6).transpose(0, 1, 3, 2)
    with label_path.open() as stream:
        labels = np.asarray([LABEL_MAP[line.strip()] for line in stream if line.strip()], dtype=np.int64)
    if remainder_epochs:
        labels = labels[:-remainder_epochs]
    labels = labels.reshape(-1, 20)
    if len(sequences) != len(labels):
        raise ValueError("ISRUC signal and label sequence counts differ")
    return sequences, labels
