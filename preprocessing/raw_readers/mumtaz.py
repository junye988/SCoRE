"""Mumtaz EDF preparation following REVE's released implementation.

Adapted from REVE preprocessing/preprocessing_mumtaz.py at commit
06a7059a07c3dabd80aee60c3dbc1eca4bdbe1c7, itself adapted from CBraMod.
Copyright (c) 2026 BRAIN - BRoader Artificial INtelligence.
Distributed under the MIT license in licenses/REVE.txt.
"""
from pathlib import Path
import numpy as np

CHANNELS = ["EEG " + name + "-LE" for name in
            "Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 Fz Cz Pz".split()]


def split_filenames(source):
    names = [path.name for path in Path(source).iterdir() if path.name.endswith(".edf") and "TASK" not in path.name]
    healthy = sorted(name for name in names if "MDD" not in name)
    depressed = sorted(name for name in names if "MDD" in name)
    return {"train": healthy[:40] + depressed[:42],
            "val": healthy[40:48] + depressed[42:52],
            "test": healthy[48:] + depressed[52:]}


def read_epoch_windows(path):
    import mne
    raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
    try:
        raw.pick_channels(CHANNELS, ordered=True, verbose="ERROR")
        raw.resample(200, verbose="ERROR")
        raw.filter(l_freq=.3, h_freq=30, verbose="ERROR")
        raw.notch_filter(50, verbose="ERROR")
        eeg = raw.to_data_frame().values[:, 1:]
    finally:
        raw.close()
    remainder = eeg.shape[0] % 1000
    if remainder:
        eeg = eeg[:-remainder]
    eeg = eeg.reshape(-1, 5, 200, 19).transpose(0, 3, 1, 2).reshape(-1, 19, 1000)
    return np.ascontiguousarray(eeg / 100., dtype=np.float32)
