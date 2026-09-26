"""Read Mental Arithmetic EDF recordings in the published partition order."""
import re

import numpy as np


def read_records(source):
    import mne
    names = ["EEG " + name for name in ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "T3", "T4", "C3", "C4", "T5", "T6", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz", "A2-A1"]]
    partitions = {"train": [], "val": [], "test": []}
    files = sorted(source.glob("*.edf"))
    if not files:
        raise ValueError("Expected MentalArithmetic Subject00_1.edf through Subject35_2.edf")
    for path in files:
        match = re.fullmatch(r"Subject(\d+)_(1|2)", path.stem)
        if match is None or not 0 <= int(match[1]) <= 35:
            raise ValueError("Unexpected MentalArithmetic EDF filename")
        subject, label = int(match[1]), int(match[2]) - 1
        split = "train" if subject <= 27 else "val" if subject <= 31 else "test"
        raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
        try:
            raw.pick(names)
            raw.reorder_channels(names)
            raw.filter(l_freq=.5, h_freq=45., verbose="ERROR")
            if int(round(raw.info["sfreq"])) != 100:
                raw.resample(100, verbose="ERROR")
            signal = raw.get_data(units="uV").astype(np.float32)
        finally:
            raw.close()
        signal = signal[:, :signal.shape[1] // 500 * 500]
        trials = signal.reshape(20, -1, 500).transpose(1, 0, 2)
        for trial in trials:
            trial = trial.astype(np.float32)
            trial = trial - trial.mean(axis=-1, keepdims=True)
            partitions[split].append((trial[:19], label, subject))
    return partitions
