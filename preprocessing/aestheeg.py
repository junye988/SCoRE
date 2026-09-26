"""Prepare four-class AesthEEG from released subject epochs or the experiment cache."""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import numpy as np

from utils.file_io import read_json, sha256_file
from .alignment import alignment_pair
from .dataset_cache import DatasetCacheWriter, SPLITS, encode_groups_in_order

CLASS_NAMES = ["landscape ugly", "landscape beautiful", "face ugly", "face beautiful"]
FIXED_SPLIT = {
    "train": "001 002 003 010 011 013 015 016 017 021 022 023 024 025 027 028 031 032 033 034 036 037 038 039 040 042 043 044 045 046 047 048 049 050 051 054 056 057 058 060".split(),
    "val": "006 008 009 012 014 018 029 041 052 055".split(),
    "test": "004 005 007 019 020 026 030 035 053 059".split(),
}


def import_experiment_cache(source, output):
    source = Path(source)
    manifest = read_json(source / "cache_manifest.json")
    if manifest.get("format") != "fsaes_reference_four_class_subject_ea_float64_v1":
        raise ValueError("Expected the four-class AesthEEG float64 subject-EA cache")
    x = np.load(source / "aligned_windows.npy", mmap_mode="r", allow_pickle=False)
    with np.load(source / "metadata.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("y", "subject_index", "split", "A", "A_inv", "channels", "mirror_permutation")}
    writer = DatasetCacheWriter(output, "AesthEEG", data["channels"].tolist(), 250, CLASS_NAMES, samples_aligned=True,
                         permutation=data["mirror_permutation"], preprocessing={"epoch_seconds": [-.5, 2.5], "alignment_unit": "subject",
                                                                                 "eigenvalue_floor_relative": 1e-12})
    for code, split in enumerate(SPLITS):
        indices = np.flatnonzero(data["split"] == code)
        keys, groups = encode_groups_in_order(data["subject_index"][indices].tolist())
        pairs = np.stack([np.stack((data["A"][i], data["A_inv"][i])) for i in keys])
        writer.add_split(split, x, data["y"][indices], groups, pairs, indices=indices,
                         source_hash=sha256_file(source / "metadata.npz"))
    return writer.finish()


def import_subject_epochs(source, output):
    source = Path(source)
    root = source / "subjects" if (source / "subjects").is_dir() else source
    records, channels = {}, None
    for split in SPLITS:
        for subject in FIXED_SPLIT[split]:
            path = root / f"sub-{subject}.npz"
            with np.load(path, allow_pickle=True) as archive:
                category = np.asarray([str(name).strip().lower() for name in archive["category"]])
                binary = np.asarray(archive["y"], dtype=np.int64)
                these_channels = archive["channels"].astype(str).tolist()
                if str(archive["subject"].item()) != subject or str(archive["split"].item()) != split:
                    raise ValueError("AesthEEG subject/split metadata differ from the released partition")
                if float(archive["sfreq"].item()) != 250 or not np.isin(binary, (0, 1)).all():
                    raise ValueError("Expected 250-Hz epochs with binary aesthetic labels")
            if channels is None:
                channels = these_channels
            elif channels != these_channels:
                raise ValueError("Channel order differs across AesthEEG subjects")
            keep = np.isin(category, ("face", "landscape"))
            records[subject] = (path, keep, binary[keep] + 2 * (category[keep] == "face"))
    writer = DatasetCacheWriter(output, "AesthEEG", channels, 250, CLASS_NAMES, samples_aligned=True,
                         preprocessing={"epoch_seconds": [-.5, 2.5], "alignment_unit": "subject",
                                        "alignment_fit": "all released subject epochs before category eligibility", "eigenvalue_floor_relative": 1e-12})
    with tempfile.TemporaryDirectory(prefix="score-aestheeg-") as work:
        for split in SPLITS:
            subjects = sorted(FIXED_SPLIT[split])
            count = sum(int(records[s][1].sum()) for s in subjects)
            x = np.lib.format.open_memmap(Path(work) / f"{split}.npy", mode="w+", dtype=np.float64, shape=(count, 32, 751))
            labels, groups, pairs, offset = [], [], [], 0
            for group, subject in enumerate(subjects):
                path, keep, y = records[subject]
                with np.load(path, allow_pickle=False) as archive:
                    samples = np.asarray(archive["X"], dtype=np.float64)
                if samples.shape != (len(keep), 32, 751):
                    raise ValueError("Expected AesthEEG subject epochs [N,32,751]")
                reference = np.einsum("nct,ndt->cd", samples, samples, optimize=True) / len(samples)
                pair = alignment_pair(reference)
                x[offset:offset + len(y)] = pair[0] @ samples[keep]
                labels.extend(y.tolist())
                groups.extend([group] * len(y))
                pairs.append(pair)
                offset += len(y)
            x.flush()
            writer.add_split(split, x, np.asarray(labels, dtype=np.int64), np.asarray(groups, dtype=np.int64), np.stack(pairs))
            del x
    return writer.finish()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-format", choices=("raw", "subject-epochs", "experiment-cache"), default="raw")
    args = parser.parse_args(argv)
    if args.source_format == "experiment-cache":
        return import_experiment_cache(args.source, args.output)
    if args.source_format == "subject-epochs":
        return import_subject_epochs(args.source, args.output)
    from types import SimpleNamespace
    from .raw_readers.aestheeg import find_subject_dir, prepare_subject_epochs
    subjects_root = args.source / "subjects" if (args.source / "subjects").is_dir() else args.source
    with tempfile.TemporaryDirectory(prefix="score-aestheeg-raw-") as work:
        options = SimpleNamespace(out_dir=Path(work), l_freq=.5, h_freq=45., notch_freq=50.,
                                  resample_sfreq=250., tmin=-.5, tmax=2.5, baseline_start=-.5, baseline_end=0.,
                                  reject_eeg_uv=None, max_event_trial_diff=2, overwrite=False)
        for split in SPLITS:
            for subject in FIXED_SPLIT[split]:
                exact = subjects_root / subject
                directory = exact if exact.is_dir() else find_subject_dir(subjects_root, subject)
                summary, _ = prepare_subject_epochs(subject, split, directory, options)
                if summary.status != "ok":
                    raise RuntimeError(f"AesthEEG subject {subject}: {summary.reason}")
        return import_subject_epochs(Path(work), args.output)


if __name__ == "__main__":
    main()
