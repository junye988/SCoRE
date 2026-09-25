"""Prepare PhysioNet-MI from raw EDF, REVE-format LMDB, or an aligned cache."""
from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import re
import tempfile

import numpy as np

from .common import CacheWriter, SPLITS, alignment_pair, first_seen_groups, read_json, sha256

CHANNELS = "FC5 FC3 FC1 FCz FC2 FC4 FC6 C5 C3 C1 Cz C2 C4 C6 CP5 CP3 CP1 CPz CP2 CP4 CP6 Fp1 Fpz Fp2 AF7 AF3 AFz AF4 AF8 F7 F5 F3 F1 Fz F2 F4 F6 F8 FT7 FT8 T7 T8 T9 T10 TP7 TP8 P7 P5 P3 P1 Pz P2 P4 P6 P8 PO7 PO3 POz PO4 PO8 O1 Oz O2 Iz".split()
RUNS = (4, 6, 8, 10, 12, 14)
CLASSES = ["left fist", "right fist", "both fists", "both feet"]
COUNTS = {"train": 6300, "val": 1734, "test": 1803}


def source_key(key):
    match = re.fullmatch(r"S(\d{3})R(\d{2})-(\d+)", key)
    if match is None:
        raise ValueError("Expected PhysioNet SxxxRxx-epoch source keys")
    subject, run, epoch = map(int, match.groups())
    if not 1 <= subject <= 109 or run not in RUNS:
        raise ValueError("Unexpected PhysioNet subject or imagery run")
    return subject, RUNS.index(run), epoch


def from_cache(source, output):
    source = Path(source)
    manifest = read_json(source / "manifest.json")
    if manifest.get("format") not in ("physionet_subject_reve_v1", "physionet_reve_release_ea_v1", "physionet_reve_release_subject_ea_v1"):
        raise ValueError("Expected a PhysioNet-MI aligned experiment cache")
    pairs = np.load(source / "alignment_pairs.npy", allow_pickle=False)
    channels = manifest.get("channels", CHANNELS)
    writer = CacheWriter(output, "PhysioNet-MI", channels, 200, CLASSES, samples_aligned=True,
                         preprocessing={"source_format": manifest["format"], "alignment": manifest.get("alignment", {}).get("fitting_unit", "recording")})
    for split in SPLITS:
        x = np.load(source / f"{split}_aligned.npy", mmap_mode="r", allow_pickle=False)
        y = np.load(source / f"{split}_labels.npy", allow_pickle=False)
        subjects = np.load(source / f"{split}_subjects.npy", allow_pickle=False)
        sessions = np.load(source / f"{split}_sessions.npy", allow_pickle=False)
        keys, groups = first_seen_groups(list(zip(subjects.tolist(), sessions.tolist())))
        selected_pairs = np.stack([pairs[s, r] for s, r in keys])
        writer.add_split(split, x, y, groups, selected_pairs, source_hash=sha256(source / f"{split}_aligned.npy"))
    return writer.finish()


def from_lmdb(source, output, alignment_unit="recording"):
    import lmdb
    source = Path(source)
    db = lmdb.open(str(source), readonly=True, lock=False, readahead=False, subdir=source.is_dir())
    try:
        with db.begin() as transaction:
            partitions = pickle.loads(transaction.get(b"__keys__"))
            if {split: len(partitions[split]) for split in SPLITS} != COUNTS:
                raise ValueError("Released PhysioNet-MI partition sizes differ from the benchmark")
            writer = CacheWriter(output, "PhysioNet-MI", CHANNELS, 200, CLASSES, samples_aligned=True,
                                 preprocessing={"source": "REVE released LMDB", "loader_scale": .01, "alignment_unit": alignment_unit,
                                                "eigenvalue_floor_relative": 1e-12})
            for split in SPLITS:
                keys = partitions[split]
                identities = [source_key(key) for key in keys]
                if any(("train" if s <= 70 else "val" if s <= 89 else "test") != split for s, _, _ in identities):
                    raise ValueError("Released PhysioNet subject partitions differ from 1--70/71--89/90--109")
                units = [(s, r) if alignment_unit == "recording" else (s, 0) for s, r, _ in identities]
                group_keys, groups = first_seen_groups(units)
                x = np.empty((len(keys), 64, 800), dtype=np.float32)
                y = np.empty(len(keys), dtype=np.int64)
                operators = []
                for group in range(len(group_keys)):
                    indices = np.flatnonzero(groups == group)
                    payloads = [pickle.loads(transaction.get(keys[i].encode())) for i in indices]
                    samples = np.stack([item["sample"] for item in payloads]).astype(np.float64) / 100.
                    if samples.shape[1:] != (64, 800) or not np.isfinite(samples).all():
                        raise ValueError("Expected finite REVE PhysioNet samples [64,800]")
                    reference = np.einsum("nct,ndt->cd", samples, samples, optimize=True) / len(samples)
                    pair = alignment_pair(reference)
                    x[indices] = np.einsum("cd,ndt->nct", pair[0], samples, optimize=True).astype(np.float32)
                    y[indices] = [int(item["label"]) for item in payloads]
                    operators.append(pair)
                writer.add_split(split, x, y, groups, np.stack(operators))
            return writer.finish()
    finally:
        db.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-format", choices=("raw", "released", "experiment-cache"), default="raw")
    parser.add_argument("--alignment-unit", choices=("recording", "subject"), default="recording")
    args = parser.parse_args(argv)
    if args.source_format == "experiment-cache":
        return from_cache(args.source, args.output)
    if args.source_format == "released":
        return from_lmdb(args.source, args.output, args.alignment_unit)
    from ._physionet_raw import prepare
    with tempfile.TemporaryDirectory(prefix="score-physionet-") as work:
        prepare(args.source, work)
        return from_lmdb(work, args.output, args.alignment_unit)


if __name__ == "__main__":
    main()
