"""Prepare BCI-IV-2a from official GDF/MAT files or an experiment cache."""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import numpy as np

from utils.io import read_json, sha256_file
from .cache import CacheWriter, SPLITS, first_seen_groups


def from_cache(source, output):
    source = Path(source)
    metadata = read_json(source / "manifest.json")
    if metadata.get("format") != "bciciv2a_reve_paper_v1":
        raise ValueError("Expected the BCI-IV-2a session-EA cache")
    operators = np.load(source / "alignment_operators.npy")
    subjects = np.load(source / "alignment_subjects.npy")
    sessions = np.load(source / "alignment_sessions.npy")
    lookup = {}
    for subject, session, operator in zip(subjects, sessions, operators):
        a = np.asarray(operator, dtype=np.float64)
        lookup[(int(subject), int(session))] = np.stack((a, np.linalg.inv(a)))
    writer = CacheWriter(output, "BCI-IV-2a", metadata["channels"], 200,
                         ["left hand", "right hand", "feet", "tongue"], samples_aligned=True,
                         preprocessing={"epoch_seconds_from_trial_start": [2, 6], "bandpass_hz": [.5, 99.5],
                                        "filter": "Butterworth order 5, zero phase", "alignment_unit": "subject/session"})
    for split in SPLITS:
        x = np.load(source / f"{split}_samples.npy", mmap_mode="r")
        y = np.load(source / f"{split}_labels.npy")
        s = np.load(source / f"{split}_subjects.npy")
        r = np.load(source / f"{split}_sessions.npy")
        keys, groups = first_seen_groups(list(zip(s.tolist(), r.tolist())))
        pairs = np.stack([lookup[key] for key in keys])
        writer.add_split(split, x, y, groups, pairs, source_hash=sha256_file(source / f"{split}_samples.npy"))
    return writer.finish()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-format", choices=("raw", "experiment-cache"), default="raw")
    parser.add_argument("--labels", type=Path, help="Directory containing A01T.mat through A09E.mat")
    args = parser.parse_args(argv)
    if args.source_format == "experiment-cache":
        return from_cache(args.source, args.output)
    if args.labels is None:
        parser.error("--labels is required for official GDF input")
    from .readers.bci_iv_2a import build_reve_cache
    with tempfile.TemporaryDirectory(prefix="score-bci-") as work:
        cache = Path(work) / "cache"
        build_reve_cache(args.source, args.labels, cache)
        return from_cache(cache, args.output)


if __name__ == "__main__":
    main()
