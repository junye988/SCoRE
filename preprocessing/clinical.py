"""Prepare clinical EEG inputs while preserving their supplied partitions."""
from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

from utils.io import read_json, sha256_file
from .alignment import fit_group_alignment
from .cache import CacheWriter, SPLITS

DATASETS = {
    "isruc": {"name": "ISRUC", "channels": ["F3", "C3", "O1", "F4", "C4", "O2"], "samples": 3000, "rate": 100,
              "classes": ["0", "1", "2", "3", "4"], "default_alignment": "ea", "unit": "subject"},
    "hmc": {"name": "HMC", "channels": ["F4", "C4", "O2", "C3"], "samples": 3000, "rate": 100,
            "classes": ["Wake", "N1", "N2", "N3", "REM"], "default_alignment": "none", "unit": "recording"},
    "mat": {"name": "MAT", "channels": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "T7", "T8", "C3", "C4", "P7", "P8", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz"],
            "samples": 500, "rate": 100, "classes": ["baseline", "mental arithmetic"], "default_alignment": "none", "unit": "subject"},
    "mumtaz": {"name": "Mumtaz", "channels": ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"],
               "samples": 1000, "rate": 200, "classes": ["healthy", "MDD"], "default_alignment": "ea", "unit": "recording"},
}


def make_writer(dataset, output, alignment):
    spec = DATASETS[dataset]
    return CacheWriter(output, spec["name"], spec["channels"], spec["rate"], spec["classes"], samples_aligned=False,
                       preprocessing={"alignment": alignment, "alignment_unit": spec["unit"],
                                      "eigenvalue_floor_relative": 1e-12 if alignment == "ea" else None})


def pairs_for(samples, groups, alignment):
    if alignment == "none":
        return np.broadcast_to(np.eye(samples.shape[1], dtype=np.float64), (int(groups.max()) + 1, 2, samples.shape[1], samples.shape[1])).copy()
    return fit_group_alignment(samples, groups)


def from_cache(dataset, source, output, *, alignment, ea_source=None):
    source = Path(source)
    manifest = read_json(source / "manifest.json")
    spec = DATASETS[dataset]
    formats = {"isruc": "isruc_reference_float32_100hz_v1", "hmc": "hmc_reference_float32_100hz_v1",
               "mat": "mentalarith_reference_float32_100hz_v1", "mumtaz": "mumtaz_reve_release_float32_200hz_v1"}
    if manifest.get("format") != formats[dataset]:
        raise ValueError(f"Expected source cache format {formats[dataset]}")
    if manifest.get("num_channels") != len(spec["channels"]) or manifest.get("num_samples") != spec["samples"]:
        raise ValueError("Source cache channel/time dimensions differ from this experiment")
    if manifest.get("alignment_applied") or manifest.get("samples_aligned"):
        raise ValueError("This importer expects samples before EA")
    if ea_source is not None and alignment != "ea":
        raise ValueError("--ea-source requires --alignment ea")
    writer = make_writer(dataset, output, alignment)
    for split in SPLITS:
        x = np.load(source / f"{split}_samples.npy", mmap_mode="r", allow_pickle=False)
        y = np.load(source / f"{split}_labels.npy", allow_pickle=False)
        g = np.load(source / f"{split}_group_indices.npy", allow_pickle=False)
        if ea_source is None:
            pairs = pairs_for(x, g, alignment)
        else:
            ea_source = Path(ea_source)
            pairs = np.load(ea_source / f"{split}_alignment_pairs.npy", allow_pickle=False)
            if not np.array_equal(g, np.load(ea_source / f"{split}_group_indices.npy", allow_pickle=False)):
                raise ValueError("EA artifacts use different epoch-group assignments")
        writer.add_split(split, x, y, g, pairs, source_hash=sha256_file(source / f"{split}_samples.npy"))
    return writer.finish()


def from_records(dataset, partitions, output, load_record, *, alignment):
    """Convert one sample at a time, preserving each supplied partition order."""
    spec = DATASETS[dataset]
    writer = make_writer(dataset, output, alignment)
    group_sets = {}
    with tempfile.TemporaryDirectory(prefix="score-preprocess-") as work:
        for split in SPLITS:
            keys = partitions[split]
            shape = (len(keys), len(spec["channels"]), spec["samples"])
            x = np.lib.format.open_memmap(Path(work) / f"{split}.npy", mode="w+", dtype=np.float32, shape=shape)
            y, g = np.empty(len(keys), dtype=np.int64), np.empty(len(keys), dtype=np.int64)
            lookup = {}
            for index, key in enumerate(keys):
                sample, label, group = load_record(key)
                if sample.shape != shape[1:] or not np.isfinite(sample).all():
                    raise ValueError(f"Unexpected {spec['name']} sample shape or values")
                if group not in lookup:
                    lookup[group] = len(lookup)
                x[index], y[index], g[index] = sample, label, lookup[group]
            x.flush()
            group_sets[split] = set(lookup)
            for other in group_sets:
                if other != split and group_sets[other] & group_sets[split]:
                    raise ValueError("A recording/subject group occurs in multiple partitions")
            writer.add_split(split, x, y, g, pairs_for(x, g, alignment))
            del x
    return writer.finish()


def parser_for(dataset, description, *, source_formats=("released", "experiment-cache")):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-format", choices=source_formats, default="released")
    parser.add_argument("--alignment", choices=("ea", "none"), default=DATASETS[dataset]["default_alignment"])
    parser.add_argument("--ea-source", type=Path, help="Existing frozen EA artifact directory for an experiment cache")
    return parser
