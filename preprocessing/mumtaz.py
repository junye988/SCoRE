"""Prepare Mumtaz from raw EDF recordings, REVE LMDB, or an experiment cache."""
import pickle
import re
from pathlib import Path
import tempfile
import numpy as np
from .clinical import from_cache, from_records, parser_for
from .cache import SPLITS


def from_raw(source, output, alignment):
    import mne
    import scipy
    from .readers.mumtaz import prepare_recording, split_filenames
    from .clinical import make_writer, pairs_for
    partitions = split_filenames(source)
    if any(not files for files in partitions.values()):
        raise ValueError("Mumtaz raw directory must contain the complete EC/EO recording collection")
    writer = make_writer("mumtaz", output, alignment)
    writer.manifest["preprocessing"].update({"source": "REVE released EDF pipeline", "resample_hz": 200,
                                            "bandpass_hz": [.3, 30], "notch_hz": 50, "window_seconds": 5,
                                            "loader_scale": .01,
                                            "versions": {"mne": mne.__version__, "numpy": np.__version__,
                                                         "scipy": scipy.__version__}})
    with tempfile.TemporaryDirectory(prefix="score-mumtaz-") as work:
        for split in SPLITS:
            paths, counts = [], []
            for group, filename in enumerate(partitions[split]):
                samples = prepare_recording(source / filename)
                path = Path(work) / f"{split}-{group}.npy"
                np.save(path, samples)
                paths.append(path)
                counts.append(len(samples))
                del samples
            n = sum(counts)
            x = np.lib.format.open_memmap(Path(work) / f"{split}.npy", mode="w+", dtype=np.float32, shape=(n, 19, 1000))
            labels, groups, offset = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64), 0
            for group, (filename, path, count) in enumerate(zip(partitions[split], paths, counts)):
                x[offset:offset + count] = np.load(path, mmap_mode="r")
                labels[offset:offset + count] = int("MDD" in filename)
                groups[offset:offset + count] = group
                offset += count
            x.flush()
            writer.add_split(split, x, labels, groups, pairs_for(x, groups, alignment))
            print(f"Mumtaz {split}: {n} epochs", flush=True)
            del x
    return writer.finish()


def main(argv=None):
    args = parser_for("mumtaz", __doc__, source_formats=("raw", "released", "experiment-cache")).parse_args(argv)
    if args.source_format == "raw":
        return from_raw(args.source, args.output, args.alignment)
    if args.source_format == "experiment-cache":
        return from_cache("mumtaz", args.source, args.output, alignment=args.alignment, ea_source=args.ea_source)
    import lmdb
    db = lmdb.open(str(args.source), readonly=True, lock=False, readahead=False, subdir=args.source.is_dir())
    try:
        with db.begin() as transaction:
            partitions = pickle.loads(transaction.get(b"__keys__"))
            def load(key):
                data = pickle.loads(transaction.get(key.encode() if isinstance(key, str) else key))
                match = re.fullmatch(r"(.+)_(\d+)", key)
                if match is None:
                    raise ValueError("Expected EDF_basename_window-index Mumtaz keys")
                sample = np.asarray(data["sample"])
                if sample.shape != (19, 1000):
                    raise ValueError("Expected REVE Mumtaz samples [19,1000]")
                return np.ascontiguousarray(sample / 100., dtype=np.float32), int(data["label"]), match[1]
            return from_records("mumtaz", partitions, args.output, load, alignment=args.alignment)
    finally:
        db.close()


if __name__ == "__main__":
    main()
