"""Prepare MAT from its preprocessed 100-Hz LMDB or the experiment cache."""
import pickle
import numpy as np
from .clinical import DATASETS, from_cache, from_records, parser_for


def from_raw(source, output, alignment):
    from .readers.mat import read_records
    return from_records("mat", read_records(source), output, lambda value: value, alignment=alignment)


def main(argv=None):
    args = parser_for("mat", __doc__, source_formats=("raw", "released", "experiment-cache")).parse_args(argv)
    if args.source_format == "raw":
        return from_raw(args.source, args.output, args.alignment)
    if args.source_format == "experiment-cache":
        return from_cache("mat", args.source, args.output, alignment=args.alignment, ea_source=args.ea_source)
    import lmdb
    db = lmdb.open(str(args.source), readonly=True, lock=False, readahead=False, subdir=args.source.is_dir())
    try:
        with db.begin() as transaction:
            partitions = pickle.loads(transaction.get(b"__keys__"))
            def load(key):
                data = pickle.loads(transaction.get(key.encode() if isinstance(key, str) else key))
                field = next(name for name in ("sample", "signal", "X") if name in data)
                x = np.asarray(data[field], dtype=np.float32).reshape(20, -1)
                if x.shape != (20, 500) or int(round(float(data.get("fs", 100)))) != 100:
                    raise ValueError("Expected MAT preprocessed [20,500] inputs at 100 Hz")
                aliases = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
                defaults = ["EEG " + name for name in ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "T3", "T4", "C3", "C4", "T5", "T6", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz", "A2-A1"]]
                keep, names = [], []
                for i, name in enumerate(data.get("ch_names", defaults)):
                    name = str(name).removeprefix("EEG ").strip()
                    if name.upper() in ("A2-A1", "A1-A2"):
                        continue
                    keep.append(i)
                    names.append(aliases.get(name, name))
                if names != DATASETS["mat"]["channels"]:
                    raise ValueError("MAT scalp channel order differs from the experiment")
                label = np.asarray(data.get("label", data.get("y"))).reshape(-1)[0]
                return x[keep], int(label), int(data["subject_id"])
            return from_records("mat", partitions, args.output, load, alignment=args.alignment)
    finally:
        db.close()


if __name__ == "__main__":
    main()
