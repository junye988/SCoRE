"""Prepare HMC from preprocessed 100-Hz pickle epochs or the experiment cache."""
import pickle
import re
import numpy as np
from .clinical import DATASETS, from_cache, from_records, parser_for


def main(argv=None):
    args = parser_for("hmc", __doc__).parse_args(argv)
    if args.source_format == "experiment-cache":
        return from_cache("hmc", args.source, args.output, alignment=args.alignment, ea_source=args.ea_source)
    partitions = {split: sorted((args.source / directory).glob("*.pkl")) for split, directory in (("train", "train"), ("val", "eval"), ("test", "test"))}
    if any(not values for values in partitions.values()):
        raise ValueError("Expected train/eval/test folders containing preprocessed pickle epochs")

    def load(path):
        with path.open("rb") as stream:
            data = pickle.load(stream)
        match = re.fullmatch(r"(SN\d+)-(\d+)\.pkl", path.name)
        if match is None or data["record_id"] != match[1] or data["epoch_index"] != int(match[2]):
            raise ValueError("HMC epoch filename and recording metadata disagree")
        if list(data["ch_names"]) != DATASETS["hmc"]["channels"] or float(data["fs"]) != 100:
            raise ValueError("Expected HMC F4/C4/O2/C3 at 100 Hz")
        label = np.asarray(data["y"]).reshape(-1)[0]
        return np.asarray(data["X"], dtype=np.float32), int(label), match[1]

    return from_records("hmc", partitions, args.output, load, alignment=args.alignment)


if __name__ == "__main__":
    main()
