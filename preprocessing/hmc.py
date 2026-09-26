"""Prepare HMC from official EDF recordings or preprocessed 100-Hz epochs."""
import pickle
import re
from pathlib import Path
import numpy as np
from utils.file_io import read_json
from .recording_import import DATASETS, import_experiment_cache, import_record_partitions, create_preprocessing_parser


def build_from_raw(source, output, *, alignment):
    from .raw_readers.hmc import read_labeled_epochs

    source = Path(source)
    if (source / "recordings").is_dir():
        source = source / "recordings"
    metadata_path = Path(__file__).parent / "metadata" / "hmc_recording_splits.json"
    metadata = read_json(metadata_path)
    partitions = {}
    seen = set()
    for split in ("train", "val", "test"):
        recordings = metadata["recordings"][split]
        if len(recordings) != len(set(recordings)) or seen.intersection(recordings):
            raise ValueError("HMC recording partitions must be unique and disjoint")
        seen.update(recordings)
        keys = []
        for recording in recordings:
            for filename in (recording + ".edf", recording + "_sleepscoring.txt"):
                if not (source / filename).is_file():
                    raise FileNotFoundError(source / filename)
            count = metadata["epoch_counts"][split][recording]
            keys.extend((recording, index) for index in range(count))
        partitions[split] = sorted(keys, key=lambda key: f"{key[0]}-{key[1]}.pkl")

    expected_counts = {
        recording: count
        for split in metadata["epoch_counts"].values()
        for recording, count in split.items()
    }
    cached_recording, cached_epochs, cached_labels = None, None, None

    def load(key):
        nonlocal cached_recording, cached_epochs, cached_labels
        recording, index = key
        if recording != cached_recording:
            cached_epochs, cached_labels = None, None
            cached_epochs, cached_labels = read_labeled_epochs(source / f"{recording}.edf")
            if len(cached_labels) != expected_counts[recording]:
                raise ValueError(f"Unexpected number of labeled epochs in {recording}")
            cached_recording = recording
            print(f"HMC {recording}: {len(cached_labels)} epochs", flush=True)
        return cached_epochs[index], int(cached_labels[index]), recording

    return import_record_partitions("hmc", partitions, output, load, alignment=alignment)


def main(argv=None):
    parser = create_preprocessing_parser("hmc", __doc__, source_formats=("raw", "released", "experiment-cache"))
    parser.set_defaults(source_format="raw")
    args = parser.parse_args(argv)
    if args.source_format == "raw":
        return build_from_raw(args.source, args.output, alignment=args.alignment)
    if args.source_format == "experiment-cache":
        return import_experiment_cache("hmc", args.source, args.output, alignment=args.alignment, ea_source=args.ea_source)
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

    return import_record_partitions("hmc", partitions, args.output, load, alignment=args.alignment)


if __name__ == "__main__":
    main()
