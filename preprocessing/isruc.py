"""Prepare ISRUC from Group I raw recordings, 200-Hz sequences, or an experiment cache."""
import os
from pathlib import Path
import re
import numpy as np
from .clinical import from_cache, from_records, parser_for
from .common import read_json, write_json

METADATA_DIR = Path(__file__).parent / "metadata"


def record_channel_layout(cache):
    """Record source labels separately from the model's nominal channel slots."""
    inventory = read_json(METADATA_DIR / "isruc_channel_inventory.json")
    manifest = read_json(cache / "manifest.json")
    manifest["channel_names_role"] = "nominal_model_slots"
    manifest["source_channel_selection"] = inventory["selection"]
    manifest["source_channels_by_subject"] = inventory["source_channels_by_subject"]
    manifest["preprocessing"].update({"bandpass_hz": [.3, 35], "notch_hz": 50,
                                      "epoch_seconds": 30, "sequence_epochs": 20,
                                      "scorer": 1, "loader_scale": .1,
                                      "downsampling": "float32 average pooling with kernel and stride 2"})
    write_json(cache / "manifest.json", manifest)
    return cache


def sequence_order(path):
    order = read_json(path)
    if set(order) != {str(subject) for subject in range(1, 101)}:
        raise ValueError("ISRUC order manifest must contain subjects 1 through 100")
    next_sequence = 0
    for subject in range(1, 101):
        files = order[str(subject)]
        matches = [re.fullmatch(rf"ISRUC-group1-{subject}-(\d+)\.npy", filename) for filename in files]
        if not files or any(match is None for match in matches):
            raise ValueError("Invalid ISRUC sequence filenames")
        sequence_ids = [int(match[1]) for match in matches]
        if sorted(sequence_ids) != list(range(next_sequence, next_sequence + len(files))):
            raise ValueError("ISRUC sequence indices must form consecutive recording blocks")
        next_sequence += len(files)
    return order


def model_input(values):
    import torch
    import torch.nn.functional as functional
    values = torch.from_numpy(np.array(values, copy=True)).float() / 10.
    return functional.avg_pool1d(values, kernel_size=2, stride=2).numpy()


def from_raw(source, output, order, alignment):
    from ._isruc_raw import prepare_recording
    inventory = read_json(METADATA_DIR / "isruc_channel_inventory.json")["source_channels_by_subject"]
    partitions = {"train": [], "val": [], "test": []}
    next_sequence = 0
    for subject in range(1, 101):
        for filename in (f"{subject}.rec", f"{subject}_1.txt"):
            if not (source / str(subject) / filename).is_file():
                raise FileNotFoundError(f"Missing ISRUC Group I recording or label file for subject {subject}")
        split = "train" if subject <= 80 else "val" if subject <= 90 else "test"
        for filename in order[str(subject)]:
            sequence = int(filename.rsplit("-", 1)[1][:-4]) - next_sequence
            partitions[split].extend((subject, sequence, epoch) for epoch in range(20))
        next_sequence += len(order[str(subject)])
    cache = {}

    def load(key):
        subject, sequence, epoch = key
        if cache.get("subject") != subject:
            cache.clear()
            sequences, labels = prepare_recording(source, subject, expected_channels=inventory[str(subject)])
            if len(sequences) != len(order[str(subject)]):
                raise ValueError(f"ISRUC subject {subject} differs from the declared sequence inventory")
            cache.update(subject=subject, x=sequences, y=labels)
            print(f"ISRUC subject {subject}: {len(sequences) * 20} epochs", flush=True)
        return model_input(cache["x"][sequence, epoch]), int(cache["y"][sequence, epoch]), subject

    return from_records("isruc", partitions, output, load, alignment=alignment)


def from_sequences(source, output, order, alignment):
    partitions = {"train": [], "val": [], "test": []}
    for subject in range(1, 101):
        split = "train" if subject <= 80 else "val" if subject <= 90 else "test"
        folder = f"ISRUC-group1-{subject}"
        seq_dir, y_dir = source / "seq" / folder, source / "labels" / folder
        files = order[str(subject)]
        if set(files) != set(os.listdir(seq_dir)) or set(files) != set(os.listdir(y_dir)):
            raise ValueError("ISRUC sequence, label, and order-manifest filenames differ")
        for filename in files:
            x = np.load(seq_dir / filename, mmap_mode="r", allow_pickle=False)
            y = np.load(y_dir / filename, mmap_mode="r", allow_pickle=False)
            if x.ndim != 3 or x.shape[1:] != (6, 6000) or y.shape != (len(x),):
                raise ValueError("Expected ISRUC [N,6,6000] sequences and N labels")
            partitions[split].extend((seq_dir / filename, y_dir / filename, i, subject) for i in range(len(x)))
    cache = {}
    def load(key):
        x_path, y_path, index, subject = key
        if cache.get("path") != x_path:
            cache.clear()
            cache.update(path=x_path, x=np.load(x_path, mmap_mode="r"), y=np.load(y_path, mmap_mode="r"))
        return model_input(cache["x"][index]), int(cache["y"][index]), subject
    return from_records("isruc", partitions, output, load, alignment=alignment)


def main(argv=None):
    parser = parser_for("isruc", __doc__, source_formats=("raw", "released", "experiment-cache"))
    parser.add_argument("--order-manifest", type=Path, default=METADATA_DIR / "isruc_sequence_order.json",
                        help="JSON mapping subject IDs to the ordered sequence filenames")
    args = parser.parse_args(argv)
    if args.source_format == "experiment-cache":
        return record_channel_layout(from_cache("isruc", args.source, args.output,
                                               alignment=args.alignment, ea_source=args.ea_source))
    if args.ea_source is not None:
        raise ValueError("--ea-source is used with --source-format experiment-cache")
    order = sequence_order(args.order_manifest)
    if args.source_format == "raw":
        return record_channel_layout(from_raw(args.source, args.output, order, args.alignment))
    return record_channel_layout(from_sequences(args.source, args.output, order, args.alignment))


if __name__ == "__main__":
    main()
