"""Prepare ISRUC from released 200-Hz sequence arrays or the experiment cache."""
import os
from pathlib import Path
import numpy as np
from .clinical import from_cache, from_records, parser_for
from .common import read_json


def main(argv=None):
    parser = parser_for("isruc", __doc__)
    parser.add_argument("--order-manifest", type=Path, default=Path(__file__).parent / "metadata" / "isruc_sequence_order.json",
                        help="JSON mapping subject IDs to the experiment's ordered sequence filenames")
    args = parser.parse_args(argv)
    if args.source_format == "experiment-cache":
        return from_cache("isruc", args.source, args.output, alignment=args.alignment, ea_source=args.ea_source)
    order = read_json(args.order_manifest)
    if set(order) != {str(subject) for subject in range(1, 101)}:
        raise ValueError("ISRUC order manifest must contain subjects 1 through 100")
    partitions = {"train": [], "val": [], "test": []}
    for subject in range(1, 101):
        split = "train" if subject <= 80 else "val" if subject <= 90 else "test"
        folder = f"ISRUC-group1-{subject}"
        seq_dir, y_dir = args.source / "seq" / folder, args.source / "labels" / folder
        files = order[str(subject)]
        if len(files) != len(set(files)):
            raise ValueError("ISRUC order manifest contains duplicate filenames")
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
        import torch
        import torch.nn.functional as functional
        values = torch.from_numpy(np.array(cache["x"][index], copy=True)).float() / 10.
        sample = functional.avg_pool1d(values, kernel_size=2, stride=2).numpy()
        return sample, int(cache["y"][index]), subject
    return from_records("isruc", partitions, args.output, load, alignment=args.alignment)


if __name__ == "__main__":
    main()
