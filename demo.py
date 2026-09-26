"""Download the BCI-IV-2a demo data and run the configured experiment."""

from experiments.cli import parse_args
from experiments.runner import run
from preprocessing.acquisition import resolve_data_dir


def main(argv=None):
    args = parse_args(argv, default_dataset="bci_iv_2a")
    if not args.show_config:
        splits = ("test",) if args.mode == "evaluate" else ("train", "val", "test")
        args.data_dir = resolve_data_dir(args.dataset, args.data_dir, splits=splits)
    return run(args)


if __name__ == "__main__":
    main()
