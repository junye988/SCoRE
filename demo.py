"""Download the BCI-IV-2a demo data and run the configured experiment."""

from experiments.run import parse_args
from experiments.experiment import run_experiment
from preprocessing.dataset_download import resolve_dataset_directory


def main(argv=None):
    args = parse_args(argv, default_dataset="bci_iv_2a")
    if not args.show_config:
        splits = ("test",) if args.mode == "evaluate" else ("train", "val", "test")
        args.data_dir = resolve_dataset_directory(args.dataset, args.data_dir, splits=splits)
    return run_experiment(args)


if __name__ == "__main__":
    main()
