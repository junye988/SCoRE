"""BCI-IV-2a demonstration and command-line access to the EEG experiments."""

from experiments.demo_data import prepare_demo_data
from experiments.run import parse_args, run


def main(argv=None):
    args = parse_args(argv, default_dataset="bci_iv_2a")
    prepare_demo_data(args)
    return run(args)


if __name__ == "__main__":
    main()
