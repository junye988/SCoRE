"""Command-line arguments and entry point for EEG experiments."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from .config import DATASETS
from .runner import run


def parse_args(argv=None, default_dataset="bci_iv_2a"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default=default_dataset)
    parser.add_argument("--config", type=Path, help="Experiment YAML; defaults to the dataset configuration")
    parser.add_argument("--data-dir", type=Path, help="Directory created by the dataset preprocessing entry point")
    parser.add_argument("--output-dir", type=Path, help="Directory for checkpoints, predictions, and metrics")
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--checkpoints", nargs="+", type=Path,
                        help="Checkpoint files in model configuration order")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, help="Override data loader worker count")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, help="Override evaluation minibatch size")
    parser.add_argument("--show-config", action="store_true", help="Print the resolved configuration and exit")
    return parser.parse_args(argv)


def main(argv=None, default_dataset="bci_iv_2a"):
    return run(parse_args(argv, default_dataset=default_dataset))
