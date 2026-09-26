"""Resolve dataset experiments and per-member model configurations."""

from copy import deepcopy
from pathlib import Path

from model import ModelConfig
from utils.config import load_yaml

DATASETS = ("bci_iv_2a", "physionet_mi", "isruc", "hmc", "mat", "mumtaz", "handmi", "ssvep_eeg", "aestheeg")
CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "experiments"


def load_config(args):
    path = args.config or CONFIG_ROOT / f"{args.dataset}.yaml"
    config = load_yaml(path)
    if config["dataset"] != args.dataset:
        raise ValueError("--dataset and the experiment configuration must agree")
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers
    if args.eval_batch_size is not None:
        config["training"]["eval_batch_size"] = args.eval_batch_size
    return config


def member_config(config, member):
    values = deepcopy(config["model"])
    values["ensemble"] = {"enabled": False, "size": 1}
    values.update(member.get("model_overrides", member.get("model", {})))
    for name in ("reflection_init", "reflection_init_strength", "reflection_init_seed",
                 "input_precision", "initial_reflection_precision"):
        if name in member:
            values[name] = member[name]
    return ModelConfig.from_dict(values)
