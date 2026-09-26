"""Train and evaluate SCoRE on a configured EEG dataset."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from model import EnsembleConfig, ModelConfig, SCoRE, WarmupModel, aggregate_logits
from .data import EEGDataset
from .metrics import classification_metrics
from .training import fit_stage, make_loader, predict, save_json, seed_all


DATASETS = ("bci_iv_2a", "physionet_mi", "isruc", "hmc", "mat", "mumtaz", "handmi", "ssvep_eeg", "aestheeg")
CONFIG_ROOT = Path(__file__).resolve().parent / "configs"


def parse_args(argv=None, default_dataset="bci_iv_2a"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default=default_dataset)
    parser.add_argument("--config", type=Path, help="Experiment JSON; defaults to the dataset configuration")
    parser.add_argument("--data-dir", type=Path, help="Directory created by the dataset preprocessing entry point")
    parser.add_argument("--output-dir", type=Path, help="Directory for checkpoints, predictions, and metrics")
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--checkpoints", nargs="+", type=Path,
                        help="One checkpoint per configured ensemble member, in configuration order")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, help="Override data loader worker count")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, help="Override evaluation minibatch size")
    parser.add_argument("--show-config", action="store_true", help="Print the resolved configuration and exit")
    return parser.parse_args(argv)


def load_config(args):
    path = args.config or CONFIG_ROOT / f"{args.dataset}.json"
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
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


def checkpoint_state(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError(f"No model state dictionary in {path}")
    return state


def checkpoint_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_inputs(config, datasets):
    """Check the prepared montage and dimensions against the experiment."""
    model = config["model"]
    data = config.get("data", {})
    expected_shape = [model["num_channels"], model["num_samples"]]
    for split, dataset in datasets.items():
        manifest = dataset.manifest
        if manifest.get("format") != "score_eeg_v1" or manifest.get("status") != "complete":
            raise ValueError("Use a complete dataset produced by the preprocessing modules")
        shape = list(dataset.samples.shape[1:])
        if shape != expected_shape:
            raise ValueError(f"{split} input shape {shape} differs from {expected_shape}")
        if manifest.get("num_classes") != model["num_classes"]:
            raise ValueError("Dataset class count differs from the model configuration")
        if manifest.get("sample_rate") != data.get("sampling_rate_hz"):
            raise ValueError("Dataset sampling rate differs from the experiment configuration")
        if manifest.get("mirror_permutation") != model.get("mirror_permutation"):
            raise ValueError("Dataset channel correspondence differs from the model configuration")
        if data.get("channel_names") is not None and manifest.get("channels") != data["channel_names"]:
            raise ValueError("Dataset channel order differs from the experiment configuration")
        labels = dataset.labels
        if not len(labels) or labels.min() < 0 or labels.max() >= model["num_classes"]:
            raise ValueError(f"Invalid or empty {split} class labels")
        if data.get("alignment") == "none":
            identity = np.eye(model["num_channels"], dtype=np.float64)
            if not np.array_equal(dataset.pairs, np.broadcast_to(identity, dataset.pairs.shape)):
                raise ValueError("This experiment requires identity alignment operators")


def train_member(model_config, member, settings, datasets, output, device):
    seed = int(member["seed"])
    deterministic = bool(settings.get("deterministic", True))
    seed_all(seed, deterministic)
    warmup = WarmupModel(model_config).to(device)
    warm_seed = seed + int(settings.get("warmup_loader_seed_offset", 20000))
    train_loader = make_loader(datasets["train"], settings, warm_seed, training=True)
    val_loader = make_loader(datasets["val"], settings, seed + 10000)
    warmup_result = fit_stage(warmup, train_loader, val_loader, settings, output, device, warmup=True)
    _, differences = predict(warmup, make_loader(datasets["train"], settings, seed + 30000),
                             device, amp=bool(settings.get("pca_amp", settings.get("amp", False))),
                             diagnostics=True)
    offset = int(settings.get("joint_seed_offset", 730019))
    reset = settings.get("joint_seed_reset", "after_model")
    if reset not in ("before_model", "after_model"):
        raise ValueError("joint_seed_reset must be before_model or after_model")
    build_seed = seed + (offset if reset == "before_model" else int(settings.get("joint_model_seed_offset", 0)))
    seed_all(build_seed, deterministic)
    model = SCoRE(model_config).to(device)
    model.initialize_from_warmup(warmup, torch.from_numpy(differences))
    del warmup, differences, train_loader
    if reset == "after_model":
        seed_all(seed + offset, deterministic)
    train_loader = make_loader(datasets["train"], settings, seed, training=True)
    joint_result = fit_stage(model, train_loader, val_loader, settings, output, device, warmup=False)
    save_json(output / "training.json", dict(seed=seed, warmup=warmup_result, joint=joint_result))
    return model, output / "joint_best.pt"


def run(args):
    config = load_config(args)
    if args.show_config:
        print(json.dumps(config, indent=2))
        return config
    if args.data_dir is None:
        raise ValueError("Supply --data-dir with the preprocessed dataset directory")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu or install a CUDA-enabled PyTorch build")
    output = args.output_dir or Path("runs") / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    settings = config["training"]
    ensemble_values = config["model"].get("ensemble", {})
    ensemble = EnsembleConfig(**ensemble_values)
    members = list(ensemble_values.get("members", []))
    if not members:
        members = [dict(name="seed_0042", seed=42)]
    if len(members) != ensemble.size:
        raise ValueError("Ensemble size differs from the number of configured members")
    checkpoints = args.checkpoints
    if args.mode == "evaluate" and checkpoints is None:
        checkpoints = [output / member.get("name", f"member_{i + 1:02d}") / "joint_best.pt"
                       for i, member in enumerate(members)]
    if checkpoints is not None and len(checkpoints) != len(members):
        raise ValueError("Provide exactly one checkpoint for each configured ensemble member")
    if args.mode == "train" and checkpoints is not None:
        raise ValueError("Use --mode evaluate with --checkpoints")
    splits = ("train", "val", "test") if args.mode == "train" else ("test",)
    datasets = {split: EEGDataset(args.data_dir, split) for split in splits}
    validate_inputs(config, datasets)
    manifest = datasets["test"].manifest
    sample_shape = list(datasets["test"].samples.shape[1:])
    saved_config = deepcopy(config)
    save_json(output / "config.json", saved_config)
    test_loader = make_loader(datasets["test"], settings, 0)
    member_outputs = []
    member_results = []
    started = time.perf_counter()
    test_labels = None
    for index, member in enumerate(members):
        model_config = member_config(config, member)
        member_name = member.get("name", f"member_{index + 1:02d}")
        if Path(member_name).name != member_name:
            raise ValueError("Member names must be single directory names")
        member_output = output / member_name
        if args.mode == "train":
            if (member_output / "joint_best.pt").exists():
                raise FileExistsError(f"Checkpoint already exists in {member_output}. Use a new output directory")
            model, path = train_member(model_config, member, settings, datasets, member_output, device)
        else:
            path = checkpoints[index]
            model = SCoRE(model_config).to(device)
            model.load_state_dict(checkpoint_state(path), strict=True)
        labels, logits = predict(model, test_loader, device, amp=bool(settings.get("amp", False)))
        if test_labels is not None and not np.array_equal(labels, test_labels):
            raise ValueError("Ensemble members evaluated different sample orders")
        test_labels = labels
        member_outputs.append(logits)
        metrics = classification_metrics(labels, logits)
        member_results.append(dict(name=member_name, seed=int(member["seed"]),
                                   checkpoint_sha256=checkpoint_hash(path), metrics=metrics))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    stacked = np.stack(member_outputs)
    combined = aggregate_logits(torch.from_numpy(stacked), ensemble).cpu().numpy()
    metrics = classification_metrics(test_labels, combined)
    np.savez_compressed(output / "test_predictions.npz", labels=test_labels, logits=combined,
                        member_logits=stacked)
    result = dict(dataset=args.dataset, mode=args.mode, ensemble=ensemble_values,
                  metrics=metrics, members=member_results, samples=len(test_labels),
                  sample_rate=manifest.get("sample_rate"), input_shape=sample_shape,
                  configuration_sha256=checkpoint_hash(output / "config.json"),
                  dataset_manifest_sha256=checkpoint_hash(args.data_dir / "manifest.json"),
                  elapsed_seconds=time.perf_counter() - started,
                  versions={"pytorch": torch.__version__, "numpy": np.__version__})
    reference = config.get("evaluation", {}).get("target_balanced_accuracy_percent")
    if reference is not None:
        result["reference_balanced_accuracy_percent"] = float(reference)
        result["difference_percentage_points"] = 100 * metrics["balanced_accuracy"] - float(reference)
    save_json(output / "results.json", result)
    print(json.dumps({"dataset": args.dataset, "balanced_accuracy_percent": 100 * metrics["balanced_accuracy"],
                      "results": str(output / "results.json")}), flush=True)
    return result


def main(argv=None, default_dataset="bci_iv_2a"):
    return run(parse_args(argv, default_dataset=default_dataset))


if __name__ == "__main__":
    main()
