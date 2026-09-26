"""Train, evaluate, and record the configured model on one dataset."""

import json
from pathlib import Path
import time

import numpy as np
import torch

from model import EnsembleConfig, SCoRE, WarmupModel, aggregate_logits
from utils.checkpoints import checkpoint_state
from utils.config import format_yaml, save_yaml
from utils.io import write_json, sha256_file
from utils.metrics import classification_metrics
from utils.reproducibility import seed_all
from .config import load_config, member_config
from .data import EEGDataset, make_loader
from .evaluation import predict
from .training import fit_stage


def validate_inputs(config, datasets):
    """Check the prepared montage and dimensions against the experiment."""
    model = config["model"]
    inputs = config.get("input", {})
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
        if manifest.get("sample_rate") != inputs.get("sampling_rate_hz"):
            raise ValueError("Dataset sampling rate differs from the experiment configuration")
        if manifest.get("mirror_permutation") != model.get("mirror_permutation"):
            raise ValueError("Dataset channel correspondence differs from the model configuration")
        if inputs.get("channel_names") is not None and manifest.get("channels") != inputs["channel_names"]:
            raise ValueError("Dataset channel order differs from the experiment configuration")
        labels = dataset.labels
        if not len(labels) or labels.min() < 0 or labels.max() >= model["num_classes"]:
            raise ValueError(f"Invalid or empty {split} class labels")
        if inputs.get("require_identity_alignment", False):
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
    write_json(output / "training.json", dict(seed=seed, warmup=warmup_result, joint=joint_result))
    return model, output / "joint_best.pt"


def run(args):
    config = load_config(args)
    if args.show_config:
        print(format_yaml(config), end="")
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
    save_yaml(output / "config.yaml", config)
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
                                   checkpoint_sha256=sha256_file(path), metrics=metrics))
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
                  configuration_sha256=sha256_file(output / "config.yaml"),
                  dataset_manifest_sha256=sha256_file(args.data_dir / "manifest.json"),
                  elapsed_seconds=time.perf_counter() - started,
                  versions={"pytorch": torch.__version__, "numpy": np.__version__})
    write_json(output / "results.json", result)
    print(json.dumps({"dataset": args.dataset, "balanced_accuracy_percent": 100 * metrics["balanced_accuracy"],
                      "results": str(output / "results.json")}), flush=True)
    return result
