"""Two-stage optimization, checkpoint selection, and training history."""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from utils.io import write_json
from utils.metrics import classification_metrics
from .augmentation import aligned_mixup, augment
from .batches import batch_to_device, forward_batch
from .evaluation import predict


def optimizer_for(model, settings, *, warmup):
    backbone_lr = float(settings.get("warmup_backbone_lr", settings.get("bootstrap_backbone_lr",
                        settings.get("backbone_lr", 3e-4))) if warmup else settings.get("backbone_lr", 3e-4))
    head_lr = float(settings.get("warmup_head_lr", settings.get("bootstrap_head_lr",
                    settings.get("head_lr", 1e-3))) if warmup else settings.get("head_lr", 1e-3))
    encoder = list(model.backbone.temporal.parameters()) + list(model.backbone.spatial.parameters())
    encoder += list(model.backbone.norm.parameters())
    head = list(model.backbone.embedding.parameters()) + list(model.classifier.parameters())
    weight_decay = float(settings.get("weight_decay", 1e-3))
    groups = [dict(params=encoder, lr=backbone_lr, weight_decay=weight_decay),
              dict(params=head, lr=head_lr, weight_decay=weight_decay)]
    if not warmup:
        axis = list(model.output_reflection_parameters())
        if axis:
            groups.append(dict(params=axis, lr=float(settings.get("axis_lr", 1e-3)),
                               weight_decay=float(settings.get("axis_weight_decay", 0))))
        groups.append(dict(params=list(model.input_reflection_parameters()),
                           lr=float(settings.get("reflection_lr", 3e-4)),
                           weight_decay=float(settings.get("reflection_weight_decay", 0))))
    for group in groups:
        group["base_lr"] = group["lr"]
    covered = [id(p) for group in groups for p in group["params"]]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if len(covered) != len(set(covered)) or set(covered) != expected:
        raise ValueError("Optimizer parameter groups must cover the model exactly once")
    return torch.optim.AdamW(groups, betas=tuple(settings.get("optimizer_betas", (0.9, 0.999))))


def learning_rate_factor(index, maximum, settings, *, warmup, steps):
    if warmup or settings.get("schedule", "cosine" if steps else "constant") == "constant":
        return 1.0
    epoch_ramp = settings.get("lr_warmup_epochs", settings.get("warmup_epochs", 0)
                              if "bootstrap_epochs" in settings else 0)
    ramp = int(settings.get("lr_warmup_steps", 0) if steps else epoch_ramp)
    if ramp and index < ramp:
        return (index + 1) / ramp
    total = maximum if steps else int(settings.get("cosine_epochs", maximum))
    progress = min(1.0, max(0.0, (index - ramp) / max(1, total - ramp - 1)))
    floor = float(settings.get("min_lr_ratio", 0))
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def fit_stage(model, train_loader, val_loader, settings, directory, device, *, warmup):
    stage = "warmup" if warmup else "joint"
    steps = settings.get("mode", "epochs") == "steps"
    if steps:
        maximum = int(settings.get("bootstrap_max_steps", 1500) if warmup else settings.get("max_steps", 20000))
        interval = int(settings.get("bootstrap_val_every", 200) if warmup else settings.get("val_every", 200))
        minimum = 1
    else:
        maximum = int(settings.get("bootstrap_epochs", settings.get("warmup_epochs", 12)) if warmup else settings.get("epochs", 100))
        interval = 1
        minimum = int(settings.get("warmup_min_epochs", 1) if warmup else settings.get("min_epochs", 1))
    patience = int((settings.get("warmup_patience", settings.get("bootstrap_patience", 3))
                    if warmup else settings.get("patience", 12)) or 0)
    checkpoint_rule = settings.get("warmup_checkpoint", "best") if warmup else "best"
    if checkpoint_rule not in ("best", "last"):
        raise ValueError("warmup_checkpoint must be best or last")
    if checkpoint_rule == "last":
        patience = 0
    if maximum < 1 or interval < 1:
        raise ValueError("Training duration and validation interval must be positive")
    optimizer = optimizer_for(model, settings, warmup=warmup)
    labels = np.asarray(train_loader.dataset.labels, dtype=np.int64)
    power = float(settings.get("classweight_power", 0) or 0)
    counts = np.bincount(labels, minlength=model.config.num_classes)
    if np.any(counts == 0):
        raise ValueError("Every configured class must occur in the training split")
    weights = counts.astype(np.float64) ** -power
    weights /= weights.mean()
    criterion_weights = torch.as_tensor(weights, device=device, dtype=torch.float32) if power else None
    criterion = nn.CrossEntropyLoss(weight=criterion_weights,
                                    label_smoothing=float(settings.get("label_smoothing", 0.05)))
    use_amp = bool(settings.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_key = None
    best_state = None
    best_position = 0
    stale = 0
    position = 0
    updates = 0
    epoch = 0
    skipped = 0
    history = []
    begin = time.perf_counter()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / f"{stage}_best.pt"

    def validate():
        nonlocal best_key, best_state, best_position, stale
        y, z = predict(model, val_loader, device, amp=use_amp)
        metrics = classification_metrics(y, z)
        key = (metrics["balanced_accuracy"],)
        if not warmup and settings.get("val_macro_f1_tiebreak", False):
            key += (metrics["macro_f1"],)
        if not warmup and settings.get("val_loss_tiebreak", False):
            selection_loss = metrics["cross_entropy"]
            if settings.get("val_loss_label_smoothing", False):
                selection_loss = float(nn.functional.cross_entropy(
                    torch.from_numpy(z).double(), torch.from_numpy(y).long(),
                    label_smoothing=float(settings.get("label_smoothing", 0.05)),
                ))
            key += (-selection_loss,)
        if best_key is None or key > best_key:
            best_key, best_position, stale = key, position, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            torch.save(dict(model_state_dict=best_state, model_config=model.config.to_dict(),
                            epoch=epoch, step=updates, stage=stage, val=metrics), checkpoint_path)
        else:
            stale += 1
        row = dict(stage=stage, epoch=epoch, step=updates, validation=metrics,
                   best_balanced_accuracy=best_key[0], seconds=time.perf_counter() - begin)
        history.append(row)
        write_json(directory / f"{stage}_history.json", history)
        print(json.dumps({"stage": stage, "epoch": epoch, "step": updates,
                          "val_balanced_accuracy": metrics["balanced_accuracy"],
                          "best": best_key[0]}), flush=True)
        model.train()

    model.train()
    while position < maximum:
        epoch += 1
        epoch_index = epoch - 1
        for batch in train_loader:
            index = updates if steps else epoch_index
            factor = learning_rate_factor(index, maximum, settings, warmup=warmup, steps=steps)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * factor
            x, y, pair = batch_to_device(batch, device, model.config.input_precision)
            x = augment(x, pair, settings)
            x, ya, yb, lam = aligned_mixup(x, y, pair, float(settings.get("mixup", settings.get("mixup_alpha", 0))))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = forward_batch(model, x, pair, batch, train_loader.dataset)
                loss = criterion(logits, ya) if lam == 1 else (
                    lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(settings.get("gradient_clip", settings.get("grad_clip", 1))))
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < old_scale:
                skipped += 1
                if skipped >= 100:
                    raise FloatingPointError("Repeated nonfinite gradients during mixed-precision training")
                continue
            skipped = 0
            updates += 1
            if steps:
                position = updates
                if checkpoint_rule == "best" and (position % interval == 0 or position == maximum):
                    validate()
                if position >= maximum or (patience and stale >= patience):
                    break
        if not steps:
            position = epoch
            if checkpoint_rule == "best":
                validate()
            else:
                history.append(dict(stage=stage, epoch=epoch, step=updates,
                                    seconds=time.perf_counter() - begin))
                write_json(directory / f"{stage}_history.json", history)
                print(json.dumps({"stage": stage, "epoch": epoch, "step": updates}), flush=True)
        if position >= minimum and patience and stale >= patience:
            break
    if checkpoint_rule == "last":
        checkpoint_path = directory / f"{stage}_last.pt"
        torch.save(dict(model_state_dict=model.state_dict(), model_config=model.config.to_dict(),
                        epoch=epoch, step=updates, stage=stage), checkpoint_path)
        return dict(stage=stage, checkpoint="last", selected_position=position,
                    position_unit="step" if steps else "epoch", optimizer_steps=updates,
                    epochs=epoch, seconds=time.perf_counter() - begin)
    if best_state is None:
        raise RuntimeError("Training completed without a validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    return dict(stage=stage, best_position=best_position, position_unit="step" if steps else "epoch",
                best_validation_balanced_accuracy=best_key[0], optimizer_steps=updates,
                epochs=epoch, seconds=time.perf_counter() - begin)
