"""Prepare batch tensors and call the model with matching alignment operators."""

from weakref import WeakKeyDictionary

import numpy as np
import torch

_WARMUP_OPERATORS = WeakKeyDictionary()


def batch_to_device(batch, device, precision):
    dtype = torch.float64 if precision == "float64" else torch.float32
    return (batch["sample"].to(device, dtype=dtype, non_blocking=True),
            batch["label"].to(device, dtype=torch.long, non_blocking=True),
            batch["alignment_pair"].to(device, dtype=torch.float64, non_blocking=True))


def forward_batch(model, x, pair, batch, dataset, *, diagnostics=False):
    """Cache transported warm-up operators and select the current groups."""
    if model.config.input_precision == "float32" and hasattr(model, "transport_reflections"):
        cache = _WARMUP_OPERATORS.setdefault(model, {})
        key = (str(dataset.root), dataset.split, str(x.device))
        if key not in cache:
            cache[key] = model.transport_reflections(dataset.pairs).to(x.device)
        groups = np.asarray(dataset.groups)[batch["index"].cpu().numpy()]
        operator = cache[key][torch.as_tensor(groups, device=x.device, dtype=torch.long)]
        return model(x, reflection_operator=operator, return_diagnostics=diagnostics)
    return model(x, pair, return_diagnostics=diagnostics)
