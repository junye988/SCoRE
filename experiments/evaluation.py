"""Batched predictions and initialization diagnostics."""

import numpy as np
import torch

from .batches import batch_to_device, forward_batch


@torch.inference_mode()
def predict(model, loader, device, *, amp=False, diagnostics=False):
    model.eval()
    outputs = []
    labels = []
    precision = model.config.input_precision
    for batch in loader:
        x, y, pair = batch_to_device(batch, device, precision)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            result = forward_batch(model, x, pair, batch, loader.dataset, diagnostics=diagnostics)
        if diagnostics:
            result = result["odd_logit_difference"]
        outputs.append(result.detach().float().cpu().numpy())
        labels.append(y.cpu().numpy())
    return np.concatenate(labels), np.concatenate(outputs)
