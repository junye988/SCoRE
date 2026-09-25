"""EEG augmentation in physical coordinates and recipient-frame mixup."""

from __future__ import annotations

import numpy as np
import random
import torch


def augment(x, pair, settings):
    gain = float(settings.get("gain_log_std", 0))
    noise = float(settings.get("noise_rms_fraction", 0))
    drop = float(settings.get("channel_drop_probability", 0))
    if gain or noise or drop:
        with torch.autocast(x.device.type, enabled=False):
            physical = torch.bmm(pair[:, 1], x.double())
            rms = physical.square().mean(dim=-1, keepdim=True).sqrt()
            if gain:
                physical = physical * torch.exp(torch.randn((*x.shape[:2], 1), device=x.device,
                                                             dtype=torch.float64) * gain)
            if noise:
                physical = physical + torch.randn_like(physical) * (noise * rms)
            if drop:
                maximum = min(int(settings.get("max_dropped_channels", 1)), x.shape[1])
                if maximum < 1:
                    raise ValueError("Channel dropout requires max_dropped_channels >= 1")
                scores = torch.rand(x.shape[:2], device=x.device, dtype=torch.float64)
                cutoff = scores.topk(maximum, dim=1, largest=False).values[:, -1:]
                physical = physical.masked_fill(((scores < drop) & (scores <= cutoff))[:, :, None], 0)
            x = torch.bmm(pair[:, 0], physical)
    noise_std = float(settings.get("noise_std", 0))
    mask_probability = float(settings.get("time_mask_prob", 0))
    if noise_std or mask_probability:
        with torch.autocast(x.device.type, enabled=False):
            physical = pair[:, 1] @ x.double()
            if noise_std:
                physical = physical + noise_std * torch.randn_like(physical)
            if mask_probability and random.random() < mask_probability:
                width = max(1, int(x.shape[-1] * settings.get("time_mask_ratio", 0.05)))
                start = random.randint(0, x.shape[-1] - width)
                physical = physical.clone()
                physical[:, :, start:start + width] = 0
            x = pair[:, 0] @ physical
    return x


def aligned_mixup(x, labels, pair, alpha):
    if alpha <= 0:
        return x, labels, labels, 1.0
    weight = float(np.random.beta(alpha, alpha))
    order = torch.randperm(len(labels), device=labels.device)
    with torch.autocast(x.device.type, enabled=False):
        values = x.double()
        donor_physical = torch.bmm(pair[order, 1], values[order])
        donor_aligned = torch.bmm(pair[:, 0], donor_physical)
        mixed = weight * values + (1 - weight) * donor_aligned
    return mixed, labels, labels[order], weight
