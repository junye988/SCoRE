"""Prediction aggregation for independently trained SCoRE members."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .config import EnsembleConfig


def aggregate_logits(logits: Tensor, config: EnsembleConfig) -> Tensor:
    """Combine [members,batch,classes] predictions using the configured rule."""
    if logits.ndim != 3 or logits.shape[0] != config.size:
        raise ValueError("Expected member logits [M,B,K] matching ensemble size")
    if config.aggregation == "mean_logits":
        return logits.double().mean(dim=0)
    values = logits.double()
    temperatures = config.temperatures or (1.0,) * config.size
    temperatures = values.new_tensor(temperatures).reshape(-1, 1, 1)
    probabilities = (values / temperatures).softmax(dim=-1)
    weights = config.weights or (1 / config.size,) * config.size
    weights = values.new_tensor(weights).reshape(-1, 1, 1)
    weights = weights / weights.sum()
    probabilities = (weights * probabilities).sum(dim=0)
    return probabilities.clamp_min(1.0e-12).log()


class SCoREEnsemble(nn.Module):
    def __init__(self, members: Sequence[nn.Module], config: EnsembleConfig | dict | None = None):
        super().__init__()
        if not members:
            raise ValueError("Provide at least one trained member")
        self.members = nn.ModuleList(members)
        self.config = config or EnsembleConfig(enabled=len(members) > 1, size=len(members))
        if isinstance(self.config, dict):
            self.config = EnsembleConfig(**self.config)
        if len(members) != self.config.size:
            raise ValueError("Model count differs from the ensemble configuration")

    def forward(self, x, alignment_pair, *, return_diagnostics=False):
        member_logits = torch.stack([member(x, alignment_pair) for member in self.members])
        logits = aggregate_logits(member_logits, self.config)
        if return_diagnostics:
            return {"logits": logits, "member_logits": member_logits}
        return logits
