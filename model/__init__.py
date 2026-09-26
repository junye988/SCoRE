"""SCoRE model and initialization interfaces."""

from .config import EnsembleConfig, ModelConfig
from .ensemble import SCoREEnsemble, aggregate_logits
from .initialization import spectral_axis_from_odd_logits
from .score import PairedDirectWarmupNet, SCoRE

WarmupModel = PairedDirectWarmupNet

__all__ = [
    "EnsembleConfig", "ModelConfig", "SCoRE", "PairedDirectWarmupNet",
    "SCoREEnsemble", "aggregate_logits", "spectral_axis_from_odd_logits", "WarmupModel",
]
