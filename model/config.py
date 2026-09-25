"""Configuration of the SCoRE architecture and prediction ensemble."""

from dataclasses import asdict, dataclass, field
from typing import Any


BCI_IV_2A_PERMUTATION = (
    0, 5, 4, 3, 2, 1, 12, 11, 10, 9, 8, 7, 6, 17, 16, 15,
    14, 13, 20, 19, 18, 21,
)


@dataclass(frozen=True)
class EnsembleConfig:
    enabled: bool = False
    size: int = 1
    aggregation: str = "mean_logits"
    weights: tuple[float, ...] | None = None
    temperatures: tuple[float, ...] | None = None
    members: tuple[dict[str, Any], ...] = ()

    def __post_init__(self):
        modes = ("mean_logits", "mean_probabilities", "weighted_probabilities")
        if self.size < 1 or self.aggregation not in modes:
            raise ValueError("Invalid ensemble size or aggregation")
        if not self.enabled and self.size != 1:
            raise ValueError("An ensemble with multiple members must be enabled")
        if self.members:
            if len(self.members) != self.size:
                raise ValueError("Member specifications must match ensemble size")
            object.__setattr__(self, "members", tuple(dict(x) for x in self.members))
        if self.weights is not None:
            weights = tuple(float(x) for x in self.weights)
            if len(weights) != self.size or any(x < 0 for x in weights) or sum(weights) <= 0:
                raise ValueError("Ensemble weights must be nonnegative and match its size")
            if self.aggregation != "weighted_probabilities":
                raise ValueError("Explicit weights require weighted_probabilities aggregation")
            object.__setattr__(self, "weights", weights)
        if self.aggregation == "weighted_probabilities" and self.weights is None:
            raise ValueError("weighted_probabilities requires member weights")
        if self.temperatures is not None:
            temperatures = tuple(float(x) for x in self.temperatures)
            if len(temperatures) != self.size or any(x <= 0 for x in temperatures):
                raise ValueError("Temperatures must be positive and match ensemble size")
            if self.aggregation == "mean_logits":
                raise ValueError("Temperatures apply to probability aggregation")
            object.__setattr__(self, "temperatures", temperatures)


@dataclass(frozen=True)
class ModelConfig:
    num_channels: int = 22
    num_samples: int = 800
    num_classes: int = 4
    mirror_permutation: tuple[int, ...] = BCI_IV_2A_PERMUTATION
    temporal_filters: int = 40
    temporal_kernel: int = 25
    pool_kernel: int = 75
    pool_stride: int = 15
    energy_bins: int = 16
    embedding_width: int = 96
    dropout: float = 0.5
    input_precision: str = "float32"
    reflection_init: str = "physical"
    reflection_init_strength: float = 0.1
    reflection_init_seed: int = 1749
    initial_reflection_precision: str = "float32"
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)

    def __post_init__(self):
        integer_fields = (
            "num_channels", "num_samples", "num_classes", "temporal_filters",
            "temporal_kernel", "pool_kernel", "pool_stride", "energy_bins",
            "embedding_width",
        )
        if any(getattr(self, name) < 1 for name in integer_fields):
            raise ValueError("Model dimensions must be positive")
        if self.num_classes < 2 or self.temporal_kernel % 2 != 1:
            raise ValueError("Use at least two classes and an odd temporal kernel")
        if self.pool_kernel > self.num_samples or not 0 <= self.dropout < 1:
            raise ValueError("Invalid pooling width or dropout probability")
        if self.input_precision not in ("float32", "float64"):
            raise ValueError("input_precision must be float32 or float64")
        if self.initial_reflection_precision not in ("float32", "float64"):
            raise ValueError("initial_reflection_precision must be float32 or float64")
        permutation = tuple(self.mirror_permutation)
        if sorted(permutation) != list(range(self.num_channels)) or any(
            permutation[permutation[i]] != i for i in range(self.num_channels)
        ):
            raise ValueError("mirror_permutation must be a channel involution")
        if self.reflection_init not in ("physical", "cayley", "random-pairs", "haar"):
            raise ValueError("Unknown reflection initialization")
        if self.reflection_init_strength < 0:
            raise ValueError("reflection_init_strength must be nonnegative")
        object.__setattr__(self, "mirror_permutation", permutation)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelConfig":
        value = dict(value)
        if isinstance(value.get("ensemble"), dict):
            value["ensemble"] = EnsembleConfig(**value["ensemble"])
        return cls(**value)

    def backbone_kwargs(self) -> dict[str, Any]:
        names = (
            "num_channels", "num_samples", "temporal_filters", "temporal_kernel",
            "pool_kernel", "pool_stride", "energy_bins", "embedding_width", "dropout",
        )
        return {name: getattr(self, name) for name in names}
