"""SCoRE with a learned input involution and a Householder output action."""

import numpy as np
import torch
from torch import Tensor, nn

from .backbone import ShallowLogEnergyBackbone
from .config import ModelConfig
from .initialization import (
    binary_axis, build_initial_reflection, canonicalize_axis_sign, contrast_unit_axis,
    spectral_axis_from_odd_logits,
)


def _build_backbone(config):
    backbone = ShallowLogEnergyBackbone(**config.backbone_kwargs())
    if config.num_classes != backbone.classifier.out_features:
        backbone.classifier = nn.Linear(config.embedding_width, config.num_classes)
        nn.init.xavier_uniform_(backbone.classifier.weight, gain=0.1)
        nn.init.zeros_(backbone.classifier.bias)
    return backbone


def _build_initial_reflection(config):
    matrix = build_initial_reflection(
        config.mirror_permutation, config.reflection_init,
        config.reflection_init_strength, config.reflection_init_seed,
    )
    return matrix.to(getattr(torch, config.initial_reflection_precision))


def _axis_from_classifier(classifier):
    with torch.no_grad():
        row_energy = classifier.weight.detach().float().square().mean(dim=1)
        candidate = row_energy - row_energy.mean()
        norm = torch.linalg.vector_norm(candidate)
        if not bool(torch.isfinite(norm)) or float(norm) <= 1.0e-12:
            candidate = torch.arange(classifier.out_features, dtype=torch.float32,
                                     device=classifier.weight.device)
            candidate = candidate - candidate.mean()
        return contrast_unit_axis(candidate).clone()


class SCoRE(nn.Module):
    """Map aligned EEG [B,C,T] and alignment pairs [B,2,C,C] to class logits."""

    def __init__(self, config: ModelConfig | dict, *, initial_axis=None):
        super().__init__()
        self.config = ModelConfig.from_dict(config) if isinstance(config, dict) else config
        config = self.config
        self.num_classes = config.num_classes
        self.backbone = _build_backbone(config)
        self.feature_dim = config.embedding_width
        axis = _axis_from_classifier(self.classifier)
        if config.num_classes == 2:
            axis = binary_axis()
        elif initial_axis is not None:
            axis = canonicalize_axis_sign(torch.as_tensor(initial_axis))
        self.axis_raw = nn.Parameter(axis, requires_grad=config.num_classes > 2)
        self.register_buffer("mirror_permutation", torch.tensor(config.mirror_permutation))
        self.register_buffer("reflection_upper_indices", torch.triu_indices(
            config.num_channels, config.num_channels, offset=1,
        ))
        self.register_buffer("initial_input_reflection", _build_initial_reflection(config))
        self.reflection_raw = nn.Parameter(torch.zeros(self.reflection_upper_indices.shape[1]))

    @property
    def classifier(self):
        return self.backbone.classifier

    @property
    def input_dtype(self):
        return getattr(torch, self.config.input_precision)

    def input_reflection_parameters(self):
        yield self.reflection_raw

    def output_reflection_parameters(self):
        if self.axis_raw.requires_grad:
            yield self.axis_raw

    def unit_axis(self):
        return contrast_unit_axis(self.axis_raw)

    def output_involution(self):
        q = self.unit_axis()
        return torch.eye(self.num_classes, device=q.device, dtype=q.dtype) - 2 * torch.outer(q, q)

    def input_reflection_matrix(self):
        with torch.autocast(device_type=self.reflection_raw.device.type, enabled=False):
            raw = self.reflection_raw.to(self.input_dtype)
            channels = self.config.num_channels
            upper = raw.new_zeros((channels, channels))
            upper[self.reflection_upper_indices[0], self.reflection_upper_indices[1]] = raw
            skew = upper - upper.T
            identity = torch.eye(channels, device=raw.device, dtype=raw.dtype)
            rotation = torch.linalg.solve(identity - skew, identity + skew)
            return rotation @ self.initial_input_reflection.to(raw.dtype) @ rotation.T

    def transported_reflection(self, alignment_pair):
        channels = self.config.num_channels
        if alignment_pair.ndim != 4 or tuple(alignment_pair.shape[1:]) != (2, channels, channels):
            raise ValueError("Expected alignment pairs [B,2,C,C]")
        if alignment_pair.device != self.reflection_raw.device:
            raise ValueError("Model and alignment pairs must be on the same device")
        if self.input_dtype == torch.float64 and alignment_pair.dtype != torch.float64:
            raise ValueError("float64 input transport requires float64 alignment pairs")
        with torch.autocast(device_type=alignment_pair.device.type, enabled=False):
            pairs = alignment_pair.to(self.input_dtype)
            return pairs[:, 0] @ self.input_reflection_matrix() @ pairs[:, 1]

    def reflected_input(self, x, alignment_pair):
        if alignment_pair.shape[0] != x.shape[0]:
            raise ValueError("Each trial requires one alignment pair")
        with torch.autocast(device_type=x.device.type, enabled=False):
            return torch.einsum("bij,bjt->bit", self.transported_reflection(alignment_pair),
                                x.to(self.input_dtype))

    def forward(self, x: Tensor, alignment_pair: Tensor, *, return_diagnostics=False):
        x = x.to(self.input_dtype)
        reflected = self.reflected_input(x, alignment_pair)
        direct_features, reflected_features = self.backbone.forward_features(
            torch.cat((x, reflected), dim=0),
        ).chunk(2, dim=0)
        direct_logits = self.classifier(direct_features)
        reflected_logits = self.classifier(reflected_features)
        q = self.unit_axis()
        odd = (reflected_logits * q).sum(dim=-1, keepdim=True) * q
        aligned_logits = (reflected_logits - odd) - odd
        logits = (0.5 * (direct_logits + aligned_logits)).float()
        if not return_diagnostics:
            return logits
        return dict(logits=logits, direct_logits=direct_logits,
                    reflected_logits=reflected_logits, aligned_reflected_logits=aligned_logits,
                    direct_features=direct_features, reflected_features=reflected_features)

    @torch.no_grad()
    def initialize_from_warmup(self, warmup, differences=None):
        """Transfer the warm-up backbone and initialize the output contrast."""
        if self.config.num_classes != warmup.config.num_classes:
            raise ValueError("Warm-up and joint model class counts differ")
        self.backbone.load_state_dict(warmup.backbone.state_dict(), strict=True)
        if self.num_classes == 2:
            axis = binary_axis()
        else:
            if differences is None:
                raise ValueError("Multiclass initialization requires training logit differences")
            axis = spectral_axis_from_odd_logits(
                differences, fallback_axis=_axis_from_classifier(warmup.classifier),
            )
        self.axis_raw.copy_(axis.to(self.axis_raw.device))
        return axis


class PairedDirectWarmupNet(nn.Module):
    """Direct-logit warm-up with joint processing of both input routes."""

    def __init__(self, config: ModelConfig | dict):
        super().__init__()
        self.config = ModelConfig.from_dict(config) if isinstance(config, dict) else config
        self.num_classes = self.config.num_classes
        self.backbone = _build_backbone(self.config)
        self.register_buffer("mirror_permutation", torch.tensor(self.config.mirror_permutation))
        self.register_buffer("initial_input_reflection", _build_initial_reflection(self.config),
                             persistent=False)

    @property
    def classifier(self):
        return self.backbone.classifier

    def transport_reflections(self, alignment_pairs: np.ndarray) -> Tensor:
        """Return CPU float32 actions [G,C,C] from float64 NumPy products."""
        pairs = np.asarray(alignment_pairs, dtype=np.float64)
        channels = self.config.num_channels
        if pairs.ndim != 4 or tuple(pairs.shape[1:]) != (2, channels, channels):
            raise ValueError("Expected alignment pairs [G,2,C,C]")
        initial = self.initial_input_reflection.detach().cpu().double().numpy()
        operators = np.stack([a @ initial @ inverse for a, inverse in pairs]).astype(np.float32)
        return torch.from_numpy(operators)

    def reflection_operator(self, alignment_pair):
        """Compute the fixed warm-up action from each group's alignment."""
        with torch.autocast(device_type=alignment_pair.device.type, enabled=False):
            pairs = alignment_pair.double()
            if self.config.input_precision == "float32":
                inverse = torch.linalg.inv(pairs[:, 0])
            else:
                inverse = pairs[:, 1]
            operator = pairs[:, 0] @ self.initial_input_reflection.double() @ inverse
            return operator.to(getattr(torch, self.config.input_precision))

    def forward(self, x, alignment_pair=None, *, reflection_operator=None,
                return_diagnostics=False):
        dtype = getattr(torch, self.config.input_precision)
        if reflection_operator is None:
            if alignment_pair is None:
                raise ValueError("Provide alignment pairs or a fixed reflection operator")
            if alignment_pair.ndim == 3:
                reflection_operator = alignment_pair
            else:
                reflection_operator = self.reflection_operator(alignment_pair)
        x = x.to(dtype)
        with torch.autocast(device_type=x.device.type, enabled=False):
            reflected = torch.einsum("bij,bjt->bit", reflection_operator.to(dtype), x)
        direct_features, reflected_features = self.backbone.forward_features(
            torch.cat((x, reflected), dim=0),
        ).chunk(2, dim=0)
        direct_logits = self.classifier(direct_features)
        reflected_logits = self.classifier(reflected_features)
        if not return_diagnostics:
            return direct_logits
        return dict(logits=direct_logits, direct_logits=direct_logits,
                    reflected_logits=reflected_logits,
                    odd_logit_difference=direct_logits - reflected_logits,
                    direct_features=direct_features, reflected_features=reflected_features)
