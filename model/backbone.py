"""Temporal-spatial log-energy backbone."""

import torch
from torch import Tensor, nn


class ShallowLogEnergyBackbone(nn.Module):
    def __init__(self, *, num_channels, num_samples, temporal_filters=40,
                 temporal_kernel=25, pool_kernel=75, pool_stride=15,
                 energy_bins=16, embedding_width=96, dropout=0.5):
        super().__init__()
        self.num_channels = int(num_channels)
        self.num_samples = int(num_samples)
        self.temporal_filters = int(temporal_filters)
        self.energy_bins = int(energy_bins)
        self.embedding_width = int(embedding_width)
        self.temporal = nn.Conv2d(1, temporal_filters, (1, temporal_kernel),
                                  padding=(0, temporal_kernel // 2), bias=False)
        self.spatial = nn.Conv2d(temporal_filters, temporal_filters,
                                 (num_channels, 1), bias=False)
        self.norm = nn.BatchNorm2d(temporal_filters, momentum=0.1)
        self.energy_pool = nn.AvgPool2d((1, pool_kernel), (1, pool_stride))
        self.bin_pool = nn.AdaptiveAvgPool2d((1, energy_bins))
        self.feature_dropout = nn.Dropout(dropout)
        self.embedding = nn.Sequential(
            nn.Linear(temporal_filters * energy_bins, embedding_width),
            nn.ELU(), nn.Dropout(min(float(dropout), 0.3)),
        )
        self.classifier = nn.Linear(embedding_width, 4)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.temporal.weight, nonlinearity="linear")
        nn.init.xavier_uniform_(self.spatial.weight)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.xavier_uniform_(self.embedding[0].weight)
        nn.init.zeros_(self.embedding[0].bias)
        nn.init.xavier_uniform_(self.classifier.weight, gain=0.1)
        nn.init.zeros_(self.classifier.bias)

    def forward_features(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.num_channels, self.num_samples):
            raise ValueError(f"Expected [B,{self.num_channels},{self.num_samples}] EEG")
        x = x - x.mean(dim=1, keepdim=True)
        scale = x.square().mean(dim=(1, 2), keepdim=True).add(1.0e-6).sqrt()
        x = (x / scale).unsqueeze(1).to(self.temporal.weight.dtype)
        response = self.norm(self.spatial(self.temporal(x)))
        energy = self.energy_pool(response.square())
        log_energy = torch.log(energy.clamp_min(1.0e-6))
        binned = self.bin_pool(log_energy).flatten(1)
        return self.embedding(self.feature_dropout(binned))

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.forward_features(x))
