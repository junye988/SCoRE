"""Input-reflection initialization and spectral output-contrast initialization."""

import math

import torch
from torch import Tensor


def contrast_unit_axis(axis: Tensor, eps: float = 1.0e-12) -> Tensor:
    value = axis.double()
    centered = value - value.mean()
    norm = torch.linalg.vector_norm(centered)
    if not bool(torch.isfinite(norm.detach())) or float(norm.detach()) <= eps:
        raise ValueError("The output contrast must have a finite nonzero norm")
    return (centered / norm).float()


def canonicalize_axis_sign(axis: Tensor) -> Tensor:
    q = contrast_unit_axis(axis)
    pivot = int(torch.argmax(q.abs()).detach())
    return q * torch.where(q[pivot] < 0, q.new_tensor(-1), q.new_tensor(1))


def binary_axis() -> Tensor:
    return torch.tensor([1.0, -1.0]) / math.sqrt(2.0)


def spectral_axis_from_odd_logits(differences: Tensor, *, fallback_axis=None) -> Tensor:
    """Estimate the leading axis of centered training-route logit differences."""
    if differences.ndim != 2 or differences.shape[1] < 2:
        raise ValueError("Expected logit differences [N,K], K >= 2")
    if differences.shape[1] == 2:
        return binary_axis()
    if differences.shape[0] < 2 or not torch.isfinite(differences).all():
        raise ValueError("At least two finite training logit differences are required")
    values = differences.detach().to(device="cpu", dtype=torch.float64)
    values = values - values.mean(dim=1, keepdim=True)
    values = values - values.mean(dim=0, keepdim=True)
    covariance = values.T @ values / (len(values) - 1)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    if eigenvalues[-1] <= 1.0e-12:
        if fallback_axis is None:
            raise ValueError("Degenerate training covariance requires a fallback axis")
        return canonicalize_axis_sign(fallback_axis.detach().cpu())
    return canonicalize_axis_sign(eigenvectors[:, -1])


def build_initial_reflection(permutation, mode="physical", strength=0.1, seed=1749):
    """Construct an involution using an independent random generator."""
    permutation = torch.as_tensor(permutation, dtype=torch.long, device="cpu")
    channels = len(permutation)
    physical = torch.eye(channels, dtype=torch.float64)[permutation]
    if mode == "physical" or (mode == "cayley" and strength == 0):
        return physical
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if mode == "random-pairs":
        pair_count = int((permutation != torch.arange(channels)).sum()) // 2
        ordering = torch.randperm(channels, generator=generator)
        mapping = torch.arange(channels)
        for index in range(pair_count):
            left, right = ordering[2 * index:2 * index + 2]
            mapping[left], mapping[right] = right, left
        return torch.eye(channels, dtype=torch.float64)[mapping]
    basis = torch.zeros(channels, channels - 1, dtype=torch.float64)
    for column in range(channels - 1):
        count = column + 1
        scale = math.sqrt(count * (count + 1))
        basis[:count, column] = 1 / scale
        basis[count, column] = -count / scale
    sample = torch.randn(channels - 1, channels - 1, generator=generator,
                         dtype=torch.float64)
    identity = torch.eye(channels, dtype=torch.float64)
    if mode == "cayley":
        skew = basis @ (sample - sample.T) @ basis.T
        spectral_norm = torch.linalg.matrix_norm(skew, ord=2)
        if spectral_norm == 0:
            return physical
        skew *= strength / spectral_norm
        rotation = torch.linalg.solve(identity - skew, identity + skew)
    elif mode == "haar":
        orthogonal, triangular = torch.linalg.qr(sample)
        signs = torch.where(triangular.diagonal() < 0, -1., 1.)
        orthogonal = orthogonal * signs
        rotation = basis @ orthogonal @ basis.T + torch.ones_like(identity) / channels
    else:
        raise ValueError("Unknown reflection initialization")
    return rotation @ physical @ rotation.T
