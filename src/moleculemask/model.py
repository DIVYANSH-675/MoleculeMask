"""Rotation/translation-equivariant masked-coordinate learning."""

import numpy as np
import torch
from torch import nn


class GeometricLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * width + 1, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU()
        )
        self.direction = nn.Sequential(nn.Linear(width, 1), nn.Tanh())
        self.update = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, features, positions, masked):
        n = positions.shape[1]
        relative = positions[:, :, None, :] - positions[:, None, :, :]
        squared = relative.square().sum(-1, keepdim=True)
        left = features[:, :, None, :].expand(-1, -1, n, -1)
        right = features[:, None, :, :].expand(-1, n, -1, -1)
        edges = self.message(torch.cat([left, right, squared], dim=-1))
        off_diagonal = (1 - torch.eye(n, device=positions.device))[None, :, :, None]
        edges = edges * off_diagonal
        displacement = (
            self.direction(edges) * relative / (1 + squared.clamp_min(1e-8).sqrt()) * off_diagonal
        ).sum(2) / (n - 1)
        positions = positions + 0.1 * masked[..., None] * displacement
        features = features + self.update(torch.cat([features, edges.sum(2) / (n - 1)], dim=-1))
        return features, positions


class MaskedMolecule(nn.Module):
    def __init__(self, width=32, depth=3):
        super().__init__()
        self.atom = nn.Embedding(119, width)
        self.mask_token = nn.Embedding(2, width)
        self.layers = nn.ModuleList([GeometricLayer(width) for _ in range(depth)])
        self.property_head = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, positions, atomic_numbers, masked):
        if positions.ndim != 3 or positions.shape[-1] != 3 or masked.shape != positions.shape[:2]:
            raise ValueError("Expected BxNx3 positions and BxN mask")
        if atomic_numbers.shape != masked.shape or positions.shape[1] < 2:
            raise ValueError("Atomic-number shape mismatch or fewer than two atoms")
        if (
            not ((masked == 0) | (masked == 1)).all()
            or (masked.sum(1) < 1).any()
            or (masked.sum(1) >= positions.shape[1]).any()
        ):
            raise ValueError("Each molecule needs both visible and masked atoms")
        if (
            not len(positions)
            or not positions.is_floating_point()
            or not torch.isfinite(positions).all()
        ):
            raise ValueError("Positions must be a nonempty finite floating-point batch")
        if (
            atomic_numbers.dtype not in (torch.int32, torch.int64)
            or (atomic_numbers < 1).any()
            or (atomic_numbers > 118).any()
        ):
            raise ValueError("Atomic numbers must be integer tensors in [1,118]")
        masked = masked.to(dtype=positions.dtype)
        visible = (1 - masked)[..., None]
        center = (positions * visible).sum(1, keepdim=True) / visible.sum(1, keepdim=True)
        coordinates = positions * visible + center * masked[..., None]
        features = self.atom(atomic_numbers) + self.mask_token(masked.long())
        for layer in self.layers:
            features, coordinates = layer(features, coordinates, masked)
        return coordinates, self.property_head(features.mean(1)).squeeze(-1)


def synthetic(count=128, atoms=6, seed=0):
    rng = np.random.default_rng(seed)
    # Geometric point clouds, not chemically validated structures.
    positions = rng.normal(size=(count, atoms, 3)).astype(np.float32)
    numbers = rng.choice([1, 6, 7, 8], size=(count, atoms)).astype(np.int64)
    distance = np.linalg.norm(positions[:, :, None] - positions[:, None, :], axis=-1)
    properties = (numbers.sum(1) / 20 + np.exp(-distance).mean((1, 2))).astype(np.float32)
    return positions, numbers, properties


def load_data(path):
    with np.load(path, allow_pickle=False) as archive:
        p, z, y = archive["positions"], archive["atomic_numbers"], archive["properties"]
    if p.ndim != 3 or p.shape[-1] != 3 or p.shape[:2] != z.shape or y.shape != (len(p),):
        raise ValueError("Expected positions[B,N,3], atomic_numbers[B,N], properties[B]")
    if p.dtype.kind not in "fiu" or y.dtype.kind not in "fiu":
        raise ValueError("Positions and properties must be real numeric arrays")
    if len(p) < 5 or p.shape[1] < 2 or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Need at least five finite molecules with at least two atoms")
    if not np.issubdtype(z.dtype, np.integer) or z.min() < 1 or z.max() > 118:
        raise ValueError("Atomic numbers must be integers in [1,118]")
    with np.errstate(over="ignore"):
        positions, properties = p.astype(np.float32), y.astype(np.float32)
    if not np.isfinite(positions).all() or not np.isfinite(properties).all():
        raise ValueError("Positions and properties must be representable as finite float32 values")
    return positions, z.astype(np.int64), properties


def one_mask(batch, atoms, random=True):
    indices = torch.randint(atoms, (batch,)) if random else torch.arange(batch) % atoms
    mask = torch.zeros(batch, atoms)
    mask[torch.arange(batch), indices] = 1
    return mask
