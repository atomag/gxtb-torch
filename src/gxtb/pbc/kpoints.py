from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch

__all__ = [
    "validate_kpoints",
    "monkhorst_pack",
]


def validate_kpoints(kpoints: Sequence[Sequence[float]], weights: Sequence[float]) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and normalize explicit k-point list and weights.

    - kpoints: list of fractional triplets [[kx,ky,kz], ...], each ∈ [0,1).
    - weights: same length; must sum to 1 within 1e-12.
    Returns (K, W) as tensors of shapes (nk,3) and (nk,).
    """
    K = torch.as_tensor(kpoints, dtype=torch.get_default_dtype())
    W = torch.as_tensor(weights, dtype=torch.get_default_dtype())
    if K.ndim != 2 or K.shape[1] != 3:
        raise ValueError(f"kpoints must have shape (nk,3), got {tuple(K.shape)}")
    if W.ndim != 1 or W.shape[0] != K.shape[0]:
        raise ValueError("kpoint weights must be a vector matching the number of kpoints")
    s = float(W.sum().item())
    if abs(s - 1.0) > 1e-12:
        raise ValueError("Sum of k-point weights must equal 1.0 exactly (within 1e-12)")
    # Enforce fractional domain [0,1) modulo 1
    K = K - torch.floor(K)
    return K, W


def monkhorst_pack(n1: int, n2: int, n3: int, s1: float, s2: float, s3: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct a shifted Monkhorst–Pack grid with explicit shift.

    No defaults (doc/theory/25_periodic_boundary_conditions.md): all inputs are required.
    Returns (K, W) with uniform weights summing to 1.0.
    """
    if not (n1 > 0 and n2 > 0 and n3 > 0):
        raise ValueError("Grid sizes must be positive integers")
    import itertools as it
    grid: List[Tuple[float, float, float]] = []
    for i, j, k in it.product(range(n1), range(n2), range(n3)):
        grid.append(((i + s1) / n1, (j + s2) / n2, (k + s3) / n3))
    K = torch.tensor(grid, dtype=torch.get_default_dtype())
    W = torch.full((K.shape[0],), 1.0 / K.shape[0], dtype=torch.get_default_dtype())
    return K, W

