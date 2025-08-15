from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import torch

__all__ = [
    "validate_cell",
    "to_cart",
    "to_frac",
    "build_lattice_translations",
]


def validate_cell(cell: torch.Tensor | Sequence[Sequence[float]] | None, pbc: Sequence[bool] | None) -> torch.Tensor:
    """Validate a 3x3 lattice matrix in Angstrom and PBC flags.

    Returns the cell as a (3,3) torch tensor (dtype from default).

    Rules (doc/theory/25_periodic_boundary_conditions.md):
    - 3D PBC only at this stage: pbc must be (True, True, True).
    - Cell vectors must form a non-singular matrix with positive volume.
    """
    if cell is None:
        raise ValueError("PBC cell is required (3x3 matrix in Angstrom).")
    A = torch.as_tensor(cell, dtype=torch.get_default_dtype())
    if A.shape != (3, 3):
        raise ValueError(f"cell must be shape (3,3), got {tuple(A.shape)}")
    if pbc is None or tuple(bool(x) for x in pbc) != (True, True, True):
        raise ValueError("Only fully periodic 3D systems are supported initially (pbc must be (True,True,True)).")
    vol = torch.det(A)
    if not torch.isfinite(vol) or float(vol.item()) <= 0.0:
        raise ValueError("Cell matrix must have positive determinant (right-handed basis).")
    return A


def to_cart(frac: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """Convert fractional to Cartesian coordinates via r = A f."""
    return frac @ cell.T


def to_frac(cart: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """Convert Cartesian to fractional coordinates via f = A^{-1} r."""
    return cart @ torch.linalg.inv(cell).T


def build_lattice_translations(cutoff: float, cell: torch.Tensor) -> List[Tuple[int, int, int]]:
    """Enumerate lattice translations R = n1 a1 + n2 a2 + n3 a3 within |R| <= cutoff.

    - Returns integer triplets sorted by increasing |R|, then lexicographic (n1,n2,n3).
    - Includes the origin (0,0,0).
    - Deterministic ordering ensures device parity (doc/theory/25_periodic_boundary_conditions.md).
    """
    if cutoff <= 0:
        raise ValueError("cutoff must be > 0")
    A = cell
    # Conservative bounds from column norms
    a1, a2, a3 = A[0], A[1], A[2]
    # Use lengths of primitive vectors to bound index ranges
    L1 = float(torch.linalg.vector_norm(a1))
    L2 = float(torch.linalg.vector_norm(a2))
    L3 = float(torch.linalg.vector_norm(a3))
    # Guard against degenerate lengths
    eps = 1e-15
    n1 = max(0, math.ceil(cutoff / max(L1, eps)))
    n2 = max(0, math.ceil(cutoff / max(L2, eps)))
    n3 = max(0, math.ceil(cutoff / max(L3, eps)))
    cand: List[Tuple[int, int, int]] = []
    for i in range(-n1, n1 + 1):
        for j in range(-n2, n2 + 1):
            for k in range(-n3, n3 + 1):
                R = i * a1 + j * a2 + k * a3
                if float(torch.linalg.vector_norm(R)) <= cutoff + 1e-12:
                    cand.append((i, j, k))
    # Sort by |R| then lexicographic
    def key(t: Tuple[int, int, int]):
        i, j, k = t
        R = i * a1 + j * a2 + k * a3
        return (float(torch.linalg.vector_norm(R)), i, j, k)
    cand.sort(key=key)
    return cand

