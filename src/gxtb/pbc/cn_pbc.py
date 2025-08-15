from __future__ import annotations

from typing import List, Tuple

import torch

from .cell import build_lattice_translations

__all__ = ["coordination_number_pbc"]


def coordination_number_pbc(
    positions: torch.Tensor,
    numbers: torch.Tensor,
    r_cov: torch.Tensor,
    k_cn: float,
    cell: torch.Tensor,
    cutoff: float,
) -> torch.Tensor:
    """Periodic CN per doc/theory/9_cn.md Eq. (47) summed over lattice images within cutoff (doc/theory/25).

    positions: (nat,3) Cartesian Angstrom
    numbers: (nat,) Z
    r_cov: (Zmax+1,) covalent radii (same units as positions)
    cutoff: real-space sum radius (Angstrom)
    Returns CN: (nat,)
    """
    device = positions.device
    dtype = positions.dtype
    nat = positions.shape[0]
    z = numbers.long()

    # Precompute element radii per atom and pair averages
    rc = r_cov[z].to(device=device, dtype=dtype)  # (nat,)
    rc_ab = 0.5 * (rc.unsqueeze(1) + rc.unsqueeze(0))  # (nat,nat)

    # Lattice translations within cutoff (include origin)
    T = build_lattice_translations(float(cutoff), cell)
    cn = torch.zeros(nat, dtype=dtype, device=device)
    eps = torch.finfo(dtype).eps
    # Sum contributions over images
    for (i, j, k) in T:
        R = i * cell[0] + j * cell[1] + k * cell[2]
        pos_B = positions + R
        rij = positions.unsqueeze(1) - pos_B.unsqueeze(0)  # (nat,nat,3)
        dist = torch.linalg.norm(rij, dim=-1)  # (nat,nat)
        # Exclude self terms for the home cell only
        if i == 0 and j == 0 and k == 0:
            mask = ~torch.eye(nat, dtype=torch.bool, device=device)
        else:
            mask = torch.ones((nat, nat), dtype=torch.bool, device=device)
        x = k_cn * (dist - rc_ab) / torch.clamp(rc_ab, min=eps)
        contrib = 0.5 * (1.0 + torch.erf(x))
        cn = cn + (contrib * mask).sum(dim=1)
    return cn

