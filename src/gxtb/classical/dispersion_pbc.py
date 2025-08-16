from __future__ import annotations

"""Periodic revD4/D4 two-body lattice sum with explicit real-space cutoff.

Implements the two-body D4 energy per unit cell by summing contributions over
translations within a cutoff radius. Three-body ATM (s9 term) under PBC is not
implemented here; if the selected method has s9 != 0, an explicit exception is
raised (no hidden approximations).

Equations: doc/theory/22_dft_revd4.md (two-body part of Eq. 161) with BJ damping
per Eq. 170. Lattice assembly per doc/theory/25_periodic_boundary_conditions.md.
"""

from typing import Dict, List, Tuple

import torch

from .dispersion import D4Method, _trapzd_twobody, _d4_weight_references, _d4_ref_alpha
from ..pbc.cell import build_lattice_translations

Tensor = torch.Tensor

__all__ = ["d4_energy_pbc"]


def _halfspace_translations(trans: List[Tuple[int,int,int]]) -> List[Tuple[int,int,int]]:
    """Select unique half-space: keep origin and translations with (k>0) or (k==0 and j>0) or (k==0 and j==0 and i>0).

    This avoids double-counting home–image pairs when summing over R and -R.
    """
    out: List[Tuple[int,int,int]] = []
    for i,j,k in trans:
        if (i,j,k) == (0,0,0):
            out.append((i,j,k))
        elif (k > 0) or (k == 0 and j > 0) or (k == 0 and j == 0 and i > 0):
            out.append((i,j,k))
    return out


def d4_energy_pbc(
    numbers: Tensor,
    positions: Tensor,
    charges: Tensor,
    method: D4Method,
    ref: Dict,
    cell: Tensor,
    rcut: float,
) -> Tensor:
    """Two-body D4 lattice sum per unit cell with explicit real-space cutoff.

    - Requires ref['cn'] to be present and consistent with periodic CN.
    - Raises NotImplementedError if method.s9 != 0 (ATM not implemented here).
    - Returns energy in Hartree.
    """
    if method.s9 != 0.0:
        raise NotImplementedError("Periodic D4 ATM (s9!=0) not implemented; select a method with s9=0 for PBC D4.")
    device, dtype = positions.device, positions.dtype
    n = numbers.shape[0]
    # Build translations within cutoff and reduce to half-space (unique)
    allT = build_lattice_translations(float(rcut), cell)
    T = _halfspace_translations(allT)

    # Precompute reference α and per-atom weights (ζ·gw) for home cell
    # Enforce presence of CN in ref (no hidden defaults)
    if 'cn' not in ref:
        raise ValueError("ref['cn'] (periodic CN) required for D4 PBC (no hidden CN computation)")
    # Home-cell weights and α
    zeta, gw = _d4_weight_references(numbers, ref['cn'].to(device=device, dtype=dtype), charges.to(device=device, dtype=dtype), ref)
    alpha = _d4_ref_alpha(numbers, ref)
    rc6_home = _trapzd_twobody(alpha)  # (n,n,R,R) references
    W_home = zeta * gw                  # (n,R)
    C6_home = torch.einsum('ijab,ia,jb->ij', rc6_home, W_home, W_home)  # (n,n)
    r4r2 = ref['r4r2'][numbers.long()].to(dtype)
    # Pair radii R0 factors for home-home pairs (used for BJ damping); we'll reuse per pair
    R0_home = method.a1 * torch.sqrt(3.0 * r4r2.unsqueeze(1) * r4r2.unsqueeze(0)) + method.a2
    C8_home = C6_home * (3.0 * r4r2.unsqueeze(1) * r4r2.unsqueeze(0))

    E = torch.zeros((), dtype=dtype, device=device)

    # Origin contribution (home-home pairs, i<j)
    rij = positions.unsqueeze(0) - positions.unsqueeze(1)
    R = torch.linalg.norm(rij, dim=-1) + torch.eye(n, device=device, dtype=dtype) * torch.finfo(dtype).eps
    i = torch.arange(n, device=device)
    j = torch.arange(n, device=device)
    mask = (i[:, None] > j[None, :])
    f6 = 1.0 / (R**6 + R0_home**6)
    f8 = 1.0 / (R**8 + R0_home**8)
    E = E - (
        method.s6 * torch.where(mask, C6_home * f6, torch.zeros_like(R)) +
        method.s8 * torch.where(mask, C8_home * f8, torch.zeros_like(R))
    ).sum()

    # Cross terms with images: for each translation in the selected half-space
    for (ti, tj, tk) in T:
        if (ti, tj, tk) == (0, 0, 0):
            continue
        Rvec = ti * cell[0] + tj * cell[1] + tk * cell[2]
        pos_img = positions + Rvec
        # Build cross distances between (home,i) and (image,j)
        rij = positions.unsqueeze(1) - pos_img.unsqueeze(0)  # (n,n,3)
        R = torch.linalg.norm(rij, dim=-1) + torch.finfo(dtype).eps
        # For cross pairs, C6 and R0 are taken from home-home element tables (depends only on elements)
        # We reuse C6_home and C8_home; indices [i,j] map directly.
        f6 = 1.0 / (R**6 + R0_home**6)
        f8 = 1.0 / (R**8 + R0_home**8)
        E = E - (method.s6 * C6_home * f6 + method.s8 * C8_home * f8).sum()

    return E

