from __future__ import annotations

from typing import List, Tuple

import torch

from .cell import to_frac, to_cart

__all__ = ["coordination_number_pbc"]


def coordination_number_pbc(
    positions: torch.Tensor,
    numbers: torch.Tensor,
    r_cov: torch.Tensor,
    k_cn: float,
    cell: torch.Tensor,
    cutoff: float,
) -> torch.Tensor:
    """Periodic CN using minimum-image distances (doc/theory/9_cn.md Eq. 47; PBC handling per doc/theory/25).

    - Eq. 47: CN_A = Σ_{B≠A} 0.5 [1 + erf(k_cn (R_AB − R_cov,AB) / R_cov,AB)].
    - PBC: use minimum-image displacement via fractional wrapping Δf -> Δf − round(Δf), then Δr = A Δf.
      This counts each pair once (no lattice-image double counting).
    - cutoff argument is required by API but not used in minimum-image evaluation; validated to be > 0.

    Inputs
    - positions: (nat,3) Cartesian Angstrom
    - numbers: (nat,) atomic numbers Z
    - r_cov: (Zmax+1,) covalent radii (same units as positions)
    - k_cn: scalar CN sharpness parameter (positive)
    - cell: (3,3) lattice matrix (Angstrom)
    - cutoff: positive float (validated; not used)

    Returns
    - CN: (nat,) tensor
    """
    device = positions.device
    dtype = positions.dtype
    nat = int(positions.shape[0])
    if cutoff is None or float(cutoff) <= 0.0:
        raise ValueError("coordination_number_pbc requires cutoff > 0 (validated by API); minimum-image path does not use it")
    # Per-atom radii and pair-averaged radii
    z = numbers.long()
    rc = r_cov[z].to(device=device, dtype=dtype)  # (nat,)
    rc_ab = 0.5 * (rc.unsqueeze(1) + rc.unsqueeze(0))  # (nat,nat)
    # Fractional coordinates and minimum-image fractional differences
    f = to_frac(positions, cell.to(device=device, dtype=dtype))  # (nat,3)
    df = f.unsqueeze(1) - f.unsqueeze(0)  # (nat,nat,3)
    df_mi = df - torch.round(df)  # wrap to [-0.5,0.5) per component
    # Convert back to Cartesian displacements
    df_flat = df_mi.view(-1, 3)
    dr_flat = to_cart(df_flat, cell.to(device=device, dtype=dtype))  # (nat*nat,3)
    dr = dr_flat.view(nat, nat, 3)
    dist = torch.linalg.norm(dr, dim=-1)  # (nat,nat)
    # Exclude self terms
    mask = ~torch.eye(nat, dtype=torch.bool, device=device)
    eps = torch.finfo(dtype).eps
    x = float(k_cn) * (dist - rc_ab) / torch.clamp(rc_ab, min=eps)
    contrib = 0.5 * (1.0 + torch.erf(x))
    cn = (contrib * mask).sum(dim=1)
    return cn
