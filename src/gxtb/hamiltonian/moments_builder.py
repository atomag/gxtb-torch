from __future__ import annotations

"""Build AO moment matrices (S, D, Q) for a given basis and geometry.

Equations:
 - Monopole overlap S_{μν} (Eq. 111a)
 - Dipole moments D^α_{μν} (Eq. 111b)
 - Quadrupole moments Q^{αβ}_{μν} (Eq. 111c)

Maps per-shell pair moment sub-blocks (real spherical) into full AO matrices,
using the same spherical transforms as overlap to preserve consistency.
"""

from typing import Tuple, List, Optional
import torch
from ..basis.moments import moment_shell_pair

Tensor = torch.Tensor

__all__ = ["build_moment_matrices"]


def build_moment_matrices(
    numbers: Tensor,
    positions: Tensor,
    basis,
    *,
    coeff_override: Optional[List[Tensor]] = None,
) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]:
    """Assemble AO-level S, D, Q matrices for current geometry.

    Returns
    -------
    S : (nao,nao)
    D : tuple (Dx,Dy,Dz) each (nao,nao)
    Q : tuple (Qxx,Qxy,Qxz,Qyy,Qyz,Qzz) each (nao,nao)
    """
    nao = basis.nao
    dtype = positions.dtype
    device = positions.device
    # Initialize full matrices
    Dx = torch.zeros((nao, nao), dtype=dtype, device=device)
    Dy = torch.zeros_like(Dx)
    Dz = torch.zeros_like(Dx)
    Qxx = torch.zeros_like(Dx)
    Qxy = torch.zeros_like(Dx)
    Qxz = torch.zeros_like(Dx)
    Qyy = torch.zeros_like(Dx)
    Qyz = torch.zeros_like(Dx)
    Qzz = torch.zeros_like(Dx)
    # We also need S (monopole) consistent with overlap build; reuse basis overlaps route
    # Since moment_shell_pair does not return S, we build it by calling overlap implementation
    from ..basis.md_overlap import overlap_shell_pair
    S = torch.zeros((nao, nao), dtype=dtype, device=device)
    # Precompute per-shell primitive tensors once
    alpha_list = []
    coeff_list = []
    for sh in basis.shells:
        alpha_list.append(torch.tensor([p[0] for p in sh.primitives], dtype=dtype, device=device))
        if coeff_override is None:
            # Static q‑vSZP baseline: c = c0 (doc/theory/7 Eq. 27 with q_eff=0)
            coeff_list.append(torch.tensor([p[1] for p in sh.primitives], dtype=dtype, device=device))
        else:
            # Use dynamic contraction coefficients provided by caller (c0 + c1 q_eff)
            coeff_list.append(coeff_override[len(coeff_list)].to(device=device, dtype=dtype))
    # Iterate shells
    for i, shi in enumerate(basis.shells):
        oi, ni = basis.ao_offsets[i], basis.ao_counts[i]
        alpha_i = alpha_list[i]
        c_i = coeff_list[i]
        li = {"s": 0, "p": 1, "d": 2, "f": 3}.get(shi.l, 0)
        Ri = positions[shi.atom_index]
        for j, shj in enumerate(basis.shells):
            oj, nj = basis.ao_offsets[j], basis.ao_counts[j]
            alpha_j = alpha_list[j]
            c_j = coeff_list[j]
            lj = {"s": 0, "p": 1, "d": 2, "f": 3}.get(shj.l, 0)
            Rj = positions[shj.atom_index]
            R = Ri - Rj
            # Dipole/quadrupole blocks
            Dxi, Dyi, Dzi, Qxxi, Qxyi, Qxzi, Qyyi, Qyzi, Qzzi = moment_shell_pair(
                li, lj, alpha_i, c_i, alpha_j, c_j, R
            )
            Dx[oi : oi + ni, oj : oj + nj] = Dxi
            Dy[oi : oi + ni, oj : oj + nj] = Dyi
            Dz[oi : oi + ni, oj : oj + nj] = Dzi
            Qxx[oi : oi + ni, oj : oj + nj] = Qxxi
            Qxy[oi : oi + ni, oj : oj + nj] = Qxyi
            Qxz[oi : oi + ni, oj : oj + nj] = Qxzi
            Qyy[oi : oi + ni, oj : oj + nj] = Qyyi
            Qyz[oi : oi + ni, oj : oj + nj] = Qyzi
            Qzz[oi : oi + ni, oj : oj + nj] = Qzzi
            # Overlap block (monopole)
            Sij = overlap_shell_pair(li, lj, alpha_i, c_i, alpha_j, c_j, R)
            S[oi : oi + ni, oj : oj + nj] = Sij
    return S, (Dx, Dy, Dz), (Qxx, Qxy, Qxz, Qyy, Qyz, Qzz)
