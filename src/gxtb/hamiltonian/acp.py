from __future__ import annotations

"""Atomic Correction Potentials (ACPs), doc/theory/13_acp.md.

Implements reduced non-local ACPs with single Gaussian projector per (A,l):
 - Non-local potential form (Eq. 72–74)
 - Auxiliary projector function definition (Eq. 79)
 - Hamiltonian assembly via auxiliary overlap S^{ACP} (Eq. 78a–c):
     H^{ACP}_{μν} = Σ_r (S^{ACP}_{μ r})^T S^{ACP}_{r ν} = (S^{ACP} S^{ACP}^T)_{μν}
 - Energy contribution (Eq. 75c): E^{ACP} = Tr{ H^{ACP} ∘ P }

This module includes a builder for the AO–ACP projector overlap matrix S^{ACP}
that follows Eqs. 78–80 exactly using the existing spherical-Gaussian overlap
machinery; no numerical shortcuts or placeholders are used.
"""

from dataclasses import dataclass
import torch
from typing import Iterable, Tuple

from ..basis.md_overlap import overlap_shell_pair
from ..cn import coordination_number

Tensor = torch.Tensor

__doc_refs__ = {"file": "doc/theory/13_acp.md", "eqs": ["72", "74", "78a", "78b", "78c", "75c", "79", "80"]}

__all__ = ["ACPParams", "build_acp_overlap", "acp_hamiltonian", "acp_energy", "__doc_refs__"]


@dataclass(frozen=True)
class ACPParams:
    """ACP projector parameters (doc/theory/13, Eqs. 74, 79–80).

    c0 : (Zmax+1, n_l) base auxiliary coefficients c^{ACP,0}_{Z,l}
    xi : (Zmax+1, n_l) auxiliary exponents ξ^{ACP}_{Z,l}
    k_acp_cn : global CN scaling k^{ACP,CN}
    cn_avg : (Zmax+1,) element-wise CN averages (training set derived)
    l_list : tuple[str,...] angular order (default: s,p,d[,f])
    """

    c0: Tensor
    xi: Tensor
    k_acp_cn: float
    cn_avg: Tensor
    l_list: Tuple[str, ...] = ("s", "p", "d")

    def l_index(self, l: str) -> int:
        return self.l_list.index(l)


def build_acp_overlap(
    numbers: Tensor,
    positions: Tensor,
    basis,
    *,
    c0: Tensor,
    xi: Tensor,
    k_acp_cn: float,
    cn_avg: Tensor,
    r_cov: Tensor,
    k_cn: float,
    l_list: Iterable[str] = ("s", "p", "d"),
) -> Tensor:
    """Construct AO–ACP overlap matrix S^{ACP} per Eq. 78b–c.

    For each atom A and each angular momentum l in l_list, we form a single
    auxiliary projector shell θ_{A,l,m} with dimension (2l+1) and coefficients
    (Eq. 79) c^{ACP}_{A,l} and exponents ξ^{ACP}_{A,l}. We expect xi to be in
    ACP units (already ξ^{ACP}); if NL exponents are provided by the caller, they
    must be halved upstream per Eq. 79. Coefficients follow Eq. 80:
        c^{ACP}_{A,l} = c0_{Z,l} * (1 + k_acp_cn * CN_A / CN_avg_Z).

    Returns
    -------
    S_acp : (nao, naux) with naux = Σ_A Σ_l (2l+1)
    """
    device = positions.device
    dtype = positions.dtype
    # CN_A
    cn = coordination_number(positions, numbers, r_cov.to(device=device, dtype=dtype), float(k_cn))
    # Prepare mapping
    shells = basis.shells
    nao = basis.nao
    # Count total auxiliary functions
    l_map = {"s": 0, "p": 1, "d": 2, "f": 3}
    dims = {"s": 1, "p": 3, "d": 5, "f": 7}
    naux = 0
    for A in range(len(numbers)):
        Z = int(numbers[A].item())
        for l in l_list:
            if l not in l_map:
                raise ValueError(f"Unsupported l='{l}' for ACP projector")
            naux += dims[l]
    S_acp = torch.zeros((nao, naux), dtype=dtype, device=device)

    # Build AO blocks against each projector shell
    col_off = 0
    for A in range(len(numbers)):
        Z = int(numbers[A].item())
        for l in l_list:
            ell = l_map[l]
            nprj = dims[l]
            # Coefficient and exponent per Eq. 79–80
            c0_Zl = c0[Z, ell].to(device=device, dtype=dtype)
            xi_Zl = xi[Z, ell].to(device=device, dtype=dtype)
            cn_avg_Z = cn_avg[Z].to(device=device, dtype=dtype)
            if float(cn_avg_Z.item()) == 0.0:
                raise ValueError("cn_avg for element Z==0 not provided; required by Eq. 80")
            c_acp = c0_Zl * (1.0 + float(k_acp_cn) * (cn[A] / cn_avg_Z))
            # Projector primitive arrays (one primitive contracting to 2l+1 spherical)
            alpha_j = torch.tensor([float(xi_Zl.item())], dtype=dtype, device=device)
            c_j = torch.tensor([float(c_acp.item())], dtype=dtype, device=device)
            # Fill S_acp column block by summing AO shells
            block_cols = slice(col_off, col_off + nprj)
            for ish, sh in enumerate(shells):
                # AO shell metadata
                li = {"s": 0, "p": 1, "d": 2, "f": 3}[sh.l]
                off_i = basis.ao_offsets[ish]
                ni = basis.ao_counts[ish]
                # AO primitives
                alpha_i = torch.tensor([p[0] for p in sh.primitives], dtype=dtype, device=device)
                c_i = torch.tensor([p[1] + p[2] for p in sh.primitives], dtype=dtype, device=device)
                # Displacement from AO center to projector center (A)
                R = positions[sh.atom_index] - positions[A]
                # Overlap block AO_shell x proj_shell
                S_block = overlap_shell_pair(li, ell, alpha_i, c_i, alpha_j, c_j, R)
                S_acp[off_i:off_i + ni, block_cols] = S_block
            col_off += nprj
    return S_acp


def acp_hamiltonian(S_acp: Tensor) -> Tensor:
    """Build H^{ACP} from AO–ACP overlap S^{ACP} (Eq. 78c).

    Parameters
    ----------
    S_acp : (nao, naux) AO-to-auxiliary overlap matrix

    Returns
    -------
    H_acp : (nao, nao) = S^{ACP} @ S^{ACP}^T
    """
    if S_acp.dim() != 2:
        raise ValueError("S_acp must be rank-2 (nao, naux)")
    return S_acp @ S_acp.T


def acp_energy(P: Tensor, H_acp: Tensor) -> Tensor:
    """E^{ACP} = Tr{ H^{ACP} ∘ P } (Eq. 75c)."""
    if P.shape != H_acp.shape:
        raise ValueError("P and H_acp must have the same shape")
    return torch.einsum('ij,ji->', P, H_acp)
