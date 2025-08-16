from __future__ import annotations

"""Periodic second-order (atomic-level) γ^{(2)} with Ewald long-range summation.

Implements (doc/theory/15, Eq. 100b–102) at the atomic level under PBC by
splitting the off-site kernel into a 1/R tail treated by Ewald and a rapidly
convergent residual summed in real space:

  1/sqrt(R^2 + c^2) = 1/R + [1/sqrt(R^2 + c^2) − 1/R],

where c = R_A^{cov} + R_B^{cov}. The residual decays as O(1/R^3) and is summed
within an explicit cutoff. The on-site term uses η_A (Eq. 102).
"""

from typing import Optional
import torch

from .ewald import ewald_sum_1over_r, realspace_translations

Tensor = torch.Tensor

__all__ = ["compute_gamma2_atomic_pbc"]


def compute_gamma2_atomic_pbc(
    numbers: Tensor,
    positions: Tensor,
    r_cov: Tensor,
    eta_A: Tensor,
    cell: Tensor,
    *,
    ewald_eta: float,
    r_cut: float,
    g_cut: float,
) -> Tensor:
    """Build atomic γ^{(2)} under PBC using Ewald + residual splitting.

    Off-diagonal (A≠B): γ_{AB} = EwaldSum(1/R_AB) + Σ_{R} [ 1/sqrt(|r+R|^2 + c^2) − 1/|r+R| ],
    Diagonal (A=A): γ_{AA} = η_A (Eq. 102 analogue at atomic level).
    """
    device = positions.device
    dtype = positions.dtype
    # Unit handling: atomic Coulomb kernel requires Bohr for distances (doc/theory/23_first_principles_constants.md).
    # Convert Angstrom inputs to Bohr before Ewald and residual sums; return γ in atomic units (Hartree).
    ANGSTROM_TO_BOHR = torch.tensor(1.8897261254535, dtype=dtype, device=device)
    nat = int(numbers.shape[0])
    z = numbers.long()
    # Distances and radii in Bohr
    rA = (r_cov[z].to(device=device, dtype=dtype)) * ANGSTROM_TO_BOHR
    cellB = cell.to(device=device, dtype=dtype) * ANGSTROM_TO_BOHR
    posB = positions.to(device=device, dtype=dtype) * ANGSTROM_TO_BOHR
    # Precompute translation vectors in Bohr for residual sum (excluding R=0 is handled in the loop)
    R_list = realspace_translations(cellB, float(r_cut) * float(ANGSTROM_TO_BOHR.item()))
    gamma = torch.zeros((nat, nat), dtype=dtype, device=device)
    # Pair loop (nat typically small for unit cells)
    for A in range(nat):
        for B in range(nat):
            if A == B:
                continue
            r = posB[B] - posB[A]
            # Ewald long-range 1/R sum in Bohr units: convert η (1/Å) and g_cut (1/Å) to 1/Bohr.
            eta_bohr = float(ewald_eta) / float(ANGSTROM_TO_BOHR.item())
            gcut_bohr = float(g_cut) / float(ANGSTROM_TO_BOHR.item())
            ewald_val = ewald_sum_1over_r(r, cellB, eta_bohr, float(r_cut) * float(ANGSTROM_TO_BOHR.item()), gcut_bohr)
            # Residual short-range correction
            csum = rA[A] + rA[B]
            resid = torch.tensor(0.0, dtype=dtype, device=device)
            eps = torch.finfo(dtype).eps
            for R in R_list:
                x = r + R
                d = torch.linalg.vector_norm(x)
                if d.item() < eps:
                    continue
                resid = resid + (1.0 / torch.sqrt(d * d + csum * csum) - 1.0 / d)
            gamma[A, B] = ewald_val + resid
    # Diagonal (on-site)
    gamma = gamma + torch.diag(eta_A[z].to(device=device, dtype=dtype))
    return gamma
