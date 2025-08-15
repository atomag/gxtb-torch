from __future__ import annotations

"""Third-order shell-resolved tight-binding (doc/theory/18_third_order_tb.md).

Energy (Eq. 129b):
  E^(3) = (1/6) Σ_{A,B} Σ_{l_A,l_B} q_{l_A} q_{l_B} ( q_A τ^{(3)}_{l_A l_B} Γ_{l_A} + q_B τ^{(3)}_{l_A l_B} Γ_{l_B} )
with τ^{(3)} derived from γ^{(3)} (Eqs. 132–133) and Γ_{l_A} = k^{(3),Γ}_{l_A} Γ^{(3)}_A (Eq. 131).

This module implements the energy using provided parameters and per-shell U^{(2)} (second-order Hubbard) values.
No hidden defaults: All parameters must be provided by the caller.
"""

from dataclasses import dataclass
from typing import Tuple
import torch

Tensor = torch.Tensor

__doc_refs__ = {"file": "doc/theory/18_third_order_tb.md", "eqs": ["129b", "132", "133b", "131"]}

__all__ = [
    "ThirdOrderParams",
    "third_order_energy",
    "compute_tau3_matrix",
    "build_third_order_potentials",
    "add_third_order_fock",
    "__doc_refs__",
]


@dataclass(frozen=True)
class ThirdOrderParams:
    # Element-wise Γ^{(3)}_A (Eq. 131)
    gamma3_elem: Tensor  # shape (Zmax+1,)
    # Global per-shell scalings k^{(3),Γ}_{l} for l in order ('s','p','d','f') (Eq. 131)
    kGamma_l: Tuple[float, float, float, float]
    # Interaction kernel parameters (Eq. 132–133)
    k3: float
    k3x: float
    l_list: Tuple[str, ...] = ("s", "p", "d", "f")

    def l_index(self, l: str) -> int:
        return self.l_list.index(l)


def _tau3_offsite(UA: Tensor, UB: Tensor, R: Tensor, k3: float, k3x: float) -> Tensor:
    """τ^{(3)} for A≠B per Eq. 133b as derivative ∂γ^{(3)}/∂U_A of Eq. 132.

    γ^{(3)}_{l_A l_B}(A≠B) = k3 * ((U_A+U_B)/2)^2 * R * exp( -k3x * ((U_A+U_B)/2) * R )
    ∂γ/∂U_A = k3 * ( (U_A+U_B)/2 ) * R * [ 1 - 0.5 * k3x * (U_A+U_B) * R ] * exp( -k3x * ((U_A+U_B)/2) * R )
    Derivation: product rule; algebra arranged to match 133b structure.
    """
    Uavg = 0.5 * (UA + UB)
    expf = torch.exp(-k3x * Uavg * R)
    return k3 * Uavg * R * (1.0 - k3x * Uavg * R) * expf


def _tau3_onsite(UA: Tensor, UB: Tensor, eps: float = 1e-14) -> Tensor:
    """τ^{(3)} onsite A=B derivative from γ^{(3)}_{onsite} = 1 / (0.5*(1/UA + 1/UB)).

    Exact expression (doc/theory/18, Eq. 133b onsite limit):
        γ = 2 UA UB / (UA + UB)
        ∂γ/∂UA = 2 UB^2 / (UA + UB)^2
    For UA,UB → 0 the limit is 0. We enforce this analytically by returning 0 when
    the denominator is below eps to avoid 0/0 → NaN.
    """
    den = (UA + UB).pow(2)
    out = torch.zeros_like(den)
    mask = den > eps
    out[mask] = 2.0 * (UB[mask] * UB[mask]) / den[mask]
    # else remains 0 (limiting value as UA,UB→0)
    return out


def third_order_energy(
    numbers: Tensor,
    positions: Tensor,
    basis,
    q_shell: Tensor,
    q_atom: Tensor,
    U_shell: Tensor,
    params: ThirdOrderParams,
) -> Tensor:
    """Compute E^(3) per Eq. 129b given required inputs.

    numbers: (nat,)
    positions: (nat,3)
    basis: AtomBasis-like with shells[], ao_offsets[], ao_counts[] (only shell meta used)
    q_shell: (n_shell,) shell charges q_{l_A}
    q_atom: (nat,) atomic charges q_A
    U_shell: (n_shell,) second-order shell Hubbard U^{(2)}_{l_A}
    params: ThirdOrderParams
    """
    device = positions.device
    dtype = positions.dtype
    shells = basis.shells
    n_sh = len(shells)
    # Gather shell meta
    atom_idx = torch.tensor([sh.atom_index for sh in shells], dtype=torch.long, device=device)
    l_idx = torch.tensor([params.l_index(sh.l) for sh in shells], dtype=torch.long, device=device)
    z_at = numbers.to(device=device, dtype=torch.long)
    # Γ_{l_A} = kGamma_l[l] * Γ^{(3)}_{A}
    gamma3_A = params.gamma3_elem[z_at]  # (nat,)
    kGamma = torch.tensor(params.kGamma_l, dtype=dtype, device=device)
    Gamma_lA = kGamma[l_idx] * gamma3_A[atom_idx]
    # Pairwise shell indices
    idx_i = torch.arange(n_sh, device=device)
    idx_j = torch.arange(n_sh, device=device)
    ii, jj = torch.meshgrid(idx_i, idx_j, indexing='ij')
    # Distances per shell pair from parent atoms
    rij = positions[atom_idx[jj]] - positions[atom_idx[ii]]
    R = torch.linalg.norm(rij, dim=-1)
    same_atom = (atom_idx[ii] == atom_idx[jj])
    UA = U_shell[ii]
    UB = U_shell[jj]
    tau = torch.where(
        same_atom,
        _tau3_onsite(UA, UB),
        _tau3_offsite(UA, UB, R, params.k3, params.k3x)
    )
    # Energy sum (Eq. 129b) using broadcasting
    ql_i = q_shell[ii]
    ql_j = q_shell[jj]
    qA = q_atom[atom_idx[ii]]
    qB = q_atom[atom_idx[jj]]
    Gamma_i = Gamma_lA[ii]
    Gamma_j = Gamma_lA[jj]
    term = ql_i * ql_j * (qA * tau * Gamma_i + qB * tau * Gamma_j)
    # Only unique pairs? Eq.129b sums A,B; shell pair loop includes all; divide by 6 as in equation
    return term.sum() / 6.0


def compute_tau3_matrix(numbers: Tensor, positions: Tensor, basis, U_shell: Tensor, params: ThirdOrderParams) -> Tensor:
    """Compute τ^{(3)}_{ij} for all shell pairs using Eqs. 132–133.

    Parameters
    ----------
    numbers : (nat,)
    positions : (nat,3)
    basis : AtomBasis-like
    U_shell : (n_shell,) per-shell U^{(2)} values
    params : ThirdOrderParams
    Returns
    -------
    tau : (n_shell, n_shell) tensor
    """
    device = positions.device
    dtype = positions.dtype
    shells = basis.shells
    atom_idx = torch.tensor([sh.atom_index for sh in shells], dtype=torch.long, device=device)
    # distances per shell pair via parent atoms
    rij = positions[atom_idx.unsqueeze(1)] - positions[atom_idx.unsqueeze(0)]
    R = torch.linalg.norm(rij, dim=-1)
    n_sh = len(shells)
    ii, jj = torch.meshgrid(torch.arange(n_sh, device=device), torch.arange(n_sh, device=device), indexing='ij')
    same_atom = (atom_idx[ii] == atom_idx[jj])
    UA = U_shell[ii]
    UB = U_shell[jj]
    tau = torch.where(
        same_atom,
        _tau3_onsite(UA, UB),
        _tau3_offsite(UA, UB, R, float(params.k3), float(params.k3x))
    )
    return tau


def build_third_order_potentials(
    numbers: Tensor,
    basis,
    q_shell: Tensor,
    q_atom: Tensor,
    tau: Tensor,
    params: ThirdOrderParams,
) -> tuple[Tensor, Tensor]:
    """Build shell and atomic third-order potentials for Fock update (cf. Eq. 136).

    V_shell[i] = (1/6) Σ_j q_j [ q_A(i) τ_{ij} Γ_i + q_A(j) τ_{ij} Γ_j ]
    V_atom[A]  = (1/6) Σ_{i∈A} Σ_j q_i q_j τ_{ij} Γ_i

    Returns
    -------
    V_shell : (n_shell,)
    V_atom : (nat,)
    """
    device = q_shell.device
    dtype = q_shell.dtype
    shells = basis.shells
    n_sh = len(shells)
    nat = len(numbers)
    # shell meta
    atom_idx = torch.tensor([sh.atom_index for sh in shells], dtype=torch.long, device=device)
    l_idx = torch.tensor([params.l_index(sh.l) for sh in shells], dtype=torch.long, device=device)
    z_at = numbers.to(device=device, dtype=torch.long)
    gamma3_A = params.gamma3_elem[z_at]
    kGamma = torch.tensor(params.kGamma_l, dtype=dtype, device=device)
    Gamma_lA = kGamma[l_idx] * gamma3_A[atom_idx]
    # Broadcast vectors
    qi = q_shell.unsqueeze(1)  # (n,1)
    qj = q_shell.unsqueeze(0)  # (1,n)
    qA_i = q_atom[atom_idx].unsqueeze(1)
    qA_j = q_atom[atom_idx].unsqueeze(0)
    Gi = Gamma_lA.unsqueeze(1)
    Gj = Gamma_lA.unsqueeze(0)
    # V_shell
    V_shell = (qj * (qA_i * tau * Gi + qA_j * tau * Gj)).sum(dim=1) / 6.0
    # V_atom by summing over shells belonging to each atom of i index
    contrib_i = (qi * qj * tau * Gi).sum(dim=1) / 6.0  # (n_shell,)
    V_atom = torch.zeros(nat, dtype=dtype, device=device)
    V_atom.index_add_(0, atom_idx, contrib_i)
    return V_shell, V_atom


def add_third_order_fock(H: Tensor, S: Tensor, basis, V_shell: Tensor, V_atom: Tensor) -> None:
    """Add third-order Fock contribution per Eq. 136 simplified form.

    F^{(3)}_{μν} = 1/2 S_{μν} ( V^{(3)}_{l(μ)} + V^{(3)}_{l(ν)} ) + 1/2 S_{μν} ( V^{(3)}_{A(μ)} + V^{(3)}_{A(ν)} )
    """
    device = H.device
    # Map AO -> shell and AO -> atom
    ao_shell = []
    ao_atom = []
    for ish, off in enumerate(basis.ao_offsets):
        n_ao = basis.ao_counts[ish]
        for k in range(n_ao):
            ao_shell.append(ish)
            ao_atom.append(basis.shells[ish].atom_index)
    ao_shell = torch.tensor(ao_shell, dtype=torch.long, device=device)
    ao_atom = torch.tensor(ao_atom, dtype=torch.long, device=device)
    Vs = V_shell[ao_shell]
    Va = V_atom[ao_atom]
    Vsum = 0.5 * (Vs.unsqueeze(1) + Vs.unsqueeze(0)) + 0.5 * (Va.unsqueeze(1) + Va.unsqueeze(0))
    H.add_(S * Vsum)
