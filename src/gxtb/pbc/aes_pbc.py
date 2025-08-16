from __future__ import annotations

"""Periodic AES potentials and energy via real-space lattice sum with cutoff.

Implements periodic anisotropic electrostatics per doc/theory/16 using the
same moment definitions and damping forms as the molecular AES implementation,
but sums pair contributions over lattice translations within an explicit cutoff.

We compute atomic potentials v_mono(A), v_dip(A), v_quad(A) for atoms in the
home cell and assemble the AO-level H_AES in the home cell. Energy is computed
as E_AES = 1/2 Tr(H_AES P_total) with P_total = Σ_k w_k P_k.

Higher-order n=7/n=9 terms are not supported in this periodic path yet.
"""

from typing import Dict, List, Tuple
import torch

from ..hamiltonian.aes import AESParams
from ..hamiltonian.aes import _third_derivative_tensor, _fourth_derivative_tensor
from ..hamiltonian.aes import compute_atomic_moments as _compute_atomic_moments
from .ewald import ewald_grad_hess_1over_r
from .cn_pbc import coordination_number_pbc

Tensor = torch.Tensor

__all__ = [
    "periodic_aes_potentials",
    "assemble_aes_hamiltonian",
]


def _pair_kernels_cross(positions: Tensor, mrad: Tensor, Rvec: Tensor, dmp3: float, dmp5: float, mask_diag: bool) -> Tuple[Tensor, Tensor]:
    """Return vec_g3 (r/R^3 * fdmp3) and Hess (∇∇(1/R) * fdmp5) for cross pair with translation Rvec.

    positions: (nat,3) of home; other cell at positions + Rvec.
    mask_diag: if True and Rvec==0, mask A==B.
    """
    device = positions.device
    dtype = positions.dtype
    nat = positions.shape[0]
    pos_img = positions + Rvec
    rij = positions.unsqueeze(1) - pos_img.unsqueeze(0)  # (nat,nat,3)
    dist = torch.linalg.norm(rij, dim=-1)
    eps = torch.finfo(dtype).eps
    invR = 1.0 / torch.clamp(dist, min=eps)
    g3 = invR * invR * invR
    g5 = g3 * invR * invR
    rr = 0.5 * (mrad.unsqueeze(1) + mrad.unsqueeze(0)) * invR
    fdmp3 = 1.0 / (1.0 + 6.0 * rr.pow(float(dmp3)))
    fdmp5 = 1.0 / (1.0 + 6.0 * rr.pow(float(dmp5)))
    vec_g3 = rij * fdmp3.unsqueeze(-1) * g3.unsqueeze(-1)
    # Hessian: (3 r_i r_j - δ_ij R^2) / R^5, damped by fdmp5
    r_i = rij.unsqueeze(-1); r_j = rij.unsqueeze(-2)
    rr_mat = r_i @ r_j  # (...,3,3)
    eye3 = torch.eye(3, device=device, dtype=dtype).view(1,1,3,3)
    Hess = (3.0 * rr_mat * g5.unsqueeze(-1).unsqueeze(-1) - eye3 * g3.unsqueeze(-1).unsqueeze(-1))
    Hess = Hess * fdmp5.unsqueeze(-1).unsqueeze(-1)
    if mask_diag:
        mask = ~torch.eye(nat, dtype=torch.bool, device=device)
        vec_g3 = torch.where(mask.unsqueeze(-1), vec_g3, torch.zeros_like(vec_g3))
        Hess = torch.where(mask.unsqueeze(-1).unsqueeze(-1), Hess, torch.zeros_like(Hess))
    return vec_g3, Hess


def periodic_aes_potentials(
    numbers: Tensor,
    positions: Tensor,
    basis,
    P_total: Tensor,
    S: Tensor,
    D: Tuple[Tensor, Tensor, Tensor],
    Q: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    params: AESParams,
    *,
    r_cov: Tensor,
    k_cn: float,
    cell: Tensor,
    cutoff: float,
    ewald_eta: float,
    ewald_r_cut: float,
    ewald_g_cut: float,
    si_rules: dict | None = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute periodic AES atomic potentials and energy per unit cell.

    Returns (v_mono, v_dip, v_quad, E_AES), where v_dip has shape (nat,3), v_quad (nat,3,3).
    """
    device = positions.device
    dtype = positions.dtype
    nat = numbers.shape[0]
    # AO->atom map
    ao_atoms: List[int] = []
    for ish, off in enumerate(basis.ao_offsets):
        ao_atoms.extend([basis.shells[ish].atom_index] * basis.ao_counts[ish])
    ao_atoms_t = torch.tensor(ao_atoms, dtype=torch.long, device=device)
    # Atomic multipoles from P_total
    atoms = _compute_atomic_moments(P_total, S, D, Q, ao_atoms_t)
    qA = atoms['q'].to(device=device, dtype=dtype)
    muA = atoms['mu'].to(device=device, dtype=dtype)
    QA = atoms['Q'].to(device=device, dtype=dtype)
    # Damping radii mrad_A using PBC CN (minimum-image CN; doc/theory/9 and 25)
    cn_pbc = coordination_number_pbc(positions, numbers, r_cov.to(device=device, dtype=dtype), float(k_cn), cell, float(cutoff))
    z = numbers.long()
    rad = params.mprad[z].to(device=device, dtype=dtype)
    vcn = params.mpvcn[z].to(device=device, dtype=dtype)
    mrad = rad + vcn * cn_pbc
    # Ewald-summed kernels for all pairs in the home cell (A≠B)
    rij = positions.unsqueeze(1) - positions.unsqueeze(0)  # (nat,nat,3)
    grad, Hess = ewald_grad_hess_1over_r(rij, cell.to(device=device, dtype=dtype), float(ewald_eta), float(ewald_r_cut), float(ewald_g_cut))
    # Apply AES damping (Eq. 110a–b with damping): multiply kernels by fdmp3/fdmp5 built from mrad
    dist = torch.linalg.norm(rij, dim=-1) + torch.eye(nat, dtype=dtype, device=device) * torch.finfo(dtype).eps
    invR = 1.0 / dist
    rr = 0.5 * (mrad.unsqueeze(1) + mrad.unsqueeze(0)) * invR
    fdmp3 = 1.0 / (1.0 + 6.0 * rr.pow(float(params.dmp3)))
    fdmp5 = 1.0 / (1.0 + 6.0 * rr.pow(float(params.dmp5)))
    vec_g3 = grad * fdmp3.unsqueeze(-1)  # (nat,nat,3)
    Hess = Hess * fdmp5.unsqueeze(-1).unsqueeze(-1)      # (nat,nat,3,3)
    # Mask diagonal (A==B)
    mask = ~torch.eye(nat, dtype=torch.bool, device=device)
    vec_g3 = torch.where(mask.unsqueeze(-1), vec_g3, torch.zeros_like(vec_g3))
    Hess = torch.where(mask.unsqueeze(-1).unsqueeze(-1), Hess, torch.zeros_like(Hess))
    # Initialize potentials and energy accumulator
    v_mono = torch.zeros(nat, dtype=dtype, device=device)
    v_dip = torch.zeros(nat, 3, dtype=dtype, device=device)
    v_quad = torch.zeros(nat, 3, 3, dtype=dtype, device=device)
    E = torch.zeros((), dtype=dtype, device=device)
    # Pair contributions (same forms as molecular AES but with Ewald-summed kernels)
    qAi = qA.unsqueeze(1)
    qBj = qA.unsqueeze(0)
    muAi = muA.unsqueeze(1)
    muBj = muA.unsqueeze(0)
    # Monopole–dipole
    term_md = (qAi.unsqueeze(-1) * (muBj * vec_g3)).sum(-1) - (qBj.unsqueeze(-1) * (muAi * vec_g3)).sum(-1)
    E = E - 0.5 * term_md.sum()
    v_mono = v_mono - (muBj * vec_g3).sum(-1).sum(dim=1)
    # Dipole–dipole
    H_muB = torch.einsum('ijab, j b -> ija', Hess, muA)
    term_dd = (muA.unsqueeze(1) * H_muB).sum(-1)
    E = E + 0.5 * term_dd.sum()
    v_dip = v_dip - (qBj.unsqueeze(-1) * vec_g3).sum(dim=1) + H_muB.sum(dim=1)
    # Monopole–quadrupole
    tr_H_ThetaB = torch.einsum('ijab, j ab -> ij', Hess, QA)
    tr_H_ThetaA = torch.einsum('ijab, i ab -> ij', Hess, QA)
    E = E + 0.5 * ((qAi * tr_H_ThetaB) + (qBj * tr_H_ThetaA)).sum()
    v_mono = v_mono + 0.5 * tr_H_ThetaB.sum(dim=1)
    v_quad = v_quad + 0.5 * torch.einsum('ij, ijab -> iab', qBj, Hess)
    return v_mono, v_dip, v_quad, E


def assemble_aes_hamiltonian(
    S: Tensor,
    D: Tuple[Tensor, Tensor, Tensor],
    Q: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    ao_atoms: Tensor,
    v_mono: Tensor,
    v_dip: Tensor,
    v_quad: Tensor,
) -> Tensor:
    """Assemble AO-level H_AES from atomic potentials (home cell).

    H_AES = -0.5 [ S ∘ (V_A + V_B) + Σ D^α ∘ (V^α_A + V^α_B) + Σ Q^{αβ} ∘ (V^{αβ}_A + V^{αβ}_B) ].
    """
    device = S.device
    dtype = S.dtype
    wA = v_mono[ao_atoms]
    WA = wA.unsqueeze(1); WB = wA.unsqueeze(0)
    H = -0.5 * S * (WA + WB)
    for comp, M in enumerate(D):
        vcomp = v_dip[:, comp]
        w = vcomp[ao_atoms]
        WA = w.unsqueeze(1); WB = w.unsqueeze(0)
        H = H - 0.5 * M * (WA + WB)
    comps = [ (0,0), (0,1), (0,2), (1,1), (1,2), (2,2) ]
    for (a,b), M in zip(comps, Q):
        vcomp = v_quad[:, a, b]
        w = vcomp[ao_atoms]
        WA = w.unsqueeze(1); WB = w.unsqueeze(0)
        H = H - 0.5 * M * (WA + WB)
    return H
