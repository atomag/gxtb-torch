from __future__ import annotations

"""Range-separated Mulliken-approximated Fock exchange (MFX) per doc/theory/20_mfx.md.

Implements:
 - γ^{MFX}_{l_A l_B}(R) kernel (Eq. 149)
 - F^{MFX} via symmetric low-cost construction (Eq. 153)
 - E^{lr,MFX} = 1/2 Tr{ F^{MFX} ∘ P } (Eq. 151b)

No hidden defaults: U^{MFX}_{l}(Z), ξ_l, and R0_{AB} must be provided explicitly
via MFXParams; otherwise raise.
"""

from dataclasses import dataclass
import torch

Tensor = torch.Tensor

__doc_refs__ = {"file": "doc/theory/20_mfx.md", "eqs": ["149", "151b", "153"]}

__all__ = [
    "MFXParams",
    "build_gamma_ao",
    "mfx_fock",
    "mfx_energy",
    "__doc_refs__",
]


@dataclass(frozen=True)
class MFXParams:
    alpha: float
    omega: float
    k1: float
    k2: float
    U_shell: Tensor   # (Zmax+1, 4) per-element shell U^{MFX}_l
    xi_l: Tensor      # (4,) average exponents per shell (valence 1, polarization 2)
    R0: Tensor | None = None  # (Zmax+1, Zmax+1) reference distances; required if k2 != 0


def _f_avg(X: Tensor, Y: Tensor, xi: Tensor) -> Tensor:
    # f_avg(X,Y,ξ) = 2^{ξ-1} (XY)^{ξ/2} / (X+Y)^{ξ-1}
    return (2.0 ** (xi - 1.0)) * (X * Y).pow(xi * 0.5) / torch.clamp(X + Y, min=torch.finfo(X.dtype).eps).pow(xi - 1.0)


def build_gamma_ao(numbers: Tensor, positions: Tensor, basis, params: MFXParams) -> Tensor:
    """Construct AO-pair γ^{MFX} matrix per Eq. 149 for current geometry.

    Requires params.U_shell (Zmax+1,4), xi_l (4,), and R0 (if k2 != 0).
    """
    device = positions.device
    dtype = positions.dtype
    z = numbers.to(dtype=torch.long, device=device)
    shells = basis.shells
    # Map AO -> (atom, element, l)
    ao_atom = []
    ao_Z = []
    ao_l = []
    lmap = {'s':0,'p':1,'d':2,'f':3}
    for ish, off in enumerate(basis.ao_offsets):
        n_ao = basis.ao_counts[ish]
        sh = shells[ish]
        for k in range(n_ao):
            ao_atom.append(sh.atom_index)
            ao_Z.append(sh.element)
            ao_l.append(lmap[sh.l])
    ao_atom = torch.tensor(ao_atom, dtype=torch.long, device=device)
    ao_Z = torch.tensor(ao_Z, dtype=torch.long, device=device)
    ao_l = torch.tensor(ao_l, dtype=torch.long, device=device)
    nao = len(ao_atom)
    # Distances between AO parent atoms (avoid large (nao,nao,3) tensor)
    pos_ao = positions[ao_atom]
    R = torch.cdist(pos_ao, pos_ao)
    # Per-AO shell U and xi
    U_ao = params.U_shell[ao_Z, ao_l].to(device=device, dtype=dtype)
    xi_l = params.xi_l.to(device=device, dtype=dtype)
    xi_ao = xi_l[ao_l]
    # Average exponent per pair is max(xi_i, xi_j)
    XI = torch.maximum(xi_ao.unsqueeze(1), xi_ao.unsqueeze(0))
    # f_avg(Ui,Uj,ξ)
    Ui = U_ao.unsqueeze(1)
    Uj = U_ao.unsqueeze(0)
    Fav = _f_avg(Ui, Uj, XI)
    # Denominator term R + 1/Fav
    denom = R + 1.0 / torch.clamp(Fav, min=torch.finfo(dtype).eps)
    # Range-separated Coulomb factor
    alpha = float(params.alpha)
    omega = float(params.omega)
    coul_lr = alpha + (1.0 - alpha) * torch.erf(omega * R)
    # Exponential screening
    k1 = float(params.k1)
    k2 = float(params.k2)
    if k2 != 0.0:
        if params.R0 is None:
            raise ValueError("MFXParams.R0 required when k2 != 0 (Eq. 149)")
        Z_i = ao_Z.unsqueeze(1)
        Z_j = ao_Z.unsqueeze(0)
        R0 = params.R0.to(device=device, dtype=dtype)[Z_i, Z_j]
    else:
        R0 = torch.zeros_like(R)
    damp = torch.exp(-(k1 + k2 * R0) * R)
    gamma = coul_lr / torch.clamp(denom, min=torch.finfo(dtype).eps) * damp
    return gamma


def mfx_fock(P: Tensor, S: Tensor, gamma: Tensor) -> Tensor:
    """Construct F^{MFX} using symmetric formulation (Eq. 153).

    Returns Fock matrix same shape as P,S.
    """
    # Helper multiplications; use @ for matmul and * for Hadamard
    SP = S @ P
    SPS = SP @ S
    term1 = (gamma * (S @ P)) @ S
    term2 = gamma * SPS
    term3 = S @ (gamma * P) @ S
    term4 = S @ ((P @ S) * gamma)
    F = -0.125 * (term1 + term2 + term3 + term4)
    return F


def mfx_energy(P: Tensor, F: Tensor) -> Tensor:
    """E^{lr,MFX} = 1/2 Tr{ F^{MFX} ∘ P } (Eq. 151b)."""
    return 0.5 * (F * P).sum()
