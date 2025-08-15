from __future__ import annotations

"""Shell-resolved spin polarization (doc/theory/17_spin_polarization.md).

Energy (Eq. 120b): E^{spin} = 1/2 Σ_A Σ_{l,l'} m_{l_A} m_{l'_A} W_{l l'}(A)
with W_{l l'}(A) = k^W_A W^0_{l l'} (Eq. 121).

This module computes energy and adds the Fock-like contribution (Eq. 124) for
given magnetizations m_l and spin constants W. No defaults: caller must provide
element-wise k^W_A and reference W^0_{l l'} matrix.
"""

from dataclasses import dataclass
from typing import Tuple
import torch

Tensor = torch.Tensor

__doc_refs__ = {"file": "doc/theory/17_spin_polarization.md", "eqs": ["120b", "121", "124"]}

__all__ = ["SpinParams", "spin_energy", "add_spin_fock", "compute_shell_magnetizations", "__doc_refs__"]


@dataclass(frozen=True)
class SpinParams:
    kW_elem: Tensor  # (Zmax+1,) element-wise scale k^W_A
    W0: Tensor       # (nL,nL) reference constants W^0_{l l'} in order ('s','p','d','f')
    l_list: Tuple[str, ...] = ("s","p","d","f")

    def l_index(self, l: str) -> int:
        return self.l_list.index(l)


def spin_energy(numbers: Tensor, basis, m_shell: Tensor, params: SpinParams) -> Tensor:
    device = m_shell.device
    dtype = m_shell.dtype
    shells = basis.shells
    nat = len(numbers)
    n_sh = len(shells)
    z = numbers.to(device=device, dtype=torch.long)
    kW_A = params.kW_elem[z]
    l_idx = torch.tensor([params.l_index(sh.l) for sh in shells], dtype=torch.long, device=device)
    atom_idx = torch.tensor([sh.atom_index for sh in shells], dtype=torch.long, device=device)
    # Build per-atom magnetization vectors per l
    nL = params.W0.shape[0]
    mA = torch.zeros((nat, nL), dtype=dtype, device=device)
    for ish in range(n_sh):
        A = int(atom_idx[ish].item())
        ell = int(l_idx[ish].item())
        mA[A, ell] = mA[A, ell] + m_shell[ish]
    # Energy per atom: 1/2 m^T (kW_A * W0) m
    E = 0.0
    for A in range(nat):
        W = params.W0.to(dtype=dtype, device=device) * kW_A[A]
        E = E + 0.5 * torch.dot(mA[A], W @ mA[A])
    return E


def add_spin_fock(H: Tensor, S: Tensor, numbers: Tensor, basis, m_shell: Tensor, params: SpinParams) -> None:
    """Add F^{spin} per Eq. 124 for collinear UHF-like channels collapsed as m_shell.

    For a full UHF implementation, separate α/β contributions per Eq. 124a/b are needed.
    Here we add the symmetric S-weighted potential constructed from per-atom V^{spin}_A.
    """
    device = H.device
    dtype = H.dtype
    shells = basis.shells
    nat = len(numbers)
    z = numbers.to(device=device, dtype=torch.long)
    kW_A = params.kW_elem[z]
    l_idx = torch.tensor([params.l_index(sh.l) for sh in shells], dtype=torch.long, device=device)
    atom_idx = torch.tensor([sh.atom_index for sh in shells], dtype=torch.long, device=device)
    # Aggregate per-atom, per-l magnetization
    nL = params.W0.shape[0]
    mA = torch.zeros((nat, nL), dtype=dtype, device=device)
    for ish, m in enumerate(m_shell):
        A = int(atom_idx[ish].item())
        ell = int(l_idx[ish].item())
        mA[A, ell] = mA[A, ell] + m
    # Build per-atom V^{spin} = Σ_{l,l'} m_l W_{l l'} (use symmetric part)
    V_atom = torch.zeros(nat, dtype=dtype, device=device)
    W0 = params.W0.to(dtype=dtype, device=device)
    for A in range(nat):
        W = W0 * kW_A[A]
        V_atom[A] = (W @ mA[A]).sum()  # contract over l'
    # AO mapping
    ao_atoms = []
    for ish, sh in enumerate(shells):
        ao_atoms.extend([sh.atom_index] * basis.ao_counts[ish])
    ao_atoms_t = torch.tensor(ao_atoms, dtype=torch.long, device=device)
    VA = V_atom[ao_atoms_t].unsqueeze(1)
    VB = V_atom[ao_atoms_t].unsqueeze(0)
    H.add_(0.5 * (VA + VB) * S)


def add_spin_fock_uhf(Ha: Tensor, Hb: Tensor, S: Tensor, numbers: Tensor, basis, Pa: Tensor, Pb: Tensor, params: SpinParams) -> None:
    """Add spin Fock to α/β Hamiltonians for UHF per Eq. 124 structure.

    Builds shell magnetizations m_l(A) from Pa, Pb, then forms per-shell potentials
    V_l(A) = Σ_{l'} W_{l l'}(A) m_{l'}(A), with W(A) = k^W_A W0. Maps to AO via shells and
    adds ±(1/4) S ∘ (V(μ)+V(ν)) to Hα/Hβ, i.e., Hα += Fspin, Hβ -= Fspin.
    """
    dtype = Ha.dtype
    device = Ha.device
    # Compute shell magnetizations
    from .spin import compute_shell_magnetizations
    m_shell = compute_shell_magnetizations(Pa, Pb, S, basis)
    # Build per-atom scaling and W matrices
    Z = numbers.to(device=device, dtype=torch.long)
    kW_A = params.kW_elem[Z].to(device=device, dtype=dtype)
    W0 = params.W0.to(device=device, dtype=dtype)
    # Shell meta
    lmap = {'s':0,'p':1,'d':2,'f':3}
    shell_atom_idx = torch.tensor([sh.atom_index for sh in basis.shells], dtype=torch.long, device=device)
    shell_l_idx = torch.tensor([lmap[sh.l] for sh in basis.shells], dtype=torch.long, device=device)
    nat = numbers.shape[0]
    nL = W0.shape[0]
    # Aggregate m_l by atom
    mA = torch.zeros((nat, nL), dtype=dtype, device=device)
    for ish, m in enumerate(m_shell):
        A = int(shell_atom_idx[ish].item()); ell = int(shell_l_idx[ish].item())
        mA[A, ell] = mA[A, ell] + m
    # Scale W per atom
    W_scaled = W0.unsqueeze(0) * kW_A.view(-1,1,1)
    # V_l(A) = Σ_{l'} W_{l l'}(A) m_{l'}(A)
    VAl = torch.einsum('aij,aj->ai', W_scaled, mA)
    # Map AO -> shell and build V_ao
    ao_shell = []
    for ish, off in enumerate(basis.ao_offsets):
        for _ in range(basis.ao_counts[ish]):
            ao_shell.append(ish)
    ao_shell = torch.tensor(ao_shell, dtype=torch.long, device=device)
    V_shell = VAl[shell_atom_idx, shell_l_idx]  # (n_shell,)
    V_ao = V_shell[ao_shell]
    # Fspin AO matrix: (1/4) S ∘ (V(μ)+V(ν))
    Fspin = 0.25 * (S * (V_ao.unsqueeze(1) + V_ao.unsqueeze(0)))
    Ha.add_(Fspin)
    Hb.sub_(Fspin)


def compute_shell_magnetizations(Pa: Tensor, Pb: Tensor, S: Tensor, basis, ref_shell_pops: Tensor | None = None) -> Tensor:
    """Compute shell magnetizations m_{l_A} from α/β densities (Eq. 119b).

    m_{l_A} = (Σ_{μ∈l_A,ν} (P^α_{μν} - P^β_{μν}) S_{μν}) - (p^{0,α}_{l_A} - p^{0,β}_{l_A}).
    For neutral references, p^{0,α}_{l_A} = p^{0,β}_{l_A} ⇒ reference magnetization is zero; thus
    the last term is zero. If ref_shell_pops is provided as (p^0_l), it is ignored here
    unless a non-zero reference magnetization is desired later.
    """
    PSa = Pa @ S
    PSb = Pb @ S
    occ_a = torch.diag(PSa)
    occ_b = torch.diag(PSb)
    n_shell = len(basis.shells)
    m_shell = torch.zeros(n_shell, dtype=S.dtype, device=S.device)
    for ish, off in enumerate(basis.ao_offsets):
        n_ao = basis.ao_counts[ish]
        m_shell[ish] = (occ_a[off:off+n_ao].sum() - occ_b[off:off+n_ao].sum())
    return m_shell
