from __future__ import annotations

"""First‑order shell‑resolved tight‑binding (doc/theory/14_first_order_tb.md).

Implements onsite and offsite first‑order terms:
  - Onsite energy:  E^{(1),on}  (Eq. 83a–83b) with CN‑dependent μ^{(1)}_{l_A} (Eq. 84)
  - Offsite energy: E^{(1),off} (Eq. 86) using shell‑resolved γ^{(2)} (Eq. 101) and Δρ^{(1)}_{0,l}

Fock contributions follow Eq. (89) for the onsite part and the analogous form
derived from Eq. (90) for the offsite part.

No placeholders: μ^{(1),0}_{l}, k^{(1),CN}_A, and global k^{(1),dis}/k^{(1),x}/k^{(1),s} must be
provided explicitly via parameters. For the offsite Δρ^{(1)}_{0,l} we expose a helper
that constructs it from electron‑configuration valence counts and basis reference
populations p^0 (Eq. 85) under the interpretation ρ_{0,l} = −p^0_{l}.
"""

from dataclasses import dataclass
from typing import Tuple
import torch

Tensor = torch.Tensor

__doc_refs__ = {"file": "doc/theory/14_first_order_tb.md", "eqs": [81, 82, "83a", "83b", 84, 85, 86, 89, 90]}

__all__ = [
    "FirstOrderParams",
    "first_order_onsite_energy_fock",
    "first_order_offsite_energy_fock",
    "build_delta_rho0_shell",
    "__doc_refs__",
]


@dataclass(frozen=True)
class FirstOrderParams:
    """First‑order TB parameter pack.

    μ10 : (Zmax+1, 4) shell‑resolved μ^{(1),0}_{l}(Z) in order l∈(s,p,d,f)
    kCN : (Zmax+1,) element‑wise k^{(1),CN}_A
    kdis, kx, ks : global switching parameters for f^{(1)} (Eq. 83b)
    """

    mu10: Tensor
    kCN: Tensor
    kdis: float
    kx: float
    ks: float
    l_list: tuple[str, ...] = ("s", "p", "d", "f")

    def l_index(self, l: str) -> int:
        return self.l_list.index(l)


def _switch_f(qA: Tensor, kdis: float, kx: float, ks: float) -> Tensor:
    """Switching function f^{(1)}(q_A) per Eq. 83b."""
    t1 = torch.erf(qA * kx - ks * kx)
    t2 = torch.erf(qA * kx + ks * kx)
    return 1.0 + float(kdis) * (t1 + t2)


def _switch_df(qA: Tensor, kdis: float, kx: float, ks: float) -> Tensor:
    """Derivative ∂f^{(1)}/∂q_A for Eq. 89 (onsite Fock)."""
    rtpi = 2.0 / torch.sqrt(torch.tensor(torch.pi, dtype=qA.dtype, device=qA.device))
    g1 = torch.exp(-(kx * (qA - ks)) ** 2)
    g2 = torch.exp(-(kx * (qA + ks)) ** 2)
    return float(kdis) * float(kx) * rtpi * (g1 + g2)


def first_order_onsite_energy_fock(
    numbers: Tensor,
    positions: Tensor,
    basis,
    q_shell: Tensor,
    q_atom: Tensor,
    cn: Tensor,
    S: Tensor,
    params: FirstOrderParams,
) -> Tuple[Tensor, Tensor]:
    """Compute E^{(1),on} (Eq. 83a–83b, 84) and F^{(1),on} (Eq. 89).

    Mapping to AO indices (Eq. 89):
      F_{μν} = Σ_A Σ_{l_A} μ_{l_A} [ f'(q_A) q_{l_A} S_{νμ}|_{ν∈A} + f(q_A) S_{νμ}|_{ν∈l_A} ]
    Using S symmetry we form a per‑column weight W_ν = f'(q_A(ν)) Σ_{l∈A(ν)} μ_l q_l + f(q_A(ν)) μ_{l(ν)},
    and assemble F = S ∘ W (column‑scaled by W_ν).
    """
    device = S.device
    dtype = S.dtype
    shells = basis.shells
    # Shell metadata
    lmap = {"s": 0, "p": 1, "d": 2, "f": 3}
    shell_atom = torch.tensor([sh.atom_index for sh in shells], dtype=torch.long, device=device)
    shell_l = torch.tensor([lmap.get(sh.l, 0) for sh in shells], dtype=torch.long, device=device)
    Z = numbers.to(device=device, dtype=torch.long)
    # μ_{l_A} per shell with CN dependence (Eq. 84)
    mu0 = params.mu10[Z[shell_atom], shell_l].to(device=device, dtype=dtype)
    kCNA = params.kCN[Z[shell_atom]].to(device=device, dtype=dtype)
    mu_shell = mu0 * (1.0 + kCNA * cn[shell_atom])
    # f(q_A) and f'(q_A)
    fA = _switch_f(q_atom.to(dtype=dtype, device=device), params.kdis, params.kx, params.ks)
    dfA = _switch_df(q_atom.to(dtype=dtype, device=device), params.kdis, params.kx, params.ks)
    # Onsite energy Eq. 83a: sum over shells μ_l(A) f(q_A) q_l(A)
    E_on = (mu_shell * fA[shell_atom] * q_shell.to(mu_shell)).sum()
    # Build per‑atom Σ_{l∈A} μ_l q_l for the f'(q) term
    nat = len(numbers)
    sum_mu_q = torch.zeros(nat, dtype=dtype, device=device)
    sum_mu_q.index_add_(0, shell_atom, (mu_shell * q_shell.to(mu_shell)))
    # AO maps to build per‑ν column weights
    ao_shell: list[int] = []
    ao_atom: list[int] = []
    for ish, off in enumerate(basis.ao_offsets):
        for k in range(basis.ao_counts[ish]):
            ao_shell.append(ish)
            ao_atom.append(basis.shells[ish].atom_index)
    ao_shell_t = torch.tensor(ao_shell, dtype=torch.long, device=device)
    ao_atom_t = torch.tensor(ao_atom, dtype=torch.long, device=device)
    # Column weights W_ν (Eq. 89 grouped): W = df(q_A(ν)) * Σ_{l∈A(ν)} μ_l q_l + f(q_A(ν)) * μ_{l(ν)}
    Wcol = dfA[ao_atom_t] * sum_mu_q[ao_atom_t] + fA[ao_atom_t] * mu_shell[ao_shell_t]
    F_on = S * Wcol.unsqueeze(0)
    return E_on, F_on


def build_delta_rho0_shell(numbers: Tensor, basis) -> Tensor:
    """Construct Δρ^{(1)}_{0,l} per shell using Eq. 85 under ρ_{0,l} = −p^0_l.

    Δρ^{(1)}_{0,l_A} = ρ_{0,l_A} + Z_{l_A} ≍ −p^0_{l_A} + Z_{l_A}.
    Here p^0_{l_A} is the reference shell population (Eq. 99) consistent with the basis,
    and Z_{l_A} is the Aufbau valence nuclear charge partitioned by shell from electron
    configuration counts.
    """
    from .second_order_tb import compute_reference_shell_populations
    p0 = compute_reference_shell_populations(numbers, basis).to(torch.float64)
    # Electron‑configuration valence counts per shell
    from .second_order_tb import _electron_configuration_valence_counts as _valence
    shells = basis.shells
    out = torch.zeros(len(shells), dtype=torch.float64)
    for ish, sh in enumerate(shells):
        z = int(sh.element)
        counts = _valence(z)
        Zl = counts.get(sh.l, 0.0)
        out[ish] = (-p0[ish]) + float(Zl)
    return out


def first_order_offsite_energy_fock(
    numbers: Tensor,
    positions: Tensor,
    basis,
    q_shell: Tensor,
    delta_rho0_shell: Tensor,
    gamma2_shell: Tensor,
    S: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Compute E^{(1),off} (Eq. 86) and its Fock contribution (Eq. 90 form).

    Implementation details:
      - Use shell‑resolved γ^{(2)} matrix (doc/theory/15 Eq. 101) evaluated for current geometry.
      - Restrict sums to inter‑atomic shell pairs (A≠B).
      - Energy: E_off = − Σ_i q_i Σ_j Δρ0_j γ_{ji} with (i,j) shell indices and atom(i)≠atom(j).
      - Fock: Group Eq. 90 into per‑column shell weights W_i = Σ_j Δρ0_j γ_{ji} (A≠B) and set
              F = − S ∘ W_col, broadcasting W over AO columns whose parent shell is i.
    """
    device = S.device
    dtype = S.dtype
    shells = basis.shells
    n_shell = len(shells)
    # Atom indices per shell; mask on different atoms for (i,j)
    atom_idx = torch.tensor([sh.atom_index for sh in shells], dtype=torch.long, device=device)
    # Energy
    dq = q_shell.to(dtype=dtype, device=device)
    dr = delta_rho0_shell.to(dtype=dtype, device=device)
    gamma = gamma2_shell.to(dtype=dtype, device=device)
    # Masked per‑column shell weights W_i = Σ_{j≠i, atom(j)≠atom(i)} Δρ0_j γ_{ji}
    W_shell = torch.zeros(n_shell, dtype=dtype, device=device)
    for i in range(n_shell):
        js = torch.nonzero(atom_idx != atom_idx[i], as_tuple=False).flatten()
        if js.numel():
            W_shell[i] = (gamma[js, i] * dr[js]).sum()
    E_off = -(dq * W_shell).sum()
    # AO column weights: map shell weights to AO columns
    ao_shell: list[int] = []
    for ish, off in enumerate(basis.ao_offsets):
        for _ in range(basis.ao_counts[ish]):
            ao_shell.append(ish)
    ao_shell_t = torch.tensor(ao_shell, dtype=torch.long, device=device)
    Wcol = W_shell[ao_shell_t]
    F_off = -(S * Wcol.unsqueeze(0))
    return E_off, F_off

