from __future__ import annotations
"""Second-order shell tight-binding corrections (Eqs. 100b, 101, 102, 103–106).

E^{(2)} = 1/2 * Σ_{A,B} Δq_A γ^{(2)}_{AB} Δq_B   (Eq. 100b)
γ^{(2)} off-diagonal modeled by shielded Coulomb (Eq. 101 form):
  γ^{(2)}_{AB} = 1 / sqrt(R_{AB}^2 + (R_A^{cov} + R_B^{cov})^2), A≠B
On-site term (Eq. 102) approximated via per-element hardness η_A.
Atomic potential shift (Eq. 103): ΔV_A = Σ_B γ^{(2)}_{AB} Δq_B.
We apply only diagonal AO shifts (simplified Eq. 104); off-diagonal Fock updates
(Eqs. 105–106) deferred for future refinement.
"""
from dataclasses import dataclass
import torch
from typing import Optional, Dict
from ..params.loader import EEQParameters  # retained for future param building (may remove if unused)

Tensor = torch.Tensor

__all__ = [
    # Atomic-level (tests expect these names)
    'SecondOrderParams',
    'compute_gamma2',
    'second_order_energy',
    'second_order_shifts',
    'apply_second_order_onsite',
    'second_order_energy_with_grad',
    'add_second_order_fock',
    # Shell-resolved utilities (design extension)
    'ShellSecondOrderParams',
    'build_shell_second_order_params',
    'compute_shell_charges',
    'compute_gamma2_shell',
    'compute_reference_shell_populations',
]

# ---------------- Atomic-level implementation (Eqs. 100b–106) -----------------
@dataclass(frozen=True)
class SecondOrderParams:
    """Atomic second-order parameters per doc/theory/15_second_order_tb.md.

    - eta[z]: on-site second-order diagonal η_A (Eq. 102 analogue at atomic level).
    - r_cov[z]: covalent radius entering shielded Coulomb (Eq. 101 form).
    - kexp: optional global exponential damping in R (design choice; set 0 to disable).

    Eq. citations:
        - Eq. (100b) energy form E^{(2)} = 1/2 Σ_A Σ_B Δq_A γ^{(2)}_{AB} Δq_B
        - Eq. (101) shielded-Coulomb off-diagonal γ^{(2)}_{AB}
        - Eq. (103) atomic potential shift ΔV_A = Σ_B γ^{(2)}_{AB} Δq_B
        - Eq. (104) onsite AO shifts; Eq. (105b) Fock-like update handled by add_second_order_fock
    """
    eta: Tensor  # shape (Zmax+1,)
    r_cov: Tensor  # shape (Zmax+1,)
    kexp: float = 0.0


def compute_gamma2(
    numbers: Tensor,
    positions: Tensor,
    params: SecondOrderParams,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Build atomic γ^{(2)} per Eqs. (101,102).

    Off-diagonal (A≠B): 1 / sqrt(R_AB^2 + (R^cov_A + R^cov_B)^2) * exp(-kexp R_AB)
    Diagonal (A=B): η_A (from params.eta via Z lookup) [Eq. 102 atomic analogue].
    """
    if device is None:
        device = positions.device
    if dtype is None:
        dtype = positions.dtype
    z = numbers.to(dtype=torch.long, device=device)
    rA = params.r_cov[z].to(device=device, dtype=dtype)
    # Pairwise distances R_AB
    R = torch.cdist(positions, positions)
    # rA + rB matrix
    rsum = rA.unsqueeze(1) + rA.unsqueeze(0)
    denom = torch.sqrt(R * R + rsum * rsum)
    gamma_raw = 1.0 / denom
    if params.kexp != 0.0:
        gamma_raw = gamma_raw * torch.exp(-float(params.kexp) * R)
    # Robustly set diagonal to zero before adding η_A (avoid inf*0 -> NaN)
    etaA = params.eta[z].to(device=device, dtype=dtype)
    eye = torch.eye(numbers.shape[0], dtype=torch.bool, device=device)
    gamma_off = torch.where(eye, torch.zeros_like(gamma_raw), gamma_raw)
    gamma = gamma_off + torch.diag(etaA)
    return gamma


def second_order_energy(gamma2: Tensor, q: Tensor, q_ref: Tensor) -> Tensor:
    """E^{(2)} = 1/2 Δq^T γ^{(2)} Δq (Eq. 100b)."""
    dq = (q - q_ref).to(gamma2)
    return 0.5 * torch.dot(dq, gamma2 @ dq)


def second_order_shifts(gamma2: Tensor, q: Tensor, q_ref: Tensor) -> Tensor:
    """ΔV = γ^{(2)} Δq (Eq. 103)."""
    dq = (q - q_ref).to(gamma2)
    return gamma2 @ dq


def apply_second_order_onsite(H: Tensor, ao_atoms: Tensor, shifts: Tensor) -> None:
    """Apply onsite AO shifts using atomic ΔV_A (Eq. 104, diagonal-only application).

    For each AO μ on atom A: H_{μμ} += ΔV_A.
    """
    A = ao_atoms.long()
    H.diagonal().add_(shifts[A])


def second_order_energy_with_grad(
    numbers: Tensor,
    positions: Tensor,
    params: SecondOrderParams,
    q: Tensor,
    q_ref: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return E^{(2)} and ∂E^{(2)}/∂R via autograd, using γ^{(2)}(R) (Eqs. 100b–101).

    This respects ZERO-TOLERANCE: exact functional form; no screening thresholds or truncations.
    """
    pos_req = positions.detach().clone().requires_grad_(True)
    gamma2 = compute_gamma2(numbers, pos_req, params)
    E2 = second_order_energy(gamma2, q, q_ref)
    grad, = torch.autograd.grad(E2, pos_req, create_graph=False)
    return E2.detach(), grad.detach()

# ---------------- Shell-resolved implementation (Eqs. 98–106) -----------------
@dataclass(frozen=True)
class ShellSecondOrderParams:
    """Shell-resolved second-order parameter container.

    U0 : (Zmax+1, n_l) base shell Hubbard parameters U^{(2),0}_{l_A}
    kU : (Zmax+1,) element-wise CN scaling k^{(2),U}_A
    kexp : scalar global damping k^{(2)}_{exp}
    l_list : tuple of angular momentum labels defining column order in U0
    """
    U0: Tensor
    kU: Tensor
    kexp: float
    l_list: tuple[str, ...] = ("s","p","d","f")

    def n_l(self) -> int:
        return self.U0.shape[1]

    def l_index(self, l: str) -> int:
        return self.l_list.index(l)

# ---------------- Shell-resolved helper functions -----------------------------

def build_shell_second_order_params(
    max_z: int,
    eta_like: Tensor,
    *,
    l_list: tuple[str,...] = ("s","p","d","f"),
    kU: Tensor | None = None,
    kexp: float = 0.0,
    broadcast_eta: bool = True,
) -> ShellSecondOrderParams:
    """Construct shell second-order params from an 'eta-like' hardness reference.

    For initial implementation we broadcast a per-element scalar (eta_like[z]) over all shells
    if broadcast_eta=True. Future: ingest distinct per-shell U0 entries.
    """
    n_l = len(l_list)
    U0 = torch.zeros((max_z+1, n_l), dtype=eta_like.dtype)
    if broadcast_eta:
        for z in range(min(max_z+1, eta_like.shape[0])):
            U0[z, :] = eta_like[z]
    else:
        # expect eta_like already shaped (Zmax+1, n_l)
        if eta_like.dim() != 2 or eta_like.shape[1] != n_l:
            raise ValueError("eta_like must have shape (Zmax+1,n_l) when broadcast_eta=False")
        U0 = eta_like.clone()
    if kU is None:
        kU = torch.zeros(max_z+1, dtype=eta_like.dtype)
    return ShellSecondOrderParams(U0=U0, kU=kU, kexp=float(kexp), l_list=l_list)


def compute_shell_charges(P: Tensor, S: Tensor, basis, ref_shell_pops: Optional[Tensor] = None) -> Tensor:
    """Compute shell Mulliken partial charges q_{l_A} (Eqs. 98–99).

    q_{l_A} = (Σ_{μ∈l_A,ν} P_{μν} S_{μν}) - p^0_{l_A}.
    Using symmetry, Σ_{ν} P_{μν}S_{μν} is obtained from diag(P S) (as in atomic Mulliken).
    ref_shell_pops supplies p^0_{l_A}. Must be provided (no placeholder use).
    """
    PS = P @ S
    occ_mu = torch.diag(PS)  # population contribution per AO
    n_shell = len(basis.shells)
    q_shell = torch.zeros(n_shell, dtype=P.dtype, device=P.device)
    for ish, off in enumerate(basis.ao_offsets):
        n_ao = basis.ao_counts[ish]
        q_shell[ish] = occ_mu[off:off+n_ao].sum()
    if ref_shell_pops is None:
        raise ValueError("ref_shell_pops required (no placeholder allowed)")
    return q_shell - ref_shell_pops.to(q_shell)


def compute_gamma2_shell(
    numbers: Tensor,
    positions: Tensor,
    basis,
    params: ShellSecondOrderParams,
    cn: Tensor,
    *,
    device=None,
    dtype=None,
) -> Tensor:
    """Compute shell-resolved γ^{(2)} per doc/theory/15 (Eqs. 101–106).

    Form and units (atomic units clarity):
      - Distances enter as bohr (a0). In a.u., Coulomb kernel 1/R has energy units (Eh).
      - On-site limit must satisfy γ_{ii}(R→0) = U_i (Hubbard hardness for shell i on atom A).
      - We use an Ohno-like regularization that recovers Coulomb at long range and the
        correct onsite limit:
            γ_{ij}(R) = 1 / sqrt( R^2 + ((a_i + a_j)/2)^2 ),
        with a_i = 1 / U_i (bohr). For i=j, γ_{ii}(0) = 1 / a_i = U_i.
      - Optional exponential damping exp(-kexp R) uses R in bohr; kexp thus has units 1/bohr.

    CN scaling: U_i = U0_{l_A} * (1 + kU_A * CN_A) (Eq. 102).
    """
    if device is None:
        device = positions.device
    if dtype is None:
        dtype = positions.dtype
    numbers = numbers.to(device=device)
    pos = positions.to(device=device, dtype=dtype)
    cn = cn.to(device=device, dtype=dtype)
    shells = basis.shells
    n_shell = len(shells)
    atom_idx = torch.tensor([sh.atom_index for sh in shells], dtype=torch.long, device=device)
    z_list = torch.tensor([sh.element for sh in shells], dtype=torch.long, device=device)
    l_indices = torch.tensor([params.l_index(sh.l) for sh in shells], dtype=torch.long, device=device)
    U0_shell = params.U0[z_list, l_indices]
    kU_shell = params.kU[z_list]
    U_shell = U0_shell * (1.0 + kU_shell * cn[atom_idx])  # Eq. 102 scaling
    # Build R_AB per shell pair via atom indices (convert Å → bohr for Coulomb units)
    dist_atom = torch.cdist(pos, pos)  # (nat,nat), Å
    ANGSTROM_TO_BOHR = torch.tensor(1.8897261254535, dtype=dtype, device=device)
    Rb = dist_atom[atom_idx.unsqueeze(1), atom_idx.unsqueeze(0)] * ANGSTROM_TO_BOHR  # (n_shell,n_shell)
    # Ohno-like mixing length a_i = 1/U_i (bohr), combined as arithmetic mean
    ai = 1.0 / torch.clamp(U_shell, min=torch.finfo(dtype).eps)
    aij = 0.5 * (ai.unsqueeze(1) + ai.unsqueeze(0))
    denom = torch.sqrt(Rb * Rb + aij * aij)
    gamma = 1.0 / torch.clamp(denom, min=torch.finfo(dtype).eps)
    if params.kexp != 0.0:
        gamma = gamma * torch.exp(-float(params.kexp) * Rb)
    return gamma


def _electron_configuration_valence_counts(z: int) -> Dict[str, float]:
    """Conservative valence shell electron counts {s,p,d,f} for element Z.

    - Periods 2–3: ns/np only.
    - Periods 4–5: s-block (ns), d-block (ns + (n-1)d), p-block (ns/np).
    - Period 6: lanthanides (4f only with ns), 5d block (ns + 5d), p-block (ns/np).
    - Period 7: actinides (5f only with ns); otherwise small s-only.

    Prevents including f in transition metals and caps d at 10.
    """
    val: Dict[str, float] = {'s': 0.0, 'p': 0.0, 'd': 0.0, 'f': 0.0}
    if z <= 2:
        val['s'] = float(z)
    elif 3 <= z <= 10:
        v = z - 2
        s = min(2, v)
        p = max(0, v - s)
        val['s'], val['p'] = float(s), float(min(6, p))
    elif 11 <= z <= 18:
        v = z - 10
        s = min(2, v)
        p = max(0, v - s)
        val['s'], val['p'] = float(s), float(min(6, p))
    elif 19 <= z <= 36:
        if z <= 20:
            val['s'] = float(z - 18)
        elif 21 <= z <= 30:
            s = min(2, z - 18)
            d = min(10, max(0, z - 20))
            val['s'], val['d'] = float(s), float(d)
        else:
            val['s'] = 2.0
            val['p'] = float((z - 18) - 2)
    elif 37 <= z <= 54:
        if z <= 38:
            val['s'] = float(z - 36)
        elif 39 <= z <= 48:
            s = min(2, z - 36)
            d = min(10, max(0, z - 38))
            val['s'], val['d'] = float(s), float(d)
        else:
            val['s'] = 2.0
            val['p'] = float((z - 36) - 2)
    elif 55 <= z <= 86:
        if z <= 56:
            val['s'] = float(z - 54)
        elif 57 <= z <= 71:
            val['s'] = 2.0
            val['f'] = float(min(14, z - 56))
        elif 72 <= z <= 80:
            val['s'] = 2.0
            val['d'] = float(min(10, z - 70))
        else:
            val['s'] = 2.0
            val['p'] = float(max(0, z - 80))
    else:
        if 89 <= z <= 103:
            val['s'] = 2.0
            val['f'] = float(min(14, z - 88))
        else:
            val['s'] = float(min(2, max(0, z - 86)))
    return {k: v for k, v in val.items() if v > 0}

def compute_reference_shell_populations(numbers: Tensor, basis) -> Tensor:
    """Algorithmic neutral reference p^0_{l_A} (Eq. 99) for all 1<=Z<=103 (Lr) using electron configuration.

    Validation: For each atom, Σ_l p^0_{l_A} equals the sum of returned valence counts (deterministic rule set).
    If a shell from the generated valence distribution is absent in the basis, its electrons are omitted and a
    validation warning (exception) is raised to avoid silent placeholders.
    """
    shells = basis.shells
    ref = torch.zeros(len(shells), dtype=torch.float64)
    by_atom: Dict[int, list[int]] = {}
    for ish, sh in enumerate(shells):
        by_atom.setdefault(sh.atom_index, []).append(ish)
    for A, sh_indices in by_atom.items():
        z = int(numbers[A].item())
        if z < 1 or z > 103:
            raise ValueError(f"Element Z={z} outside supported range 1..103 for reference populations")
        valence_counts = _electron_configuration_valence_counts(z)
        present_l: Dict[str, list[int]] = {}
        for ish in sh_indices:
            present_l.setdefault(shells[ish].l, []).append(ish)
        # Validate presence; if missing, raise (enforces no placeholder) except allow omission only if count=0
        missing = [l for l in valence_counts.keys() if l not in present_l]
        if missing:
            raise ValueError(f"Basis missing valence shells {missing} for Z={z}; cannot form p^0 without placeholders")
        for l, count in valence_counts.items():
            idxs = present_l[l]
            share = count / len(idxs)
            for ish in idxs:
                ref[ish] = share
        # Validation sum
        assigned = sum(valence_counts.values())
        if not torch.isclose(torch.tensor(assigned, dtype=torch.float64), ref[sh_indices].sum()):
            raise RuntimeError(f"Reference population mismatch for Z={z}")
    return ref


def compute_reference_shell_populations_basis_aware(numbers: Tensor, basis) -> Tensor:
    """Basis-aware neutral reference p^0_{l_A} (Eq. 99) that ignores missing valence shells.

    This variant follows the same electron-configuration rule as
    compute_reference_shell_populations but, if the basis for an atom is missing
    some valence angular momentum channels l from the rule, those electrons are
    treated as core-like and omitted from p^0 (assigned 0 to missing shells). No
    redistribution is performed. This keeps shell charges well-defined without
    inventing placeholders when using reduced bases.

    Traceability: doc/theory/15_second_order_tb.md Eq. (99) defines p^0_{l_A} as a
    reference population per shell. When the shell is not represented in the chosen
    basis, the contribution is undefined. Here we choose the strictest basis-aware
    interpretation (drop missing shells) as an explicit, opt-in policy.
    """
    shells = basis.shells
    ref = torch.zeros(len(shells), dtype=torch.float64)
    by_atom: Dict[int, list[int]] = {}
    for ish, sh in enumerate(shells):
        by_atom.setdefault(sh.atom_index, []).append(ish)
    for A, sh_indices in by_atom.items():
        z = int(numbers[A].item())
        valence_counts = _electron_configuration_valence_counts(z)
        present_l: Dict[str, list[int]] = {}
        for ish in sh_indices:
            present_l.setdefault(shells[ish].l, []).append(ish)
        # Only assign to shells that are present in the basis
        for l, count in valence_counts.items():
            if l not in present_l:
                continue  # drop missing shell contribution
            idxs = present_l[l]
            share = count / len(idxs)
            for ish in idxs:
                ref[ish] = share
    return ref

def add_second_order_fock(H: Tensor, S: Tensor, ao_atoms: Tensor, V_atom: Tensor) -> None:
    """Add second-order Fock contribution (Eq. 105b) using pure PyTorch.

    F^{(2)}_{μν} = 1/2 S_{μν} ( V_A + V_B ),  A = atom(μ), B = atom(ν)  (Eq. 105b)
    with V_A = Σ_B γ^{(2)}_{AB} Δq_B (Eq. 106 atomic-level analogue).
    Modifies H in-place (additive update).
    """
    A = ao_atoms.long()
    V_A = V_atom[A].unsqueeze(1)
    V_B = V_atom[A].unsqueeze(0)
    H.add_(0.5 * (V_A + V_B) * S)
