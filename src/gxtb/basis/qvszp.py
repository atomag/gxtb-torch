from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable

import torch

from ..params.loader import BasisQParameters, ElementBasis, ShellPrimitives
from ..cn import coordination_number

__all__ = [
    "ShellDef",
    "AtomBasis",
    "build_atom_basis",
    "compute_effective_charge",
    "build_dynamic_primitive_coeffs",
]


@dataclass(frozen=True)
class ShellDef:
    atom_index: int
    element: int
    l: str  # 's','p','d','f'
    nprims: int
    # contracted primitive tuples (alpha, c1, c2)
    primitives: Tuple[Tuple[float, float, float], ...]


@dataclass(frozen=True)
class AtomBasis:
    shells: List[ShellDef]
    # AO counts per shell for q‑vSZP
    ao_counts: List[int]
    # offsets of AO indices per atom shell sequence
    ao_offsets: List[int]
    # total number of AOs
    nao: int


def _ao_per_shell(l: str) -> int:
    if l == "s":
        return 1
    if l == "p":
        return 3
    if l == "d":
        return 5  # spherical per Eq. (37) ordering
    if l == "f":
        return 7  # per Eq. (38)
    raise ValueError(f"Unknown angular momentum l={l}")


def build_atom_basis(numbers: torch.Tensor, basis: BasisQParameters) -> AtomBasis:
    """
    Build per-atom shell definitions from q‑vSZP basis (doc/theory §1.2, Eq. 9–11).

    This constructs metadata to assemble AO blocks for overlap/Hamiltonian.
    Implementation does not compute integrals here; it organizes the shells.
    """
    shells: List[ShellDef] = []
    ao_counts: List[int] = []
    ao_offsets: List[int] = []
    nao = 0
    for ia, z in enumerate(numbers.tolist()):
        eb: ElementBasis = basis.elements[int(z)]
        # order shells deterministically: s, p, d, f if present
        for l in ("s", "p", "d", "f"):
            if l not in eb.shells:
                continue
            for block in eb.shells[l]:
                prims = tuple((p.alpha, p.c1, p.c2) for p in block.primitives)
                shells.append(ShellDef(ia, int(z), l, block.nprims, prims))
                n_ao = _ao_per_shell(l)
                ao_counts.append(n_ao)
                ao_offsets.append(nao)
                nao += n_ao

    return AtomBasis(shells=shells, ao_counts=ao_counts, ao_offsets=ao_offsets, nao=nao)


def compute_effective_charge(
    numbers: torch.Tensor,
    positions: torch.Tensor,
    q_scf: torch.Tensor,
    q_eeqbc: torch.Tensor,
    *,
    r_cov: torch.Tensor,
    k_cn: float,
    k0: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    k3: torch.Tensor,
) -> torch.Tensor:
    """Compute q^{eff}_A per doc/theory/7 Eq. (28).

    q_eff(A) = k0_A [ q_A + k1_A q_A^2 ] + k2_A sqrt(CN_A) + k3_A CN_A q_A^{EEQBC}

    All k* tensors must be shaped (Zmax+1,). No defaults are applied.
    """
    device = positions.device
    dtype = positions.dtype
    if not (q_scf.shape == q_eeqbc.shape == (numbers.shape[0],)):
        raise ValueError("q_scf and q_eeqbc must have shape (nat,)")
    cn = coordination_number(positions, numbers, r_cov.to(device=device, dtype=dtype), float(k_cn))
    Z = numbers.to(device=device, dtype=torch.long)
    k0A = k0[Z].to(device=device, dtype=dtype)
    k1A = k1[Z].to(device=device, dtype=dtype)
    k2A = k2[Z].to(device=device, dtype=dtype)
    k3A = k3[Z].to(device=device, dtype=dtype)
    return k0A * (q_scf + k1A * q_scf * q_scf) + k2A * torch.sqrt(torch.clamp(cn, min=0.0)) + k3A * cn * q_eeqbc


def build_dynamic_primitive_coeffs(
    numbers: torch.Tensor,
    basis: AtomBasis,
    q_eff: torch.Tensor,
) -> List[torch.Tensor]:
    """Return per-shell primitive contraction coefficient vectors c (len=nprims).

    Uses basis primitives (alpha, c0, c1) and applies Eq. (27): c = c0 + c1 * q_eff(A),
    where q_eff(A) is the effective charge for the parent atom A of the shell.
    """
    device = q_eff.device
    dtype = torch.float64
    coeffs: List[torch.Tensor] = []
    for sh in basis.shells:
        A = sh.atom_index
        qe = q_eff[A].to(device=device)
        c0 = torch.tensor([p[1] for p in sh.primitives], dtype=dtype, device=device)
        c1 = torch.tensor([p[2] for p in sh.primitives], dtype=dtype, device=device)
        coeffs.append(c0 + c1 * qe)
    return coeffs
