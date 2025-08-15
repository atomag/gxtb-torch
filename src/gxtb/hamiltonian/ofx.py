from __future__ import annotations

"""Onsite correction for Mulliken-approximated Fock exchange (OFX), doc/theory/21_ofx.md.

Implements energy (Eq. 155) and Fock (Eq. 159) using an AO-level onsite exchange
integral matrix Λ^0 provided by the caller and a global fraction α (Λ = α Λ^0).

No hidden defaults: This module does not fabricate Λ^0. Callers must pass a
preconstructed onsite Λ^0 (AO×AO, zero for offsite pairs). Energy and Fock are
computed exactly per the equations. Dual density ẐP follows Eq. 156.
"""

import torch
from dataclasses import dataclass

Tensor = torch.Tensor

__all__ = ["OFXParams", "dual_density", "ofx_energy", "add_ofx_fock", "build_ao_maps"]


@dataclass(frozen=True)
class OFXParams:
    alpha: float
    Lambda0_ao: Tensor  # (nao,nao) onsite exchange integrals Λ^0_{κλ} per AO pair (κ,λ) on same atom


def dual_density(P: Tensor, S: Tensor) -> Tensor:
    """ẐP = 1/2 (P S + S P) (Eq. 156)."""
    return 0.5 * (P @ S + S @ P)


def build_ao_maps(numbers: Tensor, basis) -> tuple[Tensor, Tensor, dict]:
    """Return AO→atom (nao,), AO→lindex (nao,), and mapping dict per (atom,l) -> list[ao]."""
    device = numbers.device
    ao_atom = []
    ao_l = []
    lmap = {"s": 0, "p": 1, "d": 2, "f": 3}
    groups: dict[tuple[int, int], list[int]] = {}
    for ish, off in enumerate(basis.ao_offsets):
        n_ao = basis.ao_counts[ish]
        A = basis.shells[ish].atom_index
        l = lmap[basis.shells[ish].l]
        for k in range(n_ao):
            ao = off + k
            ao_atom.append(A)
            ao_l.append(l)
            groups.setdefault((A, l), []).append(ao)
    return torch.tensor(ao_atom, dtype=torch.long, device=device), torch.tensor(ao_l, dtype=torch.long, device=device), groups


def ofx_energy(numbers: Tensor, basis, P: Tensor, S: Tensor, params: OFXParams) -> Tensor:
    """E^{OFX} per Eq. 155 using Λ = α Λ^0 and dual-density ẐP (Eq. 156).

    Λ^0 must be AO×AO with non-zero entries only for onsite (same-atom) AO pairs.
    """
    dtype = P.dtype
    device = P.device
    ao_atom, ao_l, groups = build_ao_maps(numbers.to(device), basis)
    nao = P.shape[0]
    Lam = torch.as_tensor(params.Lambda0_ao, dtype=dtype, device=device) * float(params.alpha)
    Ptil = dual_density(P, S)
    diag = torch.diag(Ptil)
    # E1: diagonal term over κ = λ (Eq. 155 first sum)
    E1 = torch.sum(diag * torch.diag(Lam) * diag)
    # E2: l in {p,d,f} onsite within-l pairs (Eq. 155 second sum)
    E2 = torch.zeros((), dtype=dtype, device=device)
    for l, two_l_plus_1 in ((1, 3), (2, 5), (3, 7)):
        for A in set(ao_atom.tolist()):
            idxs = groups.get((A, l), [])
            if not idxs:
                continue
            idx = torch.tensor(idxs, dtype=torch.long, device=device)
            t = diag[idx]  # tildeP_{λλ}
            Lam_sub = Lam[idx][:, idx]
            # Weight matrix W: w_{κλ} = ((2l+1)δ_{κλ} - 1)/(2l+1)
            n = idx.numel()
            W = (-1.0 / two_l_plus_1) * torch.ones((n, n), dtype=dtype, device=device)
            W.diagonal().add_(1.0)  # now diag = 1 - 1/(2l+1) = 2l/(2l+1)
            Eff = W * Lam_sub
            G = Eff @ t  # G_κ
            # E2 contribution: Σ_{κλ in l} w_{κλ} tildeP_{κκ} Λ_{κλ} tildeP_{λλ}
            # This equals t^T (Eff) t
            E2 = E2 + torch.dot(t, Eff @ t)
    return E1 + E2


def add_ofx_fock(H: Tensor, numbers: Tensor, basis, P: Tensor, S: Tensor, params: OFXParams) -> None:
    """Add F^{OFX} to H per Eq. 159.

    Splits into two parts:
      - Part 1 (μλ off-diagonal onsite): F1 = (M1 @ S^T) + (S^T @ M2)
        where M1_{μλ} = Λ_{μλ} ẐP_{μλ} [λ onsite with μ, λ≠μ], M2_{κν} analogous for ν.
      - Part 2 (within same-l on each atom): F2 = 1/2 S ∘ (G_row + G_col),
        where G_μ = Σ_{λ∈l(A)} w_{μλ} Λ_{μλ} ẐP_{λλ} with w_{μλ} as in Eq. 155.
    """
    dtype = H.dtype
    device = H.device
    ao_atom, ao_l, groups = build_ao_maps(numbers.to(device), basis)
    Lam = torch.as_tensor(params.Lambda0_ao, dtype=dtype, device=device) * float(params.alpha)
    Ptil = dual_density(P.to(device, dtype), S.to(device, dtype))
    nao = H.shape[0]
    # Masks
    same_atom = ao_atom.unsqueeze(1) == ao_atom.unsqueeze(0)
    eye = torch.eye(nao, dtype=torch.bool, device=device)
    # Part 1: off-diagonal onsite μλ terms (Eq. 159 first bracket) -> simplified vectorized form
    M = Lam * Ptil * same_atom
    # Remove diagonal in one go
    M_off = M.clone(); M_off.diagonal().zero_()
    # Use symmetry of S (S == S.T)
    F1 = M_off @ S + S @ M_off
    H.add_(F1)
    # Part 2: same-l groups (Eq. 159 second bracket) -> 1/2 S ∘ (G_row + G_col)
    diag = torch.diag(Ptil)
    g = torch.zeros(nao, dtype=dtype, device=device)
    for l, two_l_plus_1 in ((1, 3), (2, 5), (3, 7)):
        for A in set(ao_atom.tolist()):
            idxs = groups.get((A, l), [])
            if not idxs:
                continue
            idx = torch.tensor(idxs, dtype=torch.long, device=device)
            t = diag[idx]
            Lam_sub = Lam[idx][:, idx]
            n = idx.numel()
            W = (-1.0 / two_l_plus_1) * torch.ones((n, n), dtype=dtype, device=device)
            W.diagonal().add_(1.0)
            Eff = W * Lam_sub
            G_sub = Eff @ t
            g[idx] = g[idx] + G_sub
    VA = g.unsqueeze(1); VB = g.unsqueeze(0)
    H.add_(0.5 * S * (VA + VB))


def build_lambda0_ao_from_element(numbers: Tensor, basis, ofx_elem: dict, *, diag_rule: str = "zero") -> Tensor:
    """Construct AO×AO Λ^0 onsite matrix from per-element constants in ofx_elem.

    ofx_elem keys (each a tensor (Zmax+1,)):
      sp, pp_off, sd, pd, dd_off, sf, pf, df, ff_off
    diag_rule: how to fill diagonal same-l entries: "zero" (default) or "offdiag"
    """
    device = numbers.device
    dtype = torch.float64
    ao_atom, ao_l, groups = build_ao_maps(numbers.to(device), basis)
    nao = sum(basis.ao_counts)
    Lam0 = torch.zeros((nao, nao), dtype=dtype, device=device)
    # Fill within same atom pairs
    for A in set(ao_atom.tolist()):
        ZA = int(numbers[A].item())
        # within-l off-diagonal
        for l, key_off in ((1, 'pp_off'), (2, 'dd_off'), (3, 'ff_off')):
            idxs = groups.get((A, l), [])
            if not idxs or key_off not in ofx_elem:
                continue
            val = float(ofx_elem[key_off][ZA].item())
            idx = torch.tensor(idxs, dtype=torch.long, device=device)
            I, J = torch.meshgrid(idx, idx, indexing='ij')
            mask = (I != J)
            Lam0[I[mask], J[mask]] = val
            if diag_rule == 'offdiag':
                Lam0[idx, idx] = val
        # cross-l pairs (s-p, s-d, s-f, p-d, p-f, d-f)
        # Build AO index lists per l
        Ls = {l: torch.tensor(groups.get((A, l), []), dtype=torch.long, device=device) for l in (0,1,2,3)}
        # Helper to set pairs between two lists
        def set_pairs(l1, l2, key):
            if key not in ofx_elem:
                return
            idx1 = Ls[l1]; idx2 = Ls[l2]
            if idx1.numel() == 0 or idx2.numel() == 0:
                return
            val = float(ofx_elem[key][ZA].item())
            # Vectorized assignment for all cross pairs
            Lam0[idx1.unsqueeze(1), idx2.unsqueeze(0)] = val
            Lam0[idx2.unsqueeze(1), idx1.unsqueeze(0)] = val
        set_pairs(0, 1, 'sp')
        set_pairs(0, 2, 'sd')
        set_pairs(0, 3, 'sf')
        set_pairs(1, 2, 'pd')
        set_pairs(1, 3, 'pf')
        set_pairs(2, 3, 'df')
    return Lam0
