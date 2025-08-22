from __future__ import annotations

"""k-point SCF driver under PBC with atomic-level second-order via Ewald.

Equations:
 - Generalized eigenproblem per k: F(k) C(k) = S(k) C(k) E(k) (doc/theory/5, Eq. 12)
 - Band energy: Σ_k w_k Σ_occ ε_{n}(k)
 - Atomic second-order energy: 1/2 Δq^T γ^{(2)} Δq (doc/theory/15, Eq. 100b)
 - Fock update (atomic-level simplification): H_diag += ΔV_A with ΔV = γ^{(2)} Δq (Eq. 103, simplified Eq. 104)

Notes:
 - Shell-resolved off-diagonal Fock (Eq. 105b) is not included here to keep parity with atomic-level second_order path.
 - γ^{(2)} is computed once using PBC Ewald splitting (src/gxtb/pbc/second_order_pbc.py) and reused across iterations.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Callable, Optional, Tuple
import torch

Tensor = torch.Tensor

@dataclass
class SCFKResult:
    E_band: Tensor
    E2: Tensor
    E_extra: Tensor | None
    q: Tensor
    n_iter: int
    converged: bool
    Pk_final: List[Tensor] | None


def _lowdin(S: Tensor, spd_floor: float = 1e-8) -> Tensor:
    evals, evecs = torch.linalg.eigh(S)
    # Allow tiny negative eigenvalues up to -spd_floor as numerical noise, project to spd_floor
    # Project to SPD by flooring eigenvalues at spd_floor (standard symmetric orthogonalization safeguard)
    d = torch.clamp(evals.real, min=float(spd_floor)).rsqrt()
    return (evecs * d) @ evecs.conj().T


def _density_from_orthogonal(H: Tensor, S: Tensor, nelec: int, spd_floor: float = 1e-8) -> tuple[Tensor, Tensor]:
    """Return (P, evals) for a single k using Löwdin orthogonalization."""
    X = _lowdin(S, spd_floor=spd_floor)
    Ht = X.conj().T @ H @ X
    evals, Ctil = torch.linalg.eigh(Ht)
    nocc = nelec // 2
    Cocc = Ctil[:, :nocc]
    # Back-transform to AO
    C = X @ Cocc
    P = 2.0 * (C @ C.conj().T).real
    return P, evals


def scf_k(
    numbers: Tensor,
    basis,
    S_k: Sequence[Tensor],
    H0_k: Sequence[Tensor],
    ao_atoms: Tensor,
    K_weights: Tensor,
    nelec: int,
    *,
    gamma2_atomic: Tensor,
    q_ref: Tensor,
    max_iter: int = 50,
    tol: float = 1e-6,
    mix: float = 0.5,
    h_extra_builder: Optional[Callable[[List[Tensor], Tensor], Tuple[List[Tensor], Tensor]]] = None,
    k_builder: Optional[Callable[[Tensor], Tuple[List[Tensor], List[Tensor]]]] = None,
) -> SCFKResult:
    """Run a simple closed-shell SCF across k-points with atomic-level second-order."""
    nk = len(S_k)
    assert nk == len(H0_k)
    # Initialize charges and band energy
    nat = len(numbers)
    base_dtype = H0_k[0].real.dtype
    device = H0_k[0].device
    q = torch.zeros(nat, dtype=base_dtype, device=device)
    E_band = torch.tensor(0.0, dtype=base_dtype, device=device)
    converged = False
    Pk_prev: Optional[List[Tensor]] = None
    E_extra_last: Optional[Tensor] = None
    for it in range(1, max_iter + 1):
        # Optional dynamic rebuild of k-dependent core matrices from current charges (q)
        if k_builder is not None:
            S_k_cur, H0_k_cur = k_builder(q)
        else:
            S_k_cur, H0_k_cur = S_k, H0_k
        # Potential from current Δq
        dq = (q - q_ref)
        V = gamma2_atomic @ dq
        # Assemble H_k with onsite shifts
        Hk_list: List[Tensor] = []
        for k in range(nk):
            Hk = H0_k_cur[k].clone()
            # Add onsite shifts: H_{μμ} += V_{atom(μ)}
            A = ao_atoms.long()
            Hk.diagonal().add_(V[A].to(dtype=Hk.dtype))
            Hk_list.append(Hk)
        # Add extra Hamiltonian (e.g., AES) from previous iteration's densities
        if h_extra_builder is not None and Pk_prev is not None:
            H_add_list, E_extra = h_extra_builder(Pk_prev, q)
            E_extra_last = E_extra
            for k in range(nk):
                Hk_list[k] = Hk_list[k] + H_add_list[k].to(dtype=Hk_list[k].dtype)
        # Solve per k and build P_k
        Pk_list = []
        evals_list = []
        for k in range(nk):
            Pk, ek = _density_from_orthogonal(Hk_list[k], S_k_cur[k], nelec, spd_floor=1e-8)
            Pk_list.append(Pk)
            evals_list.append(ek)
        Pk_prev = Pk_list
        # Mulliken charges across k
        q_new = torch.zeros_like(q)
        for k in range(nk):
            PS = Pk_list[k] @ S_k_cur[k].real
            occ = torch.diag(PS).real
            qk = torch.bincount(ao_atoms.long(), weights=occ, minlength=nat)
            q_new = q_new + K_weights[k] * qk
        # Mix charges
        q = (1.0 - mix) * q + mix * q_new
        # Band energy
        E_band = torch.tensor(0.0, dtype=S_k[0].dtype, device=S_k[0].device)
        for k in range(nk):
            nocc = nelec // 2
            E_band = E_band + K_weights[k] * evals_list[k][:nocc].sum().real
        # Convergence
        if torch.linalg.vector_norm(q - q_new) < tol:
            converged = True
            break
    # One final extra-H evaluation with final densities if builder provided
    if h_extra_builder is not None and Pk_prev is not None:
        _, E_extra_final = h_extra_builder(Pk_prev, q)
    else:
        E_extra_final = None
    E2 = 0.5 * torch.dot((q - q_ref), gamma2_atomic @ (q - q_ref))
    return SCFKResult(E_band=E_band.real, E2=E2, E_extra=E_extra_final, q=q, n_iter=it, converged=converged, Pk_final=Pk_prev)
