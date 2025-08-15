import torch

from gxtb.params.loader import load_basisq
from gxtb.basis.qvszp import build_atom_basis
from gxtb.hamiltonian.moments_builder import build_moment_matrices
from gxtb.hamiltonian.ofx import OFXParams, ofx_energy, add_ofx_fock


def _projector_like_P(S: torch.Tensor, nelec: int) -> torch.Tensor:
    # Löwdin orthonormalization; simple projector onto lowest orbitals
    evals, evecs = torch.linalg.eigh(S)
    X = evecs @ torch.diag(evals.clamp_min(1e-12).rsqrt()) @ evecs.T
    nocc = max(1, min(S.shape[0], nelec // 2))
    C = X
    return 2.0 * C[:, :nocc] @ C[:, :nocc].T


def test_ofx_energy_fock_consistency_for_carbon_atom():
    dtype = torch.float64
    device = torch.device("cpu")
    numbers = torch.tensor([6], dtype=torch.int64, device=device)  # carbon
    positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
    bq = load_basisq('parameters/basisq')
    basis = build_atom_basis(numbers, bq)
    S, D, Q = build_moment_matrices(numbers, positions, basis)
    nao = S.shape[0]
    P = _projector_like_P(S, nelec=4)
    # Build a synthetic onsite Λ^0: non-zero for p-shell pairs on same atom
    from gxtb.hamiltonian.ofx import build_ao_maps
    ao_atom, ao_l, groups = build_ao_maps(numbers, basis)
    Lam0 = torch.zeros((nao, nao), dtype=dtype, device=device)
    # choose l=1 (p), assign diagonal value 0.8 and off-diagonal 0.3 within p shell
    for (A, l), idxs in groups.items():
        if l != 1:
            continue
        idx = torch.tensor(idxs, dtype=torch.long, device=device)
        Lam0[idx[:, None], idx[None, :]] = 0.3
        Lam0[idx, idx] = 0.8
    params = OFXParams(alpha=0.6, Lambda0_ao=Lam0)
    E0 = ofx_energy(numbers, basis, P, S, params)
    # Assemble Fock and verify ΔE ≈ 0.5 Tr(F ΔP) with small symmetric ΔP
    H = torch.zeros_like(S)
    add_ofx_fock(H, numbers, basis, P, S, params)
    dP = torch.randn_like(S)
    dP = 0.5 * (dP + dP.T) * 1e-5
    E1 = ofx_energy(numbers, basis, P + dP, S, params)
    dE = (E1 - E0).item()
    rhs = 0.5 * torch.einsum('ij,ji->', H, dP).item()
    assert abs(dE - rhs) < 1e-8

