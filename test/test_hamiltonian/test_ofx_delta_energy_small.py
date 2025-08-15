import torch

from gxtb.params.loader import load_basisq
from gxtb.basis.qvszp import build_atom_basis
from gxtb.hamiltonian.moments_builder import build_moment_matrices
from gxtb.hamiltonian.ofx import OFXParams, ofx_energy, add_ofx_fock, build_ao_maps


def _projector_like_P(S: torch.Tensor, nelec: int) -> torch.Tensor:
    evals, evecs = torch.linalg.eigh(S)
    X = evecs @ torch.diag(evals.clamp_min(1e-12).rsqrt()) @ evecs.T
    nocc = max(1, min(S.shape[0], nelec // 2))
    C = X
    return 2.0 * C[:, :nocc] @ C[:, :nocc].T


def test_ofx_delta_energy_consistency_for_carbon_atom():
    dtype = torch.float64
    device = torch.device("cpu")
    numbers = torch.tensor([6], dtype=torch.int64, device=device)
    positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
    bq = load_basisq('parameters/basisq')
    basis = build_atom_basis(numbers, bq)
    S, D, Q = build_moment_matrices(numbers, positions, basis)
    nao = S.shape[0]
    P = _projector_like_P(S, nelec=4)
    # Build Λ^0 onsite for p-shell block (diagonal and off-diagonal values)
    ao_atom, ao_l, groups = build_ao_maps(numbers, basis)
    Lam0 = torch.zeros((nao, nao), dtype=dtype, device=device)
    p_idx = groups.get((0, 1), [])
    if len(p_idx) >= 2:
        idx = torch.tensor(p_idx, dtype=torch.long, device=device)
        Lam0[idx[:, None], idx[None, :]] = 0.3
        Lam0[idx, idx] = 0.8
    params = OFXParams(alpha=0.6, Lambda0_ao=Lam0)
    E0 = ofx_energy(numbers, basis, P, S, params)
    # Assemble Fock and verify δE ≈ 0.5 Tr(F ΔP) with small symmetric ΔP
    H = torch.zeros_like(S)
    add_ofx_fock(H, numbers, basis, P, S, params)
    dP = torch.randn_like(S)
    dP = 0.5 * (dP + dP.T) * 1e-6
    E1 = ofx_energy(numbers, basis, P + dP, S, params)
    dE = (E1 - E0).item()
    rhs = 0.5 * torch.einsum('ij,ji->', H, dP).item()
    assert abs(dE - rhs) < 1e-8

