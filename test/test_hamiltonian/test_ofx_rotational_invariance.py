import torch
import math

from gxtb.params.loader import load_basisq
from gxtb.basis.qvszp import build_atom_basis
from gxtb.hamiltonian.moments_builder import build_moment_matrices
from gxtb.hamiltonian.ofx import OFXParams, ofx_energy, add_ofx_fock, build_ao_maps


def random_ortho_3x3(dtype=torch.float64, device=torch.device("cpu")):
    A = torch.randn(3, 3, dtype=dtype, device=device)
    Q, R = torch.linalg.qr(A)
    # ensure det=+1
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def test_ofx_rotational_invariance_on_p_block_carbon():
    dtype = torch.float64
    device = torch.device("cpu")
    numbers = torch.tensor([6], dtype=torch.int64, device=device)
    positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
    bq = load_basisq('parameters/basisq')
    basis = build_atom_basis(numbers, bq)
    S, D, Q = build_moment_matrices(numbers, positions, basis)
    nao = S.shape[0]
    # Build P as simple projector
    evals, evecs = torch.linalg.eigh(S)
    X = evecs @ torch.diag(evals.clamp_min(1e-12).rsqrt()) @ evecs.T
    nocc = max(1, nao // 2)
    C = X
    P = 2.0 * C[:, :nocc] @ C[:, :nocc].T
    # Onsite Λ^0 for p-shell only
    ao_atom, ao_l, groups = build_ao_maps(numbers, basis)
    Lam0 = torch.zeros((nao, nao), dtype=dtype, device=device)
    p_idx = groups.get((0, 1), [])
    if len(p_idx) >= 2:
        idx = torch.tensor(p_idx, dtype=torch.long, device=device)
        # off-diagonal 0.3, diagonal 0.8
        Lam0[idx[:, None], idx[None, :]] = 0.3
        Lam0[idx, idx] = 0.8
    params = OFXParams(alpha=0.7, Lambda0_ao=Lam0)
    # Compute original E,H
    E1 = ofx_energy(numbers, basis, P, S, params)
    H1 = torch.zeros_like(S)
    add_ofx_fock(H1, numbers, basis, P, S, params)
    # Build AO rotation R that rotates only p-subspace
    R = torch.eye(nao, dtype=dtype, device=device)
    if len(p_idx) >= 2:
        Qp = random_ortho_3x3(dtype, device)
        # some systems may have fewer than 3 p AOs (e.g., spherical set size=3). Clip shape accordingly
        k = min(3, len(p_idx))
        Qsub = Qp[:k, :k]
        for ii, aoi in enumerate(p_idx[:k]):
            for jj, aoj in enumerate(p_idx[:k]):
                R[aoi, aoj] = Qsub[ii, jj]
    # Rotate S,P,Lam0
    S2 = R.T @ S @ R
    P2 = R.T @ P @ R
    Lam2 = R.T @ Lam0 @ R
    params2 = OFXParams(alpha=params.alpha, Lambda0_ao=Lam2)
    E2 = ofx_energy(numbers, basis, P2, S2, params2)
    H2 = torch.zeros_like(S2)
    add_ofx_fock(H2, numbers, basis, P2, S2, params2)
    # Energy invariance
    assert abs((E2 - E1).item()) < 1e-10
    # Fock covariant transform: H2 ≈ R^T H1 R
    H1r = R.T @ H1 @ R
    assert torch.allclose(H2, H1r, atol=1e-10, rtol=1e-10)

