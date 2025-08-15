import torch

from gxtb.params.loader import load_basisq
from gxtb.basis.qvszp import build_atom_basis
from gxtb.hamiltonian.moments_builder import build_moment_matrices
from gxtb.hamiltonian.mfx import MFXParams, build_gamma_ao, mfx_fock, mfx_energy


def random_ortho_k(k: int, dtype=torch.float64, device=torch.device("cpu")):
    A = torch.randn(k, k, dtype=dtype, device=device)
    Q, R = torch.linalg.qr(A)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def test_mfx_rotational_covariance_on_p_block_carbon():
    dtype = torch.float64
    device = torch.device("cpu")
    numbers = torch.tensor([6], dtype=torch.long, device=device)
    positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
    bq = load_basisq('parameters/basisq')
    basis = build_atom_basis(numbers, bq)
    S, D, Qm = build_moment_matrices(numbers, positions, basis)
    nao = S.shape[0]
    # Build a simple projector-like P onto lowest orbitals via Löwdin
    evals, evecs = torch.linalg.eigh(S)
    X = evecs @ torch.diag(evals.clamp_min(1e-12).rsqrt()) @ evecs.T
    nocc = max(1, nao // 2)
    C = X
    P = 2.0 * C[:, :nocc] @ C[:, :nocc].T
    # Build MFX params (no exponential screening)
    Zmax = 93
    U_shell = torch.zeros((Zmax+1, 4), dtype=dtype, device=device)
    U_shell[6] = torch.tensor([0.8, 0.9, 1.0, 1.1], dtype=dtype, device=device)
    xi_l = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=dtype, device=device)
    params = MFXParams(alpha=0.6, omega=0.5, k1=0.0, k2=0.0, U_shell=U_shell, xi_l=xi_l)
    gamma = build_gamma_ao(numbers, positions, basis, params)
    # Baseline Fock and energy
    F = mfx_fock(P, S, gamma)
    E0 = mfx_energy(P, F)
    # Rotate only p-subspace (size up to 3); construct AO rotation matrix R
    # Find p-indices on atom 0
    from gxtb.hamiltonian.ofx import build_ao_maps
    ao_atom, ao_l, groups = build_ao_maps(numbers, basis)
    p_idx = groups.get((0, 1), [])
    R = torch.eye(nao, dtype=dtype, device=device)
    if len(p_idx) >= 2:
        k = min(3, len(p_idx))
        Qp = random_ortho_k(k, dtype, device)
        for ii, aoi in enumerate(p_idx[:k]):
            for jj, aoj in enumerate(p_idx[:k]):
                R[aoi, aoj] = Qp[ii, jj]
    # Transform S,P and recompute Fock with same gamma
    S2 = R.T @ S @ R
    P2 = R.T @ P @ R
    F2 = mfx_fock(P2, S2, gamma)
    # Covariance: F2 ≈ R^T F R; energy invariance
    Fr = R.T @ F @ R
    assert torch.allclose(F2, Fr, atol=1e-10, rtol=1e-10)
    E1 = mfx_energy(P2, F2)
    assert abs((E1 - E0).item()) < 1e-10

