import torch

from gxtb.params.loader import load_basisq
from gxtb.basis.qvszp import build_atom_basis
from gxtb.hamiltonian.moments_builder import build_moment_matrices
from gxtb.hamiltonian.mfx import MFXParams, build_gamma_ao, mfx_fock, mfx_energy


def _projector_like_P(S: torch.Tensor, nelec: int) -> torch.Tensor:
    # Löwdin orthonormalization; simple doubly-occupied projector onto lowest nocc
    evals, evecs = torch.linalg.eigh(S)
    X = evecs @ torch.diag(evals.clamp_min(1e-12).rsqrt()) @ evecs.T
    nocc = max(1, min(S.shape[0], nelec // 2))
    C = X
    return 2.0 * C[:, :nocc] @ C[:, :nocc].T


def test_mfx_energy_fock_linear_response_approx_for_carbon_atom():
    dtype = torch.float64
    device = torch.device("cpu")
    numbers = torch.tensor([6], dtype=torch.int64, device=device)
    positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
    bq = load_basisq('parameters/basisq')
    basis = build_atom_basis(numbers, bq)
    S, D, Q = build_moment_matrices(numbers, positions, basis)
    nao = S.shape[0]
    # Build a neutral-like projector
    P = _projector_like_P(S, nelec=4)
    # Synthetic per-element shell U and xi_l
    Zmax = 93
    U_shell = torch.zeros((Zmax+1, 4), dtype=dtype, device=device)
    # Assign moderate values for carbon Z=6
    U_shell[6] = torch.tensor([0.8, 0.9, 1.0, 1.1], dtype=dtype, device=device)
    xi_l = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=dtype, device=device)
    params = MFXParams(alpha=0.6, omega=0.5, k1=0.0, k2=0.0, U_shell=U_shell, xi_l=xi_l)
    gamma = build_gamma_ao(numbers, positions, basis, params)
    # Baseline Fock and energy
    F = mfx_fock(P, S, gamma)
    E0 = mfx_energy(P, F)
    # Small symmetric perturbation
    dP = torch.randn_like(S)
    dP = 0.5 * (dP + dP.T) * 1e-6
    # Recompute energy and Fock at P + dP
    F1 = mfx_fock(P + dP, S, gamma)
    E1 = mfx_energy(P + dP, F1)
    dE = (E1 - E0).item()
    # Since F is linear in P (Eq. 153 terms are each linear in P), δE = Tr(F δP)
    rhs = torch.einsum('ij,ji->', F, dP).item()
    assert abs(dE - rhs) < 1e-10
