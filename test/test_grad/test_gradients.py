import torch
import math

from gxtb.params.loader import load_basisq
from gxtb.basis.qvszp import build_atom_basis
from gxtb.hamiltonian.second_order_tb import SecondOrderParams
from gxtb.grad.nuclear import grad_second_order_atomic, grad_aes_energy
from gxtb.hamiltonian.moments_builder import build_moment_matrices
from gxtb.hamiltonian.aes import AESParams


def finite_diff_E(numbers, positions, energy_fn, eps=1e-5):
    pos = positions.clone().detach()
    nat = pos.shape[0]
    grad = torch.zeros_like(pos)
    E0 = energy_fn(pos)
    for a in range(nat):
        for k in range(3):
            dp = torch.zeros_like(pos)
            dp[a, k] = eps
            Ep = energy_fn(pos + dp)
            Em = energy_fn(pos - dp)
            grad[a, k] = (Ep - Em) / (2 * eps)
    return E0, grad


def test_second_order_grad_vs_fd():
    # H2 diatomic with simple gamma parameters
    numbers = torch.tensor([1, 1], dtype=torch.long)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]], dtype=torch.float64)
    Zmax = 10
    eta = torch.zeros(Zmax + 1, dtype=torch.float64)
    eta[1] = 0.3
    r_cov = torch.zeros(Zmax + 1, dtype=torch.float64)
    params = SecondOrderParams(eta=eta, r_cov=r_cov, kexp=0.0)
    q = torch.tensor([0.5, -0.5], dtype=torch.float64)
    q_ref = torch.zeros_like(q)
    E2, g = grad_second_order_atomic(numbers, positions, params, q, q_ref)

    def E_fn(pos):
        from gxtb.hamiltonian.second_order_tb import compute_gamma2, second_order_energy
        g2 = compute_gamma2(numbers, pos, params)
        return float(second_order_energy(g2, q, q_ref).item())

    E0, g_fd = finite_diff_E(numbers, positions, E_fn, eps=1e-5)
    assert math.isclose(E2.item(), E0, rel_tol=1e-8, abs_tol=1e-8)
    assert torch.allclose(g, g_fd, atol=1e-5)


def test_aes_grad_vs_fd_smoke():
    # H2 with identity density; AES params with zero CN shift so damping is simple
    numbers = torch.tensor([1, 1], dtype=torch.long)
    dtype = torch.float64
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=dtype)
    bq = load_basisq('parameters/basisq')
    basis = build_atom_basis(numbers, bq)
    S, D, Q = build_moment_matrices(numbers.to(dtype=dtype), positions, basis)
    nao = basis.nao
    P = torch.eye(nao, dtype=dtype)
    params = AESParams(dmp3=6.0, dmp5=6.0, mprad=torch.zeros(10), mpvcn=torch.zeros(10))
    r_cov = torch.zeros(10, dtype=dtype)
    E, g = grad_aes_energy(numbers, positions, basis, P, params, r_cov=r_cov, k_cn=0.0)

    def E_fn(pos):
        from gxtb.hamiltonian.moments_builder import build_moment_matrices
        from gxtb.hamiltonian.aes import aes_energy_and_fock
        S2, D2, Q2 = build_moment_matrices(numbers.to(dtype=dtype), pos, basis)
        E2, _ = aes_energy_and_fock(numbers, pos, basis, P, S2, D2, Q2, params, r_cov=r_cov, k_cn=0.0)
        return float(E2.item())

    E0, g_fd = finite_diff_E(numbers, positions, E_fn, eps=1e-4)
    assert math.isclose(E.item(), E0, rel_tol=1e-7, abs_tol=1e-7)
    # Allow looser tolerance due to basis/moment sensitivity
    assert torch.allclose(g, g_fd, atol=5e-4)

