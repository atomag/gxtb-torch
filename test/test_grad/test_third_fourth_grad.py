import torch

from gxtb.params.loader import load_basisq
from gxtb.basis.qvszp import build_atom_basis
from gxtb.grad.nuclear import grad_third_order_energy, grad_fourth_order_energy
from gxtb.hamiltonian.third_order import ThirdOrderParams, third_order_energy
from gxtb.hamiltonian.fourth_order import FourthOrderParams, fourth_order_energy


def _finite_diff_grad(energy_fn, positions: torch.Tensor, h: float = 1e-4) -> torch.Tensor:
    """Central finite-difference gradient of a scalar energy wrt positions (nat,3)."""
    device = positions.device
    dtype = positions.dtype
    nat = positions.shape[0]
    grad = torch.zeros((nat, 3), dtype=dtype, device=device)
    for A in range(nat):
        for k in range(3):
            dp = positions.clone(); dm = positions.clone()
            dp[A, k] += h
            dm[A, k] -= h
            Ep = energy_fn(dp)
            Em = energy_fn(dm)
            grad[A, k] = (Ep - Em) / (2.0 * h)
    return grad


def test_third_order_gradient_matches_fd_h2():
    # H2 along x; third-order kernel-only gradient (q, U fixed)
    dtype = torch.float64
    numbers = torch.tensor([1, 1], dtype=torch.long)
    positions = torch.tensor([[-0.35, 0.0, 0.0], [0.35, 0.0, 0.0]], dtype=dtype)
    basisq = load_basisq('parameters/basisq')
    basis = build_atom_basis(numbers, basisq)

    # Shell meta
    n_shell = len(basis.shells)
    assert n_shell >= 2  # H2 has one s-shell per atom

    # Fixed shell/atomic charges and U_shell to probe R-dependence only
    q_shell = torch.full((n_shell,), 0.2, dtype=dtype)
    q_atom = torch.tensor([0.20, -0.10], dtype=dtype)
    U_shell = torch.full((n_shell,), 1.5, dtype=dtype)

    # Third-order parameters (only Z=1 used; others ignored)
    gamma3_elem = torch.tensor([0.0, 0.8], dtype=dtype)  # index 1 (H)
    params = ThirdOrderParams(
        gamma3_elem=gamma3_elem,
        kGamma_l=(1.0, 1.0, 1.0, 1.0),
        k3=0.5,
        k3x=0.2,
    )

    # Analytic via autograd on kernel
    E3, g3 = grad_third_order_energy(numbers, positions, basis, q_shell, q_atom, U_shell, params)

    # Finite difference holding q_shell, q_atom, U_shell fixed
    def e_fun(pos):
        return third_order_energy(numbers, pos, basis, q_shell, q_atom, U_shell, params)

    g_fd = _finite_diff_grad(e_fun, positions, h=2e-4)

    assert torch.allclose(g3, g_fd, atol=5e-6, rtol=5e-6), f"E3 grad mismatch\nana={g3}\nfd={g_fd}\nE3={E3}"


def test_fourth_order_gradient_zero_under_fixed_q():
    # Any molecule; E^(4) depends only on q (fixed), so dE/dR = 0
    dtype = torch.float64
    numbers = torch.tensor([6, 1], dtype=torch.long)
    positions = torch.tensor([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=dtype)
    q = torch.tensor([0.1, -0.05], dtype=dtype)
    params = FourthOrderParams(gamma4=0.3)

    E4, g4 = grad_fourth_order_energy(numbers, positions, q, params)

    def e_fun(pos):
        # Keep q fixed under displacement
        return fourth_order_energy(q, params)

    g_fd = _finite_diff_grad(e_fun, positions, h=2e-4)

    zeros = torch.zeros_like(positions)
    assert torch.allclose(g4, zeros, atol=1e-12, rtol=1e-12)
    assert torch.allclose(g_fd, zeros, atol=1e-12, rtol=1e-12)

