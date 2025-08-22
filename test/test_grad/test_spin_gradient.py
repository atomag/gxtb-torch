import torch

from gxtb.params.loader import load_basisq
from gxtb.basis.qvszp import build_atom_basis
from gxtb.hamiltonian.spin import SpinParams, compute_shell_magnetizations, spin_energy
from gxtb.grad.nuclear import grad_spin_energy, _build_S_raw_torch


def _finite_diff_grad(energy_fn, positions: torch.Tensor, h: float = 2e-4) -> torch.Tensor:
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


def test_spin_gradient_matches_fd_h2():
    # H2 molecule with simple α/β densities producing nonzero magnetization
    dtype = torch.float64
    numbers = torch.tensor([1, 1], dtype=torch.long)
    positions = torch.tensor([[-0.35, 0.0, 0.0], [0.35, 0.0, 0.0]], dtype=dtype)
    basisq = load_basisq('parameters/basisq')
    basis = build_atom_basis(numbers, basisq)
    nao = basis.nao
    # Construct simple α/β density matrices: one α electron in first AO, none in β
    Pa = torch.zeros((nao, nao), dtype=dtype)
    Pb = torch.zeros_like(Pa)
    Pa[0, 0] = 1.0

    # Spin parameters: kW=1 for Z=1, W0=I (sufficient to induce non-zero energy/forces)
    maxz = int(numbers.max().item())
    kW = torch.zeros(maxz + 1, dtype=dtype)
    kW[1] = 1.0
    W0 = torch.eye(4, dtype=dtype)  # only s-channel used here
    params = SpinParams(kW_elem=kW, W0=W0)

    # Build energy function using differentiable S
    def e_fun(pos):
        S = _build_S_raw_torch(numbers, pos, basis, coeffs_map=None)
        m_shell = compute_shell_magnetizations(Pa, Pb, S, basis)
        return spin_energy(numbers, basis, m_shell, params)

    # Analytical gradient via autograd path
    E, g_ana = grad_spin_energy(numbers, positions, basis, Pa, Pb, params)
    # Finite-difference gradient
    g_fd = _finite_diff_grad(e_fun, positions, h=2e-4)
    # Compare
    assert torch.allclose(g_ana, g_fd, atol=5e-6, rtol=5e-6), f"Spin grad mismatch\nana={g_ana}\nfd={g_fd}\nE={E}"

