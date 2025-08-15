import pytest, torch
from gxtb.hamiltonian.second_order_tb import (
    SecondOrderParams,
    compute_gamma2,
    second_order_energy,
    second_order_shifts,
    apply_second_order_onsite,
    add_second_order_fock,
    second_order_energy_with_grad,
)


def test_gamma2_symmetry_and_energy():
    numbers = torch.tensor([1,1,8])  # H H O
    pos = torch.tensor([[0.0,0.0,0.0],[0.9,0.0,0.0],[0.3,0.8,0.0]], dtype=torch.float64)
    # Dummy params (eta ~1, radii ~0.5)
    maxz = int(numbers.max().item())
    eta = torch.zeros(maxz+1, dtype=torch.float64)
    r_cov = torch.zeros(maxz+1, dtype=torch.float64)
    eta[1] = 1.0; eta[8] = 1.2
    r_cov[1] = 0.3; r_cov[8] = 0.5
    params = SecondOrderParams(eta=eta, r_cov=r_cov)
    gamma2 = compute_gamma2(numbers, pos, params)
    assert torch.allclose(gamma2, gamma2.T, atol=1e-12)
    q = torch.tensor([0.1,-0.05,-0.05], dtype=torch.float64)
    q_ref = torch.zeros_like(q)
    e2 = second_order_energy(gamma2, q, q_ref)
    assert e2 > 0
    # Shifts linear in dq
    shifts = second_order_shifts(gamma2, q, q_ref)
    assert shifts.shape == q.shape
    # Build dummy AO mapping: 1s for each atom
    nao = numbers.shape[0]
    ao_atoms = torch.arange(nao)
    H = torch.zeros(nao, nao, dtype=torch.float64)
    S = torch.eye(nao, dtype=torch.float64)
    apply_second_order_onsite(H, ao_atoms, shifts)
    # Off-diagonal with identity S gives zero change
    H_before = H.clone()
    # Build atomic potential V^{(2)} (Eq. 106 atomic analogue using gamma2 * q)
    V_atom = (gamma2 @ q)
    add_second_order_fock(H, S, ao_atoms, V_atom)
    # Manual expected: 0.5*(V_A+V_B)*S_ij
    A = ao_atoms
    exp = H_before + 0.5 * (V_atom[A].unsqueeze(1) + V_atom[A].unsqueeze(0)) * S
    assert torch.allclose(H, exp, atol=1e-12)
    # Gradient support (autograd path)
    E2g, grad = second_order_energy_with_grad(numbers, pos, params, q, q_ref)
    assert E2g > 0 and grad.shape == pos.shape
