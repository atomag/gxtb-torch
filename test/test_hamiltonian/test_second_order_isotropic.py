import torch

from gxtb.hamiltonian.second_order_tb import (
    SecondOrderParams,
    compute_gamma2,
    second_order_energy,
    second_order_shifts,
)


def test_gamma2_limits_and_energy_monotonic():
    # Two atoms with controllable r_cov and eta
    numbers = torch.tensor([1, 1], dtype=torch.int64)
    # Positions along x with two separations
    R1 = 1.0
    R2 = 3.0
    pos1 = torch.tensor([[0.0, 0.0, 0.0], [R1, 0.0, 0.0]], dtype=torch.float64)
    pos2 = torch.tensor([[0.0, 0.0, 0.0], [R2, 0.0, 0.0]], dtype=torch.float64)
    # r_cov zero to test pure 1/R off-diagonal; eta constant
    Zmax = 10
    eta = torch.zeros(Zmax + 1, dtype=torch.float64)
    eta[1] = 0.5
    r_cov = torch.zeros(Zmax + 1, dtype=torch.float64)
    params = SecondOrderParams(eta=eta, r_cov=r_cov, kexp=0.0)
    # gamma off-diagonal ~ 1/R
    g_far = compute_gamma2(numbers, pos2, params)
    off_far = g_far[0, 1].item()
    assert abs(off_far - 1.0 / R2) < 1e-12
    # Energy monotonic vs distance for dq=[+1,-1]
    q = torch.tensor([1.0, -1.0], dtype=torch.float64)
    q_ref = torch.zeros_like(q)
    g_near = compute_gamma2(numbers, pos1, params)
    E_near = second_order_energy(g_near, q, q_ref).item()
    E_far = second_order_energy(g_far, q, q_ref).item()
    assert E_far > E_near  # attraction weaker at longer range
    # Potential shifts linearity
    dv = second_order_shifts(g_near, q, q_ref)
    assert dv.shape == (2,)
    # Diagonal equals eta
    assert abs(g_near[0, 0].item() - eta[1].item()) < 1e-12
    assert abs(g_near[1, 1].item() - eta[1].item()) < 1e-12

