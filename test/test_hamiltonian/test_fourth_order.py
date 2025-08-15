import torch
from gxtb.hamiltonian.fourth_order import FourthOrderParams, fourth_order_energy, add_fourth_order_fock


def test_fourth_order_energy_simple():
    q = torch.tensor([0.1, -0.05, 0.2], dtype=torch.float64)
    params = FourthOrderParams(gamma4=2.0)
    E = fourth_order_energy(q, params)
    # Expected: (2/24) * sum(q^4)
    exp = (2.0/24.0) * (q**4).sum()
    assert torch.allclose(E, exp)


def test_fourth_order_fock_adds_diagonal_when_S_is_I():
    q = torch.tensor([0.1, -0.05], dtype=torch.float64)
    params = FourthOrderParams(gamma4=1.5)
    # 1 AO per atom
    S = torch.eye(2, dtype=torch.float64)
    H = torch.zeros_like(S)
    ao_atoms = torch.tensor([0, 1])
    add_fourth_order_fock(H, S, ao_atoms, q, params)
    # With S=I, diagonal increments: (1/2)*(V_A+V_A) = V_A; V_A = (q_A^3)*(gamma4/6)
    V = (q**3) * (params.gamma4 / 6.0)
    assert torch.allclose(torch.diag(H), V)
    assert torch.allclose(H[0, 1], torch.tensor(0.0, dtype=H.dtype))
