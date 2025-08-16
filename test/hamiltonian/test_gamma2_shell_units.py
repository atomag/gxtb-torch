import torch

from gxtb.basis.qvszp import build_atom_basis
from gxtb.params.loader import load_basisq
from gxtb.hamiltonian.second_order_tb import build_shell_second_order_params, compute_gamma2_shell


def test_gamma2_shell_units_and_limits():
    # Two hydrogen atoms far apart; each has one s shell in q-vSZP
    numbers = torch.tensor([1, 1], dtype=torch.long)
    R_ang = 10.0  # Å, large separation
    positions = torch.tensor([[0.0, 0.0, 0.0], [R_ang, 0.0, 0.0]], dtype=torch.float64)
    basis_params = load_basisq('parameters/basisq')
    basis = build_atom_basis(numbers, basis_params)
    # Set U0 from a simple constant hardness (~1 Eh) and zero CN scaling
    U_like = torch.zeros(200, dtype=torch.float64)
    U_like[1] = 1.0
    sp = build_shell_second_order_params(max_z=200, eta_like=U_like, kU=torch.zeros(200, dtype=torch.float64))
    cn = torch.zeros(2, dtype=torch.float64)
    gamma = compute_gamma2_shell(numbers, positions, basis, sp, cn)
    # Diagonal onsite should equal U (≈1.0)
    diag = torch.diag(gamma)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-8)
    # Off-diagonal ~ 1 / sqrt( (R_bohr)^2 + ((a_i+a_j)/2)^2 ) with a_i=1/U_i=1 bohr
    ANG_TO_BOHR = 1.8897261254535
    Rb = R_ang * ANG_TO_BOHR
    aij = 0.5 * (1.0 + 1.0)
    expected = 1.0 / ( (Rb ** 2 + aij ** 2) ** 0.5 )
    off = gamma[0, 1].item()
    assert abs(off - expected) / expected < 1e-10
