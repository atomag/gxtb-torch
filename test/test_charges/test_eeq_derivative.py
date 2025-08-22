import torch, pytest

from gxtb.params.loader import load_eeq_params
from gxtb.charges.eeq import compute_eeq_charges, compute_eeq_charge_derivative


@pytest.mark.parametrize('geom', [
    torch.tensor([[0.0,0.0,0.0],[0.9,0.0,0.0]], dtype=torch.float64),
    torch.tensor([[0.0,0.0,0.0],[0.0,1.2,0.0]], dtype=torch.float64),
])
def test_eeq_charge_derivative_fd_match_cpu(geom):
    numbers = torch.tensor([1,1], dtype=torch.long)
    eeq = load_eeq_params('parameters/eeq')
    q0 = compute_eeq_charges(numbers, geom, eeq, total_charge=0.0)
    dq = compute_eeq_charge_derivative(numbers, geom, eeq, total_charge=0.0)
    assert dq.shape == (2,2,3)
    # Finite difference check for atom 0 x-displacement
    eps = 1e-5
    for X in range(2):
        for k in range(3):
            pos_p = geom.clone(); pos_p[X,k] += eps
            pos_m = geom.clone(); pos_m[X,k] -= eps
            qp = compute_eeq_charges(numbers, pos_p, eeq, total_charge=0.0)
            qm = compute_eeq_charges(numbers, pos_m, eeq, total_charge=0.0)
            dq_fd = (qp - qm) / (2*eps)
            assert torch.allclose(dq[:, X, k], dq_fd, rtol=1e-4, atol=1e-6)

