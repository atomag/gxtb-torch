import torch
import pytest

from gxtb.params.loader import load_gxtb_params, load_basisq
from gxtb.params.schema import load_schema
from gxtb.basis.qvszp import build_atom_basis
from gxtb.hamiltonian.eht import build_eht_hamiltonian, first_order_energy, eht_energy_gradient


def _prep(numbers, positions):
    dtype = torch.float64
    device = torch.device("cpu")
    numbers = numbers.to(device)
    positions = positions.to(device).to(dtype)
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    basisq = load_basisq('parameters/basisq')
    basis = build_atom_basis(numbers, basisq)
    res = build_eht_hamiltonian(numbers, positions, basis, gparams, schema)
    Ssc = 0.5 * (res.S_scaled + res.S_scaled.T)
    P = Ssc.clone()
    return gparams, schema, basis, P


@pytest.mark.parametrize(
    "numbers, positions",
    [
        (torch.tensor([1, 1], dtype=torch.long), torch.tensor([[-0.375, 0.0, 0.0], [0.375, 0.0, 0.0]], dtype=torch.float64)),
        (torch.tensor([6, 1], dtype=torch.long), torch.tensor([[-0.65, 0.0, 0.0], [0.65, 0.0, 0.0]], dtype=torch.float64)),
    ],
)
def test_eht_grad_translational_invariance(numbers, positions):
    gparams, schema, basis, P = _prep(numbers, positions)
    grad = eht_energy_gradient(numbers, positions, basis, gparams, schema, P)
    net = grad.sum(dim=0)
    assert torch.allclose(net, torch.zeros_like(net), atol=1e-8)


@pytest.mark.parametrize(
    "numbers, positions",
    [
        (torch.tensor([1, 1], dtype=torch.long), torch.tensor([[-0.375, 0.0, 0.0], [0.375, 0.0, 0.0]], dtype=torch.float64)),
        (torch.tensor([6, 1], dtype=torch.long), torch.tensor([[-0.65, 0.0, 0.0], [0.65, 0.0, 0.0]], dtype=torch.float64)),
    ],
)
def test_eht_grad_action_reaction_for_diatomics(numbers, positions):
    # For two-atom systems, forces must be equal and opposite
    gparams, schema, basis, P = _prep(numbers, positions)
    grad = eht_energy_gradient(numbers, positions, basis, gparams, schema, P)
    assert grad.shape[0] == 2
    assert torch.allclose(grad[0] + grad[1], torch.zeros(3, dtype=grad.dtype), atol=1e-8)


@pytest.mark.parametrize(
    "numbers, positions",
    [
        (torch.tensor([1, 1], dtype=torch.long), torch.tensor([[-0.375, 0.0, 0.0], [0.375, 0.0, 0.0]], dtype=torch.float64)),
    ],
)
def test_eht_grad_fd_convergence_h2(numbers, positions):
    # Finite-difference error decreases at smaller step sizes
    gparams, schema, basis, P = _prep(numbers, positions)
    ana = eht_energy_gradient(numbers, positions, basis, gparams, schema, P)
    # FD at multiple step sizes
    errs = []
    for h in (2e-3, 1e-3, 5e-4, 2.5e-4):
        gfd = torch.zeros_like(positions)
        for A in range(numbers.shape[0]):
            for k in range(3):
                dp = positions.clone(); dm = positions.clone()
                dp[A, k] += h; dm[A, k] -= h
                Hp = build_eht_hamiltonian(numbers, dp, basis, gparams, schema).H
                Hm = build_eht_hamiltonian(numbers, dm, basis, gparams, schema).H
                Ep = first_order_energy(P, Hp); Em = first_order_energy(P, Hm)
                gfd[A, k] = (Ep - Em) / (2.0 * h)
        errs.append((ana - gfd).abs().max().item())
    # Error should not increase as h decreases; last should be small
    assert all(errs[i+1] <= errs[i] + 1e-9 for i in range(len(errs)-1))
    assert errs[-1] < 5e-6

