import torch
import pytest

from gxtb.params.loader import load_gxtb_params, load_basisq
from gxtb.params.schema import load_schema
from gxtb.basis.qvszp import build_atom_basis
from gxtb.hamiltonian.eht import build_eht_hamiltonian, first_order_energy, eht_energy_gradient


def _finite_diff_eht_grad(numbers, positions, basis, gparams, schema, P, h=1e-4):
    """Central finite-difference gradient of E^{EHT} = Tr(P H) holding P fixed.

    Returns tensor (nat,3).
    """
    dtype = positions.dtype
    device = positions.device
    nat = numbers.shape[0]
    grad = torch.zeros((nat, 3), dtype=dtype, device=device)
    for A in range(nat):
        for k in range(3):
            dp = positions.clone(); dm = positions.clone()
            dp[A, k] += h
            dm[A, k] -= h
            Hp = build_eht_hamiltonian(numbers, dp, basis, gparams, schema).H.to(device=device, dtype=dtype)
            Hm = build_eht_hamiltonian(numbers, dm, basis, gparams, schema).H.to(device=device, dtype=dtype)
            Ep = first_order_energy(P, Hp)
            Em = first_order_energy(P, Hm)
            grad[A, k] = (Ep - Em) / (2.0 * h)
    return grad


@pytest.mark.parametrize("numbers, positions", [
    # H2 aligned on x-axis
    (torch.tensor([1, 1], dtype=torch.long), torch.tensor([[-0.375, 0.0, 0.0], [0.375, 0.0, 0.0]], dtype=torch.float64)),
])
def test_eht_overlap_gradient_matches_finite_difference_h2(numbers, positions):
    device = torch.device("cpu")
    dtype = torch.float64
    numbers = numbers.to(device)
    positions = positions.to(device)
    # Load parameters and basis
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    basisq = load_basisq('parameters/basisq')
    basis = build_atom_basis(numbers, basisq)
    # Build EHT once to get S_scaled and construct a fixed P with off-diagonal entries
    res0 = build_eht_hamiltonian(numbers, positions, basis, gparams, schema)
    Ssc = res0.S_scaled.to(device=device, dtype=dtype)
    # Use P = S_scaled (symmetric) to ensure non-zero off-diagonal block weights
    P = 0.5 * (Ssc + Ssc.T)
    # Analytical gradient
    g_ana = eht_energy_gradient(numbers, positions, basis, gparams, schema, P)
    # Finite-difference gradient holding P fixed
    g_fd = _finite_diff_eht_grad(numbers, positions, basis, gparams, schema, P, h=1e-4)
    # Tolerance: tight for double precision; Eq. 39 chain handled by autograd
    assert torch.allclose(g_ana, g_fd, atol=5e-6, rtol=5e-6), f"Analytic vs FD mismatch\nana={g_ana}\nfd={g_fd}"


@pytest.mark.xfail(strict=True, reason="CH autograd path residual mismatch under investigation (expected tracked gap)")
def test_eht_overlap_gradient_matches_finite_difference_ch():
    device = torch.device("cpu")
    dtype = torch.float64
    numbers = torch.tensor([6, 1], dtype=torch.long, device=device)
    positions = torch.tensor([[-0.65, 0.0, 0.0], [0.65, 0.0, 0.0]], dtype=dtype, device=device)
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    basisq = load_basisq('parameters/basisq')
    basis = build_atom_basis(numbers, basisq)
    res0 = build_eht_hamiltonian(numbers, positions, basis, gparams, schema)
    Ssc = res0.S_scaled.to(device=device, dtype=dtype)
    P = 0.5 * (Ssc + Ssc.T)
    tot, comps = eht_energy_gradient(numbers, positions, basis, gparams, schema, P, return_components=True)
    g_ana = tot
    g_fd = _finite_diff_eht_grad(numbers, positions, basis, gparams, schema, P, h=1e-4)
    # Provide rich debug information in assertion message
    msg = (
        f"Total mismatch\nana={g_ana}\nfd={g_fd}\n"
        f"d_eps={comps['d_eps']}\n"
        f"d_pi={comps['d_pi']}\n"
        f"d_s={comps['d_s']}\n"
        f"d_coeff={comps['d_coeff']}\n"
    )
    assert torch.allclose(g_ana, g_fd, atol=5e-6, rtol=5e-6), msg
