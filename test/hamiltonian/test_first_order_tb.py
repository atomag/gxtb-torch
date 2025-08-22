import math
import torch
import pytest

from gxtb.hamiltonian.first_order import (
    _switch_f, _switch_df,
    first_order_onsite_energy_fock,
    first_order_offsite_energy_fock,
    FirstOrderParams,
)


def test_switch_function_limits_and_derivative_cpu():
    dtype = torch.float64
    device = torch.device('cpu')
    kdis, kx, ks = 0.5, 1.2, 0.3
    # Limits q -> +inf and q -> -inf
    q_pos = torch.tensor([10.0], dtype=dtype, device=device)
    q_neg = torch.tensor([-10.0], dtype=dtype, device=device)
    f_pos = _switch_f(q_pos, kdis, kx, ks).item()
    f_neg = _switch_f(q_neg, kdis, kx, ks).item()
    assert math.isfinite(f_pos) and math.isfinite(f_neg)
    assert abs(f_pos - (1.0 + 2.0 * kdis)) < 1e-12
    assert abs(f_neg - (1.0 - 2.0 * kdis)) < 1e-12
    # Center value q=0 -> erf terms cancel, f(0)=1
    q0 = torch.tensor([0.0], dtype=dtype, device=device)
    f0 = _switch_f(q0, kdis, kx, ks).item()
    assert abs(f0 - 1.0) < 1e-12
    # Derivative check by finite differences
    q = torch.tensor([0.1], dtype=dtype, device=device)
    df_analytic = _switch_df(q, kdis, kx, ks).item()
    h = 1e-6
    f_plus = _switch_f(q + h, kdis, kx, ks).item()
    f_minus = _switch_f(q - h, kdis, kx, ks).item()
    df_fd = (f_plus - f_minus) / (2.0 * h)
    assert abs(df_analytic - df_fd) < 1e-6


def _make_minimal_basis(nshells: int):
    # Simple namespace with required attributes for first-order calls
    from types import SimpleNamespace
    shells = []
    ao_offsets = []
    ao_counts = []
    for i in range(nshells):
        shells.append(SimpleNamespace(atom_index=i, element=1, l='s', nprims=1, primitives=((1.0, 1.0, 0.0),)))
        ao_offsets.append(i)
        ao_counts.append(1)
    return SimpleNamespace(shells=shells, ao_offsets=ao_offsets, ao_counts=ao_counts, nao=nshells)


def test_first_order_onsite_energy_sign_cpu():
    device = torch.device('cpu')
    dtype = torch.float64
    numbers = torch.tensor([1, 1], dtype=torch.long, device=device)
    positions = torch.zeros((2, 3), dtype=dtype, device=device)
    basis = _make_minimal_basis(2)
    # Charges per shell (q_l) and per atom (q_A)
    q_shell = torch.tensor([0.1, -0.2], dtype=dtype, device=device)
    q_atom = torch.zeros(2, dtype=dtype, device=device)
    cn = torch.zeros(2, dtype=dtype, device=device)
    # Overlap for Fock assembly (not used in energy expression)
    S = torch.eye(2, dtype=dtype, device=device)
    # Params: only μ^{(1),0}_s for H (Z=1) non-zero, no CN scaling, no switching (kdis=0)
    mu10 = torch.zeros((2, 4), dtype=dtype, device=device)
    mu10[1, 0] = 0.5
    kCN = torch.zeros(2, dtype=dtype, device=device)
    p = FirstOrderParams(mu10=mu10, kCN=kCN, kdis=0.0, kx=1.0, ks=0.0)
    E_on, F_on = first_order_onsite_energy_fock(numbers, positions, basis, q_shell, q_atom, cn, S, p)
    # Expected: sum(mu * q_l) over shells since f(0)=1, CN=0
    expected = 0.5 * (0.1 - 0.2)
    assert torch.isfinite(E_on)
    assert abs(E_on.item() - expected) < 1e-12
    assert F_on.shape == S.shape


def test_first_order_offsite_energy_sign_cpu():
    device = torch.device('cpu')
    dtype = torch.float64
    numbers = torch.tensor([1, 1], dtype=torch.long, device=device)
    positions = torch.zeros((2, 3), dtype=dtype, device=device)
    basis = _make_minimal_basis(2)
    # Shell charges and Δρ0 per shell
    q_shell = torch.tensor([0.3, -0.4], dtype=dtype, device=device)
    dr0 = torch.tensor([1.0, -1.0], dtype=dtype, device=device)
    # γ^{(2)} shell (only off-diagonal entries contribute)
    gamma = torch.tensor([[0.0, 0.1], [0.1, 0.0]], dtype=dtype, device=device)
    S = torch.eye(2, dtype=dtype, device=device)
    E_off, F_off = first_order_offsite_energy_fock(numbers, positions, basis, q_shell, dr0, gamma, S)
    # Manual expectation: E_off = -[ q0*(γ10*dr1) + q1*(γ01*dr0) ] = -[ q0*(0.1*-1) + q1*(0.1*1) ]
    expected = - (0.3 * (-0.1) + (-0.4) * 0.1)
    assert torch.isfinite(E_off)
    assert abs(E_off.item() - expected) < 1e-12
    assert F_off.shape == S.shape

