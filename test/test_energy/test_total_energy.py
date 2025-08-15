import torch
import pytest

from gxtb.params.loader import load_gxtb_params, load_basisq, load_eeq_params
from gxtb.params.schema import load_schema
from gxtb.basis.qvszp import build_atom_basis
from gxtb.energy.total import compute_total_energy


def test_total_energy_decomposition_h2():
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    basisq = load_basisq('parameters/basisq')
    eeq = load_eeq_params('parameters/eeq')
    numbers = torch.tensor([1,1], dtype=torch.long)
    positions = torch.tensor([[0.0,0.0,0.0],[0.74,0.0,0.0]], dtype=torch.float64)
    basis = build_atom_basis(numbers, basisq)
    res = compute_total_energy(numbers, positions, basis, gparams, schema, eeq, total_charge=0.0, nelec=2, second_order=False)
    # Basic checks
    assert res.E_total.shape == ()
    assert res.E_el.shape == () and res.E_incr.shape == () and res.E_rep.shape == ()
    # Decomposition sum
    assert torch.allclose(res.E_total, res.E_el + res.E_incr + res.E_rep, atol=1e-10)


def test_total_energy_second_order_runs():
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    basisq = load_basisq('parameters/basisq')
    eeq = load_eeq_params('parameters/eeq')
    numbers = torch.tensor([1,1], dtype=torch.long)
    positions = torch.tensor([[0.0,0.0,0.0],[0.74,0.0,0.0]], dtype=torch.float64)
    basis = build_atom_basis(numbers, basisq)
    res = compute_total_energy(numbers, positions, basis, gparams, schema, eeq, total_charge=0.0, nelec=2, second_order=False)
    assert res.E_total.isfinite()
