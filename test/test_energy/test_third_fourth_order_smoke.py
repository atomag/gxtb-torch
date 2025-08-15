import torch

from gxtb.params.loader import load_gxtb_params, load_basisq, load_eeq_params
from gxtb.params.schema import load_schema
from gxtb.basis.qvszp import build_atom_basis
from gxtb.energy.total import compute_total_energy


def test_third_order_smoke_runs():
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    eeq = load_eeq_params('parameters/eeq')
    bq = load_basisq('parameters/basisq')
    numbers = torch.tensor([6, 1, 1], dtype=torch.long)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [-0.4, 0.9, 0.0]], dtype=torch.float64)
    basis = build_atom_basis(numbers, bq)
    res = compute_total_energy(numbers, positions, basis, gparams, schema, eeq,
                               total_charge=0.0, nelec=8, third_order=True, second_order=True, shell_second_order=True)
    assert isinstance(res.scf.converged, bool)
    # E3 optional; ensure SCF ran
    assert res.scf.E_elec is not None


def test_fourth_order_smoke_runs():
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    eeq = load_eeq_params('parameters/eeq')
    bq = load_basisq('parameters/basisq')
    numbers = torch.tensor([6, 1, 1], dtype=torch.long)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [-0.4, 0.9, 0.0]], dtype=torch.float64)
    basis = build_atom_basis(numbers, bq)
    # Pick a small gamma4 for stability
    res = compute_total_energy(numbers, positions, basis, gparams, schema, eeq,
                               total_charge=0.0, nelec=8, fourth_order=True, gamma4=0.1)
    assert isinstance(res.scf.converged, bool)
    assert res.scf.E_elec is not None
