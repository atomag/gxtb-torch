import torch

from gxtb.params.loader import load_gxtb_params, load_basisq, load_eeq_params
from gxtb.params.schema import load_schema
from gxtb.basis.qvszp import build_atom_basis
from gxtb.energy.total import compute_total_energy, energy_report


def test_energy_report_includes_shift_and_matches_sum():
    # H2 molecule
    numbers = torch.tensor([1,1], dtype=torch.long)
    positions = torch.tensor([[0.0,0.0,0.0],[0.74,0.0,0.0]], dtype=torch.float64)
    g = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    eeq = load_eeq_params('parameters/eeq')
    basis = build_atom_basis(numbers, load_basisq('parameters/basisq'))
    # Closed-shell 2 electrons
    res = compute_total_energy(numbers, positions, basis, g, schema, eeq, total_charge=0.0, nelec=2, second_order=False)
    rep = energy_report(res)
    assert 'E_shift' in rep
    # E_shift = E_incr + E_rep
    assert abs(rep['E_shift'] - (rep['E_incr'] + rep['E_rep'])) < 1e-12

