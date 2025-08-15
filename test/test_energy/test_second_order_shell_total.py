import torch

from gxtb.params.loader import load_gxtb_params, load_basisq, load_eeq_params
from gxtb.params.schema import load_schema
from gxtb.basis.qvszp import build_atom_basis
from gxtb.energy.total import compute_total_energy


def test_total_energy_with_shell_second_order_runs():
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    eeq = load_eeq_params('parameters/eeq')
    bq = load_basisq('parameters/basisq')
    numbers = torch.tensor([6, 1, 1], dtype=torch.long)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [-0.4, 0.9, 0.0]], dtype=torch.float64)
    basis = build_atom_basis(numbers, bq)
    res = compute_total_energy(numbers, positions, basis, gparams, schema, eeq,
                               total_charge=0.0, nelec=8, second_order=True, shell_second_order=True)
    # SCF should run; E2 may be None if Δq -> 0, but SCF should converge or bounded iterations
    assert isinstance(res.scf.converged, bool)
    # E_total should be finite
    assert torch.isfinite(res.E_total)
