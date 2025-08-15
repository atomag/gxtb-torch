import torch

from gxtb.params.loader import load_gxtb_params, load_basisq, load_eeq_params
from gxtb.params.schema import load_schema
from gxtb.basis.qvszp import build_atom_basis
from gxtb.energy.total import compute_total_energy


def test_total_energy_with_ofx_schema_mapping():
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    bq = load_basisq('parameters/basisq')
    eeq = load_eeq_params('parameters/eeq')
    # Carbon atom
    numbers = torch.tensor([6], dtype=torch.long)
    positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    basis = build_atom_basis(numbers, bq)
    # Rely only on schema mapping for OFX Λ^0; supply alpha
    res = compute_total_energy(numbers, positions, basis, gparams, schema, eeq=eeq,
                               total_charge=0.0, nelec=4, ofx=True, ofx_params={'alpha': 0.5})
    assert res.scf is not None
    # E_OFX may be zero with current default values, but must be present
    assert res.scf.E_OFX is not None

