import torch

from gxtb.params.loader import load_gxtb_params, load_basisq, load_eeq_params
from gxtb.params.schema import load_schema
from gxtb.energy.total import compute_total_energy, energy_report
from gxtb.basis.qvszp import build_atom_basis


def test_total_energy_with_aes_smoke_CH():
    dtype = torch.float64
    device = torch.device("cpu")
    g = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    eeq = load_eeq_params('parameters/eeq')
    numbers = torch.tensor([6, 1], dtype=torch.long, device=device)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=dtype, device=device)
    basis = build_atom_basis(numbers, load_basisq('parameters/basisq'))
    res = compute_total_energy(numbers, positions, basis, g, schema, eeq,
                               total_charge=0.0, nelec=8,
                               aes=True)
    assert res.scf.E_AES is not None
    rep = energy_report(res)
    assert 'E_AES' in rep
    assert isinstance(rep['E_AES'], float)
