import torch

from gxtb.params.loader import load_gxtb_params, load_basisq, load_eeq_params
from gxtb.params.schema import load_schema
from gxtb.basis.qvszp import build_atom_basis
from gxtb.energy.total import compute_total_energy, energy_report


def test_total_energy_with_mfx_smoke_CH():
    dtype = torch.float64
    device = torch.device("cpu")
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    eeq = load_eeq_params('parameters/eeq')
    numbers = torch.tensor([6, 1], dtype=torch.long, device=device)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=dtype, device=device)
    basis = build_atom_basis(numbers, load_basisq('parameters/basisq'))
    # Build minimal mfx_params (no exponential screening)
    Zmax = max(gparams.elements)
    U_shell = torch.zeros((Zmax+1, 4), dtype=dtype, device=device)
    U_shell[6] = torch.tensor([0.8, 0.9, 1.0, 1.1], dtype=dtype, device=device)
    U_shell[1] = torch.tensor([0.7, 0.7, 0.7, 0.7], dtype=dtype, device=device)
    xi_l = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=dtype, device=device)
    mfx_params = {'alpha': 0.6, 'omega': 0.5, 'k1': 0.0, 'k2': 0.0, 'U_shell': U_shell, 'xi_l': xi_l}
    res = compute_total_energy(numbers, positions, basis, gparams, schema, eeq,
                               total_charge=0.0, nelec=8,
                               mfx=True, mfx_params=mfx_params)
    # E_MFX must be present in SCF result and energy report
    assert res.scf.E_MFX is not None
    rep = energy_report(res)
    assert 'E_MFX' in rep
    assert abs(rep['E_MFX']) < 1e3

