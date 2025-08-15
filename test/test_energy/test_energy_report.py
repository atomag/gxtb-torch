import torch

from gxtb.params.loader import load_gxtb_params, load_basisq, load_eeq_params
from gxtb.params.schema import load_schema
from gxtb.basis.qvszp import build_atom_basis
from gxtb.energy.total import compute_total_energy, energy_report


def test_energy_report_includes_ofx_key_when_enabled():
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    bq = load_basisq('parameters/basisq')
    eeq = load_eeq_params('parameters/eeq')
    numbers = torch.tensor([6], dtype=torch.long)
    positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    basis = build_atom_basis(numbers, bq)
    # Provide simple per-element OFX constants via ofx_elem to exercise the reporter
    Zmax = 93
    def vec(val):
        v = torch.zeros(Zmax+1, dtype=torch.float64)
        v[6] = val
        return v
    ofx_elem = {
        'sp': vec(0.4), 'pp_off': vec(0.3), 'sd': vec(0.2), 'pd': vec(0.1),
        'dd_off': vec(0.0), 'sf': vec(0.0), 'pf': vec(0.0), 'df': vec(0.0), 'ff_off': vec(0.0)
    }
    res = compute_total_energy(numbers, positions, basis, gparams, schema, eeq=eeq,
                               total_charge=0.0, nelec=4, ofx=True, ofx_params={'alpha': 0.6, 'ofx_elem': ofx_elem})
    rep = energy_report(res)
    assert 'E_total' in rep and isinstance(rep['E_total'], float)
    assert 'E_OFX' in rep and isinstance(rep['E_OFX'], float)
    assert 'scf_n_iter' in rep and 'scf_converged' in rep

