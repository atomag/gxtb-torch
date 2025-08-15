import torch

from gxtb.params.loader import load_gxtb_params, load_basisq
from gxtb.params.schema import load_schema, map_hubbard_params
from gxtb.basis.qvszp import build_atom_basis
from gxtb.hamiltonian.scf_adapter import build_eht_core, make_core_builder
from gxtb.scf import scf


def _zp(Zmax, zvals):
    out = torch.zeros(Zmax+1, dtype=torch.float64)
    for z, v in zvals.items():
        out[z] = v
    return out


def test_scf_acp_reports_energy_for_CH():
    dtype = torch.float64
    device = torch.device("cpu")
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    numbers = torch.tensor([6, 1], dtype=torch.long, device=device)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=dtype, device=device)
    basis = build_atom_basis(numbers, load_basisq('parameters/basisq'))
    core = build_eht_core(numbers, positions, basis, gparams, schema)
    S = core['S'].to(dtype=dtype)
    ao_atoms = core['ao_atoms']
    builder = make_core_builder(basis, gparams, schema)
    hub = map_hubbard_params(gparams, schema)
    # ACP parameters
    Zmax = max(gparams.elements)
    c0 = torch.zeros((Zmax+1, 4), dtype=dtype, device=device)
    xi = torch.zeros_like(c0)
    c0[6, 0] = 0.3; c0[6, 1] = 0.2
    c0[1, 0] = 0.1; c0[1, 1] = 0.05
    xi[6, 0] = 0.8; xi[6, 1] = 0.7
    xi[1, 0] = 0.6; xi[1, 1] = 0.5
    r_cov = _zp(Zmax, {1: 0.3, 6: 0.7})
    cn_avg = _zp(Zmax, {1: 1.0, 6: 4.0})
    acp_params = {
        'c0': c0,
        'xi': xi,
        'k_acp_cn': 0.0,
        'cn_avg': cn_avg,
        'r_cov': r_cov,
        'k_cn': 1.0,
        'l_list': ("s","p"),
    }
    res = scf(numbers, positions, basis, builder, S, hubbard=hub, ao_atoms=ao_atoms, nelec=8,
              acp=True, acp_params=acp_params)
    assert res.E_ACP is not None
    assert abs(float(res.E_ACP.item())) < 1e3

