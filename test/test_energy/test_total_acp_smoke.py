import torch

from gxtb.params.loader import load_gxtb_params, load_basisq, load_eeq_params
from gxtb.params.schema import load_schema, map_cn_params
from gxtb.energy.total import compute_total_energy, energy_report
from gxtb.basis.qvszp import build_atom_basis


def test_total_energy_with_acp_smoke_CH():
    dtype = torch.float64
    device = torch.device("cpu")
    g = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    eeq = load_eeq_params('parameters/eeq')
    numbers = torch.tensor([6, 1], dtype=torch.long, device=device)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=dtype, device=device)
    basis = build_atom_basis(numbers, load_basisq('parameters/basisq'))
    # Build minimal but finite ACP parameters for elements present
    Zmax = max(g.elements)
    # Coefficients and exponents per (Z,l): only fill s,p; d,f left zero
    c0 = torch.zeros((Zmax+1, 4), dtype=dtype, device=device)
    xi = torch.zeros_like(c0)
    # Small magnitudes to keep energy bounded
    c0[6, 0] = 0.02; c0[6, 1] = 0.01
    c0[1, 0] = 0.01
    xi[6, 0] = 0.8; xi[6, 1] = 0.8
    xi[1, 0] = 0.8
    cn_avg = torch.ones(Zmax+1, dtype=dtype, device=device)
    cn_avg[6] = 4.0; cn_avg[1] = 1.0
    cn = map_cn_params(g, schema)
    acp_params = {
        'c0': c0, 'xi': xi,
        'k_acp_cn': 0.1, 'cn_avg': cn_avg,
        'r_cov': cn['r_cov'].to(device=device, dtype=dtype), 'k_cn': float(cn['k_cn']),
        'l_list': ("s",),
    }
    res = compute_total_energy(numbers, positions, basis, g, schema, eeq,
                               total_charge=0.0, nelec=8,
                               acp=True, acp_params=acp_params)
    assert res.scf.E_ACP is not None
    rep = energy_report(res)
    assert 'E_ACP' in rep
    assert isinstance(rep['E_ACP'], float)
