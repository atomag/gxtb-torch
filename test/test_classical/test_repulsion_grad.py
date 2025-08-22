import torch, pytest

from gxtb.params.loader import load_gxtb_params, load_eeq_params
from gxtb.params.schema import load_schema, map_repulsion_params, map_cn_params
from gxtb.classical.repulsion import RepulsionParams, repulsion_energy_and_gradient
from gxtb.charges.eeq import compute_eeq_charges


def _build_repulsion_params_zero_kq():
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    rep = map_repulsion_params(gparams, schema)
    cnm = map_cn_params(gparams, schema)
    # Zero-out kq and kq2 to avoid needing dq/dR (Eq. 58); test kernel + CN chain only
    Zmax = max(gparams.elements)
    kq0 = torch.zeros(Zmax + 1, dtype=torch.float64)
    kq20 = torch.zeros(Zmax + 1, dtype=torch.float64)
    return RepulsionParams(
        z_eff0=rep['z_eff0'], alpha0=rep['alpha0'],
        kq=kq0, kq2=kq20,
        kcn_elem=rep['kcn'], r0=rep['r0'],
        kpen1_hhe=float(rep['kpen1_hhe']), kpen1_rest=float(rep['kpen1_rest']),
        kpen2=float(rep['kpen2']), kpen3=float(rep['kpen3']), kpen4=float(rep['kpen4']),
        kexp=float(rep['kexp']), r_cov=cnm['r_cov'], k_cn=float(cnm['k_cn'])
    )


@pytest.mark.parametrize('geom', [
    torch.tensor([[0.0,0.0,0.0],[0.0,0.0,0.8]], dtype=torch.float64),
    torch.tensor([[0.0,0.0,0.0],[1.2,0.0,0.0]], dtype=torch.float64),
])
def test_repulsion_gradient_matches_fd_cpu(geom):
    numbers = torch.tensor([1,1], dtype=torch.long)
    eeq = load_eeq_params('parameters/eeq')
    params = _build_repulsion_params_zero_kq()
    q = compute_eeq_charges(numbers, geom, eeq, total_charge=0.0)
    E, g = repulsion_energy_and_gradient(geom.clone(), numbers, params, q)
    assert torch.isfinite(E)
    assert torch.isfinite(g).all()
    # Finite differences
    eps = 1e-4
    g_fd = torch.zeros_like(g)
    for a in range(geom.shape[0]):
        for k in range(3):
            pos_plus = geom.clone(); pos_plus[a,k] += eps
            pos_minus = geom.clone(); pos_minus[a,k] -= eps
            q_plus = compute_eeq_charges(numbers, pos_plus, eeq, total_charge=0.0)
            q_minus = compute_eeq_charges(numbers, pos_minus, eeq, total_charge=0.0)
            E_plus, _ = repulsion_energy_and_gradient(pos_plus, numbers, params, q_plus)
            E_minus, _ = repulsion_energy_and_gradient(pos_minus, numbers, params, q_minus)
            g_fd[a,k] = (E_plus - E_minus) / (2*eps)
    # Allow small tolerance due to EEQ charge changes (but kq,kq2=0 -> no Z^eff chain)
    assert torch.allclose(g, g_fd, rtol=1e-5, atol=1e-6)

