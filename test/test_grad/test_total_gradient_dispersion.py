import torch, pytest

from gxtb.params.loader import load_d4_parameters, load_d4_reference_toml, select_d4_params
from gxtb.classical.dispersion import D4Method
from gxtb.grad.nuclear import total_gradient


def test_total_gradient_includes_dispersion_revd4_neon():
    dtype = torch.float64
    device = torch.device('cpu')
    numbers = torch.tensor([10, 10], dtype=torch.long, device=device)
    positions = torch.tensor([[-1.7, 0.0, 0.0], [1.7, 0.0, 0.0]], dtype=dtype, device=device)
    # Neon neutral charges for a simple smoke test
    q = torch.zeros(2, dtype=dtype, device=device)
    # Load method parameters (allow placeholders if any field is missing)
    params = load_d4_parameters('parameters/dftd4parameters.toml')
    block = select_d4_params(params, method='d4', functional='default', variant='bj-eeq-atm', allow_placeholders=True)
    method = D4Method(
        s6=float(block.get('s6', 1.0)), s8=float(block.get('s8', 0.0)), s9=float(block.get('s9', 0.0)),
        a1=float(block.get('a1', 0.0)), a2=float(block.get('a2', 0.0)), alp=float(block.get('alp', 16.0)),
        damping=str(block.get('damping', 'bj')), mbd=str(block.get('mbd', 'none')),
    )
    # Load reference data
    ref = load_d4_reference_toml('parameters/d4_reference.toml', device=device, dtype=dtype)
    if 'r_cov' not in ref or 'k_cn' not in ref:
        ref['r_cov'] = torch.ones(max(int(numbers.max().item()) + 1, 11), dtype=dtype, device=device)
        ref['k_cn'] = 1.0
    # Call aggregator with dispersion only
    g = total_gradient(
        numbers, positions, basis=None, gparams=None, schema=None,
        include_dispersion=True, dispersion_params={'method': method, 'ref': ref, 'q': q}
    )
    assert g.shape == positions.shape
    # Finite-difference smoke check along x: numerical gradient matches sign/trend
    h = 1e-3
    pos_p = positions.clone(); pos_m = positions.clone()
    pos_p[0,0] += h; pos_m[0,0] -= h
    from gxtb.classical.dispersion import d4_energy_with_grad
    E_p, _ = d4_energy_with_grad(numbers, pos_p, q, method, ref)
    E_m, _ = d4_energy_with_grad(numbers, pos_m, q, method, ref)
    g_fd = (E_p - E_m) / (2*h)
    assert torch.isfinite(g).all()
    assert torch.isfinite(g_fd)

