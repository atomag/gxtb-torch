import torch

from gxtb.params.loader import load_d4_parameters, load_d4_reference_toml, select_d4_params
from gxtb.classical.dispersion import D4Method, d4_energy_with_grad


def _finite_diff_grad(energy_fn, positions: torch.Tensor, h: float = 2e-4) -> torch.Tensor:
    device = positions.device
    dtype = positions.dtype
    nat = positions.shape[0]
    grad = torch.zeros((nat, 3), dtype=dtype, device=device)
    for A in range(nat):
        for k in range(3):
            dp = positions.clone(); dm = positions.clone()
            dp[A, k] += h
            dm[A, k] -= h
            Ep = energy_fn(dp)
            Em = energy_fn(dm)
            grad[A, k] = (Ep - Em) / (2.0 * h)
    return grad


def test_revd4_grad_ne2_matches_fd():
    # Neon dimer (neutral), small separation
    dtype = torch.float64
    device = torch.device('cpu')
    numbers = torch.tensor([10, 10], dtype=torch.long, device=device)
    positions = torch.tensor([[-1.7, 0.0, 0.0], [1.7, 0.0, 0.0]], dtype=dtype, device=device)
    q = torch.zeros(2, dtype=dtype, device=device)
    # Load D4 method with placeholders allowed if missing
    params = load_d4_parameters('parameters/dftd4parameters.toml')
    block = select_d4_params(params, method='d4', functional='default', variant='bj-eeq-atm', allow_placeholders=True)
    method = D4Method(
        s6=float(block.get('s6', 1.0)), s8=float(block.get('s8', 0.0)), s9=float(block.get('s9', 0.0)),
        a1=float(block.get('a1', 0.0)), a2=float(block.get('a2', 0.0)), alp=float(block.get('alp', 16.0)),
        damping=str(block.get('damping', 'bj')), mbd=str(block.get('mbd', 'none')),
    )
    # Reference data
    ref = load_d4_reference_toml('parameters/d4_reference.toml', device=device, dtype=dtype)
    # Provide CN ingredients if not included already
    if 'r_cov' not in ref or 'k_cn' not in ref:
        # Rough placeholders to enable CN path; this uses explicit exception for revD4 testing.
        ref['r_cov'] = torch.ones( max(int(numbers.max().item())+1, 11), dtype=dtype, device=device)
        ref['k_cn'] = 1.0
    # Energy and analytic gradient
    E, g_ana = d4_energy_with_grad(numbers, positions, q, method, ref)

    def e_fun(pos):
        E2, g = d4_energy_with_grad(numbers, pos, q, method, ref)
        return E2

    g_fd = _finite_diff_grad(e_fun, positions, h=2e-3)
    assert torch.allclose(g_ana, g_fd, atol=2e-4, rtol=1e-4), f"revd4 grad mismatch\nana={g_ana}\nfd={g_fd}\nE={E}"

