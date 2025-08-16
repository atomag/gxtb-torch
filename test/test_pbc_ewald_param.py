import torch
import pytest

import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
_src = _root / 'src'
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from gxtb.pbc.ewald import ewald_grad_hess_1over_r


@pytest.mark.parametrize('eta', [0.25, 0.35, 0.50])
def test_ewald_convergence_eta_sweep(eta):
    # Fixed displacement in a simple cubic cell
    cell = torch.tensor([[10.0, 0.0, 0.0],
                         [0.0, 10.0, 0.0],
                         [0.0, 0.0, 10.0]], dtype=torch.float64)
    r = torch.tensor([1.1, -0.7, 0.3], dtype=torch.float64)
    # Increasing cutoffs should reduce successive differences
    params = [(6.0, 6.0), (8.0, 8.0), (10.0, 10.0)]
    grads = []
    hess = []
    for rc, gc in params:
        g, H = ewald_grad_hess_1over_r(r, cell, eta=eta, r_cut=rc, g_cut=gc)
        grads.append(g)
        hess.append(H)
    # Differences between successive cutoffs should be small by (8,8)->(10,10)
    d_grad = torch.linalg.vector_norm(grads[2] - grads[1]).item()
    d_hess = torch.linalg.vector_norm((hess[2] - hess[1]).reshape(-1)).item()
    # Absolute tolerance for refinement step; loose enough to allow eta=0.25
    assert d_grad <= 5e-4
    assert d_hess <= 5e-4
