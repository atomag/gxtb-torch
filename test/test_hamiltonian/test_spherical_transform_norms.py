import torch
from gxtb.basis.md_overlap import _spherical_transform, _cart_list

def _row_norms(mat):
    return torch.sqrt((mat.double()**2).sum(dim=1))

def test_spdfg_transform_row_norms():
    for l in [0,1,2,3,4]:
        T = _spherical_transform(l, torch.float64, torch.device('cpu'))
        norms = _row_norms(T)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-8)
