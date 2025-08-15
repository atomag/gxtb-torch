import torch

from gxtb.basis.md_overlap import (
    _cart_list,
    _metric_orthonormal_sph_transform,
    _metric_transform_for_shell,
    _overlap_cart_block,
)


def _random_shell(li: int, nprim: int = 3):
    # Deterministic but nontrivial primitive sets
    alphas = torch.linspace(0.5, 1.5, nprim, dtype=torch.float64)
    coeffs = torch.linspace(0.8, 1.2, nprim, dtype=torch.float64)
    return alphas, coeffs


def test_metric_whitening_oncenter_spdf():
    # For l in s,p,d,f ensure T Scc T^T = I to ~1e-12
    for li in [0, 1, 2, 3]:
        alpha, c = _random_shell(li)
        Scc = _overlap_cart_block(li, li, alpha, c, alpha, c, torch.zeros(3, dtype=torch.float64))
        T = _metric_orthonormal_sph_transform(li, Scc)
        G = T @ Scc @ T.T
        eye = torch.eye(G.shape[0], dtype=G.dtype)
        assert torch.allclose(G, eye, atol=1e-10, rtol=0.0)


def test_overlap_shell_pair_uses_metric_transform():
    # For a same-center shell pair, the spherical overlap should be exactly identity
    from gxtb.basis.md_overlap import overlap_shell_pair

    for li in [0, 1, 2, 3, 4]:
        alpha, c = _random_shell(li)
        S = overlap_shell_pair(li, li, alpha, c, alpha, c, torch.zeros(3, dtype=torch.float64))
        eye = torch.eye(S.shape[0], dtype=S.dtype)
        assert torch.allclose(S, eye, atol=1e-10, rtol=0.0)

