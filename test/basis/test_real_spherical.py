import torch
import pytest


def _metric_for_shell(l: int, alpha: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    from gxtb.basis.md_overlap import _overlap_cart_block
    R0 = alpha.new_zeros(3)
    return _overlap_cart_block(l, l, alpha, c, alpha, c, R0)


@pytest.mark.parametrize("l", [0, 1, 2, 3, 4])
def test_real_spherical_transform_metric_identity(l: int):
    # Random but deterministic alpha,c for the contracted shell
    torch.manual_seed(42 + l)
    nprim = 2 if l < 3 else 3
    alpha = torch.rand(nprim, dtype=torch.float64) * 1.5 + 0.2
    c = torch.rand(nprim, dtype=torch.float64)

    from gxtb.basis.md_overlap import _metric_orthonormal_sph_transform
    Scc = _metric_for_shell(l, alpha, c)
    # Compute transform T such that T Scc T^T = I
    T = _metric_orthonormal_sph_transform(l, Scc)
    G = T @ Scc @ T.T
    I = torch.eye(G.shape[0], dtype=G.dtype)
    err = torch.max(torch.abs(G - I)).item()
    assert err < 1e-10, f"metric orthonormality failed for l={l} with max error {err}"

