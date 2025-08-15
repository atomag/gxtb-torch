import torch
import pytest

from gxtb.classical.dispersion import build_revd4_c6_c8, RevD4DataMissing


def _numbers_positions_water():
    numbers = torch.tensor([8, 1, 1], dtype=torch.int64)
    positions = torch.tensor(
        [
            [0.000000, 0.000000, 0.000000],
            [0.958, 0.000000, 0.000000],
            [-0.239, 0.927, 0.000000],
        ],
        dtype=torch.float64,
    )
    return numbers, positions


def test_build_revd4_c6_c8_requires_zeta_params():
    numbers, _ = _numbers_positions_water()
    cn = torch.zeros(numbers.shape[0], dtype=torch.float64)
    q = torch.zeros_like(cn)
    Zmax = int(numbers.max().item())
    beta2 = torch.ones(Zmax + 1, Zmax + 1, dtype=torch.float64)
    with pytest.raises(RevD4DataMissing):
        build_revd4_c6_c8(numbers, cn, q, beta2, zeta_params=None)


def test_build_revd4_requires_reference_dataset():
    numbers, positions = _numbers_positions_water()
    device = positions.device
    dtype = positions.dtype
    nat = numbers.shape[0]
    cn = torch.zeros(nat, dtype=dtype, device=device)
    q = torch.zeros(nat, dtype=dtype, device=device)
    Zmax = int(numbers.max().item())
    beta2 = torch.ones(Zmax + 1, Zmax + 1, dtype=dtype, device=device)
    W = 23
    zeta_params = {
        "A": torch.ones(Zmax + 1, W, dtype=dtype, device=device),
        "B": torch.zeros(Zmax + 1, W, dtype=dtype, device=device),
        "C": torch.zeros(Zmax + 1, W, dtype=dtype, device=device),
        "D": torch.zeros(Zmax + 1, W, dtype=dtype, device=device),
    }
    with pytest.raises(RevD4DataMissing):
        build_revd4_c6_c8(numbers, cn, q, beta2, zeta_params, ref=None)
