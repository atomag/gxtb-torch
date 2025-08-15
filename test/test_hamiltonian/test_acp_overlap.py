import torch

from gxtb.params.loader import load_basisq
from gxtb.basis.qvszp import build_atom_basis
from gxtb.hamiltonian.acp import build_acp_overlap
from gxtb.params.schema import load_schema, map_cn_params
from pathlib import Path


def _dummy_acp_tables(Zmax=20, fill=0.3, xi=0.7):
    c0 = torch.zeros(Zmax+1, 4, dtype=torch.float64)
    xit = torch.zeros_like(c0)
    # set same value for all shells; tests will pick their Z
    c0[:, :] = fill
    xit[:, :] = xi
    cn_avg = torch.ones(Zmax+1, dtype=torch.float64)
    return c0, xit, cn_avg


def test_acp_overlap_shape_and_zero(monkeypatch):
    # Single carbon atom at origin
    numbers = torch.tensor([6], dtype=torch.long)
    positions = torch.zeros((1,3), dtype=torch.float64)
    bq = load_basisq('parameters/basisq')
    basis = build_atom_basis(numbers, bq)
    # CN params from schema
    schema = load_schema(Path('parameters') / 'gxtb.schema.toml')
    from gxtb.params.loader import load_gxtb_params
    cnmap = map_cn_params(load_gxtb_params('parameters/gxtb'), schema)
    r_cov = cnmap['r_cov']
    k_cn = float(cnmap['k_cn'])
    # Non-zero c0 -> non-zero S_acp
    c0, xi, cn_avg = _dummy_acp_tables(Zmax=int(numbers.max().item()))
    S_acp = build_acp_overlap(numbers, positions, basis, c0=c0, xi=xi, k_acp_cn=0.0, cn_avg=cn_avg, r_cov=r_cov, k_cn=k_cn, l_list=("s","p","d"))
    nao = basis.nao
    naux = 1 + 3 + 5
    assert S_acp.shape == (nao, naux)
    assert torch.count_nonzero(S_acp) > 0
    # Zero c0 -> zero S_acp
    c0z = torch.zeros_like(c0)
    S0 = build_acp_overlap(numbers, positions, basis, c0=c0z, xi=xi, k_acp_cn=0.0, cn_avg=cn_avg, r_cov=r_cov, k_cn=k_cn, l_list=("s","p","d"))
    assert torch.allclose(S0, torch.zeros_like(S0))


def test_acp_overlap_translation_invariance():
    # Two-atom system; translating all positions leaves AO-ACP displacements invariant
    numbers = torch.tensor([6, 1], dtype=torch.long)
    positions = torch.tensor([[0.0,0.0,0.0],[1.1,0.0,0.0]], dtype=torch.float64)
    bq = load_basisq('parameters/basisq')
    basis = build_atom_basis(numbers, bq)
    from gxtb.params.schema import load_schema, map_cn_params
    from gxtb.params.loader import load_gxtb_params
    schema = load_schema('parameters/gxtb.schema.toml')
    cnmap = map_cn_params(load_gxtb_params('parameters/gxtb'), schema)
    r_cov = cnmap['r_cov']
    k_cn = float(cnmap['k_cn'])
    c0, xi, cn_avg = _dummy_acp_tables(Zmax=int(numbers.max().item()))
    S1 = build_acp_overlap(numbers, positions, basis, c0=c0, xi=xi, k_acp_cn=0.0, cn_avg=cn_avg, r_cov=r_cov, k_cn=k_cn)
    shift = torch.tensor([0.3, -0.2, 0.5], dtype=positions.dtype)
    S2 = build_acp_overlap(numbers, positions + shift, basis, c0=c0, xi=xi, k_acp_cn=0.0, cn_avg=cn_avg, r_cov=r_cov, k_cn=k_cn)
    assert torch.allclose(S1, S2, atol=1e-12)

