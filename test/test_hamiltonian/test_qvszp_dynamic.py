import pytest
import torch

from gxtb.params.loader import load_basisq, load_gxtb_params
from gxtb.params.schema import load_schema
from gxtb.basis.qvszp import build_atom_basis, compute_effective_charge, build_dynamic_primitive_coeffs
from gxtb.hamiltonian.overlap_tb import build_overlap_dynamic, build_overlap


def test_dynamic_coeffs_change_overlap_with_qeff():
    # H–C diatomic; off-diagonal overlap will depend on contraction coefficients
    numbers = torch.tensor([6, 1], dtype=torch.long)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=torch.float64)
    bq = load_basisq('parameters/basisq')
    basis = build_atom_basis(numbers, bq)
    # Baseline static overlap (legacy c0+c1)
    S_static = build_overlap(numbers, positions, basis)
    # Build dynamic coeffs with q_eff = 0 and q_eff = 1 for comparison
    q_eff0 = torch.zeros(numbers.shape[0], dtype=torch.float64)
    q_eff1 = torch.ones(numbers.shape[0], dtype=torch.float64)
    coeffs0 = {i: c for i, c in enumerate(build_dynamic_primitive_coeffs(numbers, basis, q_eff0))}
    coeffs1 = {i: c for i, c in enumerate(build_dynamic_primitive_coeffs(numbers, basis, q_eff1))}
    S0 = build_overlap_dynamic(numbers, positions, basis, coeffs0)
    S1 = build_overlap_dynamic(numbers, positions, basis, coeffs1)
    # S_static should equal S1 when q_eff=1 (since static used c0+c1)
    assert torch.allclose(S_static, S1, atol=1e-12)
    # And differ from static when q_eff=0 on at least one atom
    assert not torch.allclose(S_static, S0)


@pytest.mark.xfail(strict=True, reason="q‑vSZP prefactor mapping missing from schema; guarded by theory Eq. 28")
def test_qvszp_prefactors_mapping_missing():
    from gxtb.params.schema import map_qvszp_prefactors
    g = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    _ = map_qvszp_prefactors(g, schema)
