import torch, pytest

from gxtb.params.loader import load_gxtb_params, load_basisq
from gxtb.params.schema import load_schema, map_hubbard_params, map_cn_params
from gxtb.basis.qvszp import build_atom_basis
from gxtb.hamiltonian.scf_adapter import build_eht_core, make_core_builder
from gxtb.scf import scf

from gxtb.hamiltonian.first_order import FirstOrderParams as _FOParams


backend_available = True
try:
    import dxtb  # type: ignore
except Exception:
    backend_available = False


def _first_order_pack(numbers: torch.Tensor, mu_s_by_Z: dict[int, float]):
    Zmax = int(numbers.max().item())
    mu10 = torch.zeros((Zmax + 1, 4), dtype=torch.float64)
    kCN = torch.zeros((Zmax + 1,), dtype=torch.float64)
    for z, val in mu_s_by_Z.items():
        if z <= Zmax:
            mu10[z, 0] = float(val)
    return {
        'mu10': mu10,
        'kCN': kCN,
        'kdis': 0.2,
        'kx': 1.5,
        'ks': 0.1,
    }


@pytest.mark.skipif(not backend_available, reason="dxtb backend unavailable for overlap integrals")
def test_scf_first_order_h2_cpu():
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    hub = map_hubbard_params(gparams, schema)
    cn_map = map_cn_params(gparams, schema)
    basisq = load_basisq('parameters/basisq')
    numbers = torch.tensor([1, 1], dtype=torch.long)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], dtype=torch.float64)
    basis = build_atom_basis(numbers, basisq)
    core = build_eht_core(numbers, positions, basis, gparams, schema)
    builder = make_core_builder(basis, gparams, schema)
    fo = _first_order_pack(numbers, {1: 0.3})
    res = scf(
        numbers, positions, basis, builder, core['S'], hub, core['ao_atoms'], nelec=2,
        max_iter=10, first_order=True, first_order_params=fo,
        dynamic_overlap=False,
        qvszp_params={'r_cov': cn_map['r_cov'], 'k_cn': float(cn_map['k_cn'])},
    )
    assert res.H.shape == core['H0'].shape
    assert res.E_First is not None and torch.isfinite(res.E_First)


@pytest.mark.skipif(not backend_available, reason="dxtb backend unavailable for overlap integrals")
def test_scf_first_order_h2o_cpu():
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    hub = map_hubbard_params(gparams, schema)
    cn_map = map_cn_params(gparams, schema)
    basisq = load_basisq('parameters/basisq')
    numbers = torch.tensor([8, 1, 1], dtype=torch.long)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2399872, 0.927297, 0.0]], dtype=torch.float64)
    basis = build_atom_basis(numbers, basisq)
    core = build_eht_core(numbers, positions, basis, gparams, schema)
    builder = make_core_builder(basis, gparams, schema)
    fo = _first_order_pack(numbers, {1: 0.2, 8: 0.1})
    res = scf(
        numbers, positions, basis, builder, core['S'], hub, core['ao_atoms'], nelec=10,
        max_iter=12, first_order=True, first_order_params=fo,
        dynamic_overlap=False,
        qvszp_params={'r_cov': cn_map['r_cov'], 'k_cn': float(cn_map['k_cn'])},
    )
    assert res.H.shape[0] == basis.nao
    assert res.E_First is not None and torch.isfinite(res.E_First)


@pytest.mark.skipif(not backend_available, reason="dxtb backend unavailable for overlap integrals")
def test_scf_first_order_hplus_cpu():
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    hub = map_hubbard_params(gparams, schema)
    cn_map = map_cn_params(gparams, schema)
    basisq = load_basisq('parameters/basisq')
    numbers = torch.tensor([1], dtype=torch.long)
    positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    basis = build_atom_basis(numbers, basisq)
    core = build_eht_core(numbers, positions, basis, gparams, schema)
    builder = make_core_builder(basis, gparams, schema)
    fo = _first_order_pack(numbers, {1: 0.3})
    res = scf(
        numbers, positions, basis, builder, core['S'], hub, core['ao_atoms'], nelec=0,
        max_iter=6, first_order=True, first_order_params=fo,
        dynamic_overlap=False,
        qvszp_params={'r_cov': cn_map['r_cov'], 'k_cn': float(cn_map['k_cn'])},
    )
    assert res.H.shape == core['H0'].shape
    # With no electrons, E_First may be zero; still must be a finite scalar
    if res.E_First is not None:
        assert torch.isfinite(res.E_First)


@pytest.mark.skipif(not backend_available, reason="dxtb backend unavailable for overlap integrals")
def test_scf_first_order_schema_mapping_cpu():
    """If first_order_params is None, scf maps via provided gparams/schema."""
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    hub = map_hubbard_params(gparams, schema)
    cn_map = map_cn_params(gparams, schema)
    basisq = load_basisq('parameters/basisq')
    numbers = torch.tensor([1, 1], dtype=torch.long)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=torch.float64)
    basis = build_atom_basis(numbers, basisq)
    core = build_eht_core(numbers, positions, basis, gparams, schema)
    builder = make_core_builder(basis, gparams, schema)
    # Do not pass first_order_params; provide gparams/schema for internal mapping.
    res = scf(
        numbers, positions, basis, builder, core['S'], hub, core['ao_atoms'], nelec=2,
        max_iter=8, first_order=True, first_order_params=None,
        dynamic_overlap=False,
        qvszp_params={'r_cov': cn_map['r_cov'], 'k_cn': float(cn_map['k_cn'])},
        gparams=gparams, schema=schema,
    )
    # Path must run and produce a finite scalar (may be zero depending on mapping)
    assert res.E_First is None or torch.isfinite(res.E_First)
