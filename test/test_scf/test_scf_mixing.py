import torch

from gxtb.params.loader import load_gxtb_params, load_basisq
from gxtb.params.schema import load_schema, map_hubbard_params
from gxtb.hamiltonian.scf_adapter import build_eht_core, make_core_builder
from gxtb.basis.qvszp import build_atom_basis
from gxtb.scf import scf


def test_scf_with_anderson_mixing_runs():
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    hub = map_hubbard_params(gparams, schema)
    bq = load_basisq('parameters/basisq')
    numbers = torch.tensor([6], dtype=torch.long)
    positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    basis = build_atom_basis(numbers, bq)
    core = build_eht_core(numbers, positions, basis, gparams, schema)
    builder = make_core_builder(basis, gparams, schema)
    S = core['S']
    res = scf(
        numbers, positions, basis, builder, S, hub, core['ao_atoms'], nelec=4,
        max_iter=5, mixing={'scheme': 'anderson', 'beta': 0.5, 'history': 3}
    )
    assert res.n_iter <= 5
    assert isinstance(res.converged, bool)
    assert len(res.E_history) >= 1


def test_scf_anderson_fallback_singular_history():
    # Force small steps to make ΔF ~ 0 and ensure robust fallback path
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    hub = map_hubbard_params(gparams, schema)
    bq = load_basisq('parameters/basisq')
    numbers = torch.tensor([1], dtype=torch.long)
    positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    basis = build_atom_basis(numbers, bq)
    core = build_eht_core(numbers, positions, basis, gparams, schema)
    builder = make_core_builder(basis, gparams, schema)
    S = core['S']
    # Use very small beta to reduce movement in q (ΔF columns nearly zero)
    res = scf(
        numbers, positions, basis, builder, S, hub, core['ao_atoms'], nelec=1,
        max_iter=3, mixing={'scheme': 'anderson', 'beta': 1e-6, 'history': 3}
    )
    assert res.n_iter <= 3
    assert len(res.E_history) >= 1
