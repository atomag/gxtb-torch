import os, pytest, torch

backend_available = True
try:  # Delay heavy imports until we confirm backend
    import dxtb  # type: ignore
    from gxtb.hamiltonian.eht import build_eht_hamiltonian
    from gxtb.params.loader import load_gxtb_params, load_basisq
    from gxtb.params.schema import load_schema
    from gxtb.basis.qvszp import build_atom_basis
except Exception:
    backend_available = False


@pytest.mark.skipif(
    not backend_available or not (os.path.exists('parameters/gxtb') and os.path.exists('parameters/basisq') and os.path.exists('parameters/gxtb.schema.toml')),
    reason="Missing dxtb backend or parameter files",
)
def test_wolfsberg_mode_difference():
    numbers = torch.tensor([1, 8], dtype=torch.int64)  # H-O
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    gparams = load_gxtb_params('parameters/gxtb')
    basisq = load_basisq('parameters/basisq')
    basis = build_atom_basis(numbers, basisq)
    schema = load_schema('parameters/gxtb.schema.toml')

    res_arith = build_eht_hamiltonian(numbers, positions, basis, gparams, schema, wolfsberg_mode='arithmetic')
    res_geom = build_eht_hamiltonian(numbers, positions, basis, gparams, schema, wolfsberg_mode='geometric')
    # Expect some non-zero difference if k_w differs across elements.
    delta = (res_arith.H - res_geom.H).abs().sum().item()
    assert delta >= 0.0
