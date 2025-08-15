import torch, pytest
try:
    from gxtb.params.loader import load_gxtb_params, load_basisq
    from gxtb.params.schema import load_schema, map_eht_params, map_cn_params
    from gxtb.basis.qvszp import build_atom_basis
    from gxtb.hamiltonian.eht import build_eht_hamiltonian, first_order_energy
    backend_ok = True
except Exception:
    backend_ok = False


@pytest.mark.skipif(not backend_ok, reason="dxtb backend or gxtb modules unavailable")
def test_eht_h_s_only():
    # Synthetic tiny system: H2 (assuming Z=1 present in params)
    numbers = torch.tensor([1,1], dtype=torch.int64)
    positions = torch.tensor([[0.0,0.0,0.0],[0.0,0.0,0.74]], dtype=torch.float64)  # ~ Bohr/Ang? assume consistent units
    gparams = load_gxtb_params('parameters/gxtb')
    basisq = load_basisq('parameters/basisq')
    basis = build_atom_basis(numbers, basisq)
    schema = load_schema('parameters/gxtb.schema.toml')
    cn_params = map_cn_params(gparams, schema)
    res = build_eht_hamiltonian(numbers, positions, basis, gparams, schema, cn_params['r_cov'], cn_params['k_cn'])
    # Basic symmetry
    assert torch.allclose(res.S_scaled, res.S_scaled.T, atol=1e-12)
    assert torch.allclose(res.H, res.H.T, atol=1e-12)
    # Diagonal equals eps
    assert torch.allclose(torch.diag(res.H), res.eps)
    # Simple density: occupy first AO fully (toy)
    P = torch.zeros_like(res.H)
    P[0,0] = 2.0
    e1 = first_order_energy(P, res.H)
    assert e1.item() == res.eps[0].item()*2.0


@pytest.mark.skipif(not backend_ok, reason="dxtb backend or gxtb modules unavailable")
def test_eht_offdiag_nonzero_with_eps_mapping():
    """Ensure that using schema 'eps_*' onsite keys populates H off-diagonals (no collapse to zero).

    Regression for onsite mapping mismatch (eps_* vs h_*) that zeroed avg_eps and off-diagonals, collapsing E^{(1)}.
    """
    # Use heteronuclear pair (H-O) to avoid accidental zero avg_eps on homonuclear H2 if eps_s(H)=0 in params
    numbers = torch.tensor([1, 8], dtype=torch.int64)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    gparams = load_gxtb_params('parameters/gxtb')
    basisq = load_basisq('parameters/basisq')
    basis = build_atom_basis(numbers, basisq)
    schema = load_schema('parameters/gxtb.schema.toml')
    cn_params = map_cn_params(gparams, schema)
    res = build_eht_hamiltonian(numbers, positions, basis, gparams, schema, cn_params['r_cov'], cn_params['k_cn'])
    H = res.H
    offdiag_norm = (H - torch.diag(torch.diag(H))).abs().sum().item()
    # Expect non-zero coupling between atoms
    assert offdiag_norm > 0.0
