import torch, pytest
from gxtb.params.loader import load_gxtb_params, load_basisq
from gxtb.params.schema import load_schema, map_hubbard_params
from gxtb.hamiltonian.scf_adapter import build_eht_core, make_core_builder
from gxtb.scf import scf


backend_available = True
try:
    import dxtb  # type: ignore
except Exception:
    backend_available = False

@pytest.mark.skipif(not backend_available, reason="dxtb backend unavailable for overlap integrals")
def test_scf_runs_minimal():
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    hub = map_hubbard_params(gparams, schema)
    basisq = load_basisq('parameters/basisq')
    # Simple H2 system (Z=1) with two atoms
    numbers = torch.tensor([1,1], dtype=torch.long)
    positions = torch.tensor([[0.0,0.0,0.0],[0.74,0.0,0.0]], dtype=torch.float64)
    # Build minimal AtomBasis wrapper from basisq entry for H (Z=1)
    eb = basisq.elements[1]
    # Flatten shells
    from types import SimpleNamespace
    shells = []
    ao_counts = []
    ao_offsets = []
    offset = 0
    for l, sh_group in eb.shells.items():
        for sh in sh_group:
            ncart = {'s':1,'p':3,'d':5}.get(l,1)
            shells.append(SimpleNamespace(atom_index=0, element=1, l=l, nprims=sh.nprims, primitives=[(p.alpha,p.c1,p.c2) for p in sh.primitives]))
            ao_counts.append(ncart)
            ao_offsets.append(offset)
            offset += ncart
    # Duplicate shells for second atom with shifted atom_index
    shells2 = []
    for sh in shells:
        shells2.append(SimpleNamespace(atom_index=1, element=1, l=sh.l, nprims=sh.nprims, primitives=sh.primitives))
    shells.extend(shells2)
    ao_counts.extend(ao_counts)
    ao_offsets.extend([o+offset for o in ao_offsets])
    nao = sum(ao_counts)
    atom_basis = SimpleNamespace(shells=shells, ao_counts=ao_counts, ao_offsets=ao_offsets, nao=nao)
    core = build_eht_core(numbers, positions, atom_basis, gparams, schema)
    builder = make_core_builder(atom_basis, gparams, schema)
    res = scf(numbers, positions, atom_basis, builder, core['S'], hub, core['ao_atoms'], nelec=2, max_iter=5)
    assert res.H.shape == core['H0'].shape
    assert res.P.shape == core['H0'].shape
    assert res.q.shape[0] == numbers.shape[0]
    assert res.E_history is not None
    assert len(res.E_history) >= 1
