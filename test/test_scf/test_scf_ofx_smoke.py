import torch

from gxtb.params.loader import load_gxtb_params, load_basisq
from gxtb.params.schema import load_schema, map_hubbard_params
from gxtb.hamiltonian.scf_adapter import build_eht_core, make_core_builder
from gxtb.hamiltonian.ofx import build_lambda0_ao_from_element
from gxtb.scf import scf


def _atom_basis_from_element(numbers, basisq):
    # Build AtomBasis-like from basisq for given numbers
    from types import SimpleNamespace
    shells = []
    ao_counts = []
    ao_offsets = []
    offset = 0
    l_ao = {'s': 1, 'p': 3, 'd': 5, 'f': 7}
    for ia, z in enumerate(numbers.tolist()):
        eb = basisq.elements[int(z)]
        for l, sh_group in eb.shells.items():
            for sh in sh_group:
                n_sph = l_ao.get(l, 1)
                shells.append(SimpleNamespace(atom_index=ia, element=int(z), l=l, nprims=sh.nprims,
                                              primitives=[(p.alpha, p.c1, p.c2) for p in sh.primitives]))
                ao_counts.append(n_sph)
                ao_offsets.append(offset)
                offset += n_sph
    nao = sum(ao_counts)
    return SimpleNamespace(shells=shells, ao_counts=ao_counts, ao_offsets=ao_offsets, nao=nao)


def test_scf_with_ofx_runs_smoke():
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    hub = map_hubbard_params(gparams, schema)
    basisq = load_basisq('parameters/basisq')
    # Carbon atom
    numbers = torch.tensor([6], dtype=torch.long)
    positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    atom_basis = _atom_basis_from_element(numbers, basisq)
    core = build_eht_core(numbers, positions, atom_basis, gparams, schema)
    builder = make_core_builder(atom_basis, gparams, schema)
    S = core['S']
    # Build synthetic per-element OFX constants for carbon
    Zmax = 93
    def vec(val):
        v = torch.zeros(Zmax+1, dtype=torch.float64)
        v[6] = val
        return v
    ofx_elem = {
        'sp': vec(0.4), 'pp_off': vec(0.3), 'sd': vec(0.2), 'pd': vec(0.1),
        'dd_off': vec(0.0), 'sf': vec(0.0), 'pf': vec(0.0), 'df': vec(0.0), 'ff_off': vec(0.0)
    }
    from gxtb.hamiltonian.ofx import build_ao_maps
    Lam0 = build_lambda0_ao_from_element(numbers, atom_basis, ofx_elem)
    res = scf(numbers, positions, atom_basis, builder, S, hub, core['ao_atoms'], nelec=4,
              max_iter=5, ofx=True, ofx_params={'alpha': 0.6, 'Lambda0_ao': Lam0})
    assert res.converged or res.n_iter <= 5
    assert res.E_OFX is not None
    assert isinstance(res.E_OFX.item(), float)

