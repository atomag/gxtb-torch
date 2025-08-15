import torch

from gxtb.params.loader import load_basisq
from gxtb.basis.qvszp import build_atom_basis
from gxtb.hamiltonian.ofx import build_lambda0_ao_from_element, build_ao_maps


def test_ofx_builder_assigns_pairs_correctly_for_CH():
    dtype = torch.float64
    device = torch.device("cpu")
    # CH: Carbon (Z=6), Hydrogen (Z=1)
    numbers = torch.tensor([6, 1], dtype=torch.int64, device=device)
    bq = load_basisq('parameters/basisq')
    basis = build_atom_basis(numbers, bq)
    ao_atom, ao_l, groups = build_ao_maps(numbers, basis)
    # Synthetic per-element constants: only for carbon Z=6
    # Set sp=0.4, pp_off=0.3, sd=0.2, pd=0.5; others 0
    Zmax = 93
    def vec(val):
        v = torch.zeros(Zmax+1, dtype=dtype)
        v[6] = val
        return v
    ofx_elem = {
        'sp': vec(0.4), 'pp_off': vec(0.3), 'sd': vec(0.2), 'pd': vec(0.5),
        'dd_off': vec(0.0), 'sf': vec(0.0), 'pf': vec(0.0), 'df': vec(0.0), 'ff_off': vec(0.0)
    }
    Lam0 = build_lambda0_ao_from_element(numbers, basis, ofx_elem, diag_rule='zero')
    # Check some entries: cross-l between carbon s and p should be 0.4
    # Find first s AO on carbon and first p AO on carbon
    c_s = groups.get((0,0), [])
    c_p = groups.get((0,1), [])
    if c_s and c_p:
        i = c_s[0]; j = c_p[0]
        assert abs(Lam0[i,j].item() - 0.4) < 1e-12
        assert abs(Lam0[j,i].item() - 0.4) < 1e-12
    # Within p shell off-diagonal should be 0.3
    if len(c_p) >= 2:
        i, j = c_p[0], c_p[1]
        assert abs(Lam0[i,j].item() - 0.3) < 1e-12
        assert abs(Lam0[j,i].item() - 0.3) < 1e-12
    # Hydrogen should have no entries (no ofx constants set for Z=1)
    for idxs in groups.values():
        for i in idxs:
            # any AO on hydrogen atom (A=1)
            if ao_atom[i].item() == 1:
                assert torch.allclose(Lam0[i], torch.zeros_like(Lam0[i]))

