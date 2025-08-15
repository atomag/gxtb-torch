import torch
from gxtb.basis.qvszp import AtomBasis, ShellDef
from gxtb.hamiltonian.overlap_tb import build_overlap

class DummyBasis(AtomBasis):
    pass

def build_basis(l_list):
    shells=[]; ao_counts=[]; ao_offsets=[]; nao=0
    for l in l_list:
        shells.append(ShellDef(atom_index=0, element=8, l=l, nprims=1, primitives=((1.0,1.0,0.0),)))
        if l=='s': n=1
        elif l=='p': n=3
        elif l=='d': n=5
        elif l=='f': n=7
        else: raise ValueError
        ao_counts.append(n); ao_offsets.append(nao); nao+=n
    return DummyBasis(shells=shells, ao_counts=ao_counts, ao_offsets=ao_offsets, nao=nao)

def test_internal_md_spdf():
    basis = build_basis(['s','p','d','f'])
    numbers=torch.tensor([8])
    positions=torch.zeros((1,3),dtype=torch.float64)
    S = build_overlap(numbers, positions, basis)
    assert S.shape[0]==basis.nao
    # Diagonal ones
    assert torch.allclose(torch.diag(S), torch.ones(basis.nao, dtype=S.dtype))
    # Symmetry
    assert torch.allclose(S, S.T)
    # Off-diagonal magnitude bounded
    assert (S.abs() <= 1.0 + 1e-8).all()
