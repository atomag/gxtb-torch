import torch
from gxtb.basis.qvszp import AtomBasis, ShellDef
from gxtb.hamiltonian.overlap_tb import build_overlap

class DummyBasis(AtomBasis):
    pass

def build_f_basis():
    # Single atom with s and f shells (primitive coefficients arbitrary for shape test)
    shells = [
        ShellDef(atom_index=0, element=8, l='s', nprims=1, primitives=((1.0,1.0,0.0),)),
        ShellDef(atom_index=0, element=8, l='f', nprims=1, primitives=((0.8,1.0,0.0),)),
    ]
    ao_counts = [1,7]
    ao_offsets = [0,1]
    return DummyBasis(shells=shells, ao_counts=ao_counts, ao_offsets=ao_offsets, nao=8)

def test_overlap_includes_f_block():
    numbers = torch.tensor([8])
    positions = torch.zeros((1,3), dtype=torch.float64)
    basis = build_f_basis()
    S = build_overlap(numbers, positions, basis)
    assert S.shape == (8,8)
    # Diagonal normalized to 1
    assert torch.allclose(torch.diag(S), torch.ones(8, dtype=S.dtype))
    # s-f off-diagonal couplings should be finite and <=1
    sf_block = S[0:1,1:8]
    assert torch.all(torch.isfinite(sf_block))
    assert (sf_block.abs() <= 1.0 + 1e-12).all()
