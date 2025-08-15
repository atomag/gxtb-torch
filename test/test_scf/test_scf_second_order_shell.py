import torch, pytest
from gxtb.basis.qvszp import AtomBasis, ShellDef
from gxtb.scf import scf
from gxtb.hamiltonian.second_order_tb import build_shell_second_order_params

# Minimal core builder returning zero baseline Hamiltonian
class DummyBasis(AtomBasis):
    pass

def build_dummy_basis(numbers):
    """Minimal basis with s for all atoms and p for p-valence elements (B–Ne).

    Ensures reference shell population generator finds required valence shells.
    """
    shells=[]; ao_counts=[]; ao_offsets=[]; nao=0
    p_elements = set(range(5, 11))
    deg = {'s':1,'p':3}
    for ia,z in enumerate(numbers.tolist()):
        z = int(z)
        shells.append(ShellDef(atom_index=ia, element=z, l='s', nprims=1, primitives=((1.0,1.0,0.0),)))
        ao_counts.append(deg['s']); ao_offsets.append(nao); nao += deg['s']
        if z in p_elements:
            shells.append(ShellDef(atom_index=ia, element=z, l='p', nprims=1, primitives=((1.0,1.0,0.0),)))
            ao_counts.append(deg['p']); ao_offsets.append(nao); nao += deg['p']
    return DummyBasis(shells=shells, ao_counts=ao_counts, ao_offsets=ao_offsets, nao=nao)

def dummy_core(numbers, positions, ctx):
    # Build zero core Hamiltonian sized to current basis (captured via closure in test)
    return {"H0": torch.zeros((dummy_core.nao, dummy_core.nao), dtype=positions.dtype, device=positions.device)}

def test_scf_shell_second_order_runs():
    numbers = torch.tensor([1,8], dtype=torch.int64)
    positions = torch.tensor([[0.0,0.0,0.0],[0.95,0.0,0.0]], dtype=torch.float64)
    basis = build_dummy_basis(numbers)
    nao = basis.nao
    dummy_core.nao = nao  # stash for core size
    # Simple overlap identity
    S = torch.eye(nao, dtype=torch.float64)
    # Map each AO to its parent atom index
    ao_atom_list=[]
    for ish, off in enumerate(basis.ao_offsets):
        n_ao = basis.ao_counts[ish]
        atom_idx = basis.shells[ish].atom_index
        ao_atom_list.extend([atom_idx]*n_ao)
    ao_atoms = torch.tensor(ao_atom_list, dtype=torch.long)
    nelec = 2  # occupy first shell
    # Hubbard gamma (simple positive constant per Z) required by SCF
    maxz = int(numbers.max().item())
    gamma = torch.zeros(maxz+1, dtype=torch.float64)
    gamma[1]=1.0; gamma[8]=1.2
    hubbard={"gamma": gamma, "gamma3": None}
    # Second-order shell params from gamma broadcast (eta-like placeholder)
    so_shell_params = build_shell_second_order_params(maxz, gamma, kexp=0.05)
    result = scf(numbers, positions, basis, dummy_core, S, hubbard, ao_atoms, nelec,
                 max_iter=10, tol=1e-6, second_order=True,
                 so_params={"shell_params": so_shell_params, "cn": torch.zeros(len(numbers))})
    assert result.H.shape == (nao,nao)
    # Second-order energy computed
    assert result.E2 is not None
