import torch, pytest
from gxtb.basis.qvszp import AtomBasis, ShellDef
from gxtb.hamiltonian.second_order_tb import (
    build_shell_second_order_params,
    compute_shell_charges,
    compute_gamma2_shell,
)

class DummyBasis(AtomBasis):
    pass

def build_dummy_basis(numbers):
    """Minimal basis with mandatory valence shells.

    Adds an s shell for every atom and a p shell for atoms whose ground-state
    valence includes p (roughly Z in B–Ne range and beyond). This satisfies the
    strict reference population builder which raises if a required valence
    shell is absent.
    """
    shells=[]; ao_counts=[]; ao_offsets=[]; nao=0
    p_elements = set(range(5, 11))  # B, C, N, O, F, Ne (s/p valence window needed for tests)
    deg = {'s':1,'p':3}
    for ia,z in enumerate(numbers.tolist()):
        z = int(z)
        # s shell
        shells.append(ShellDef(atom_index=ia, element=z, l='s', nprims=1, primitives=((1.0,1.0,0.0),)))
        ao_counts.append(deg['s']); ao_offsets.append(nao); nao += deg['s']
        # optional p shell if element requires p valence description
        if z in p_elements:
            shells.append(ShellDef(atom_index=ia, element=z, l='p', nprims=1, primitives=((1.0,1.0,0.0),)))
            ao_counts.append(deg['p']); ao_offsets.append(nao); nao += deg['p']
    return DummyBasis(shells=shells, ao_counts=ao_counts, ao_offsets=ao_offsets, nao=nao)

def test_shell_charges_and_gamma():
    numbers = torch.tensor([1,8], dtype=torch.int64)  # H O
    positions = torch.tensor([[0.0,0.0,0.0],[0.95,0.0,0.0]], dtype=torch.float64)
    basis = build_dummy_basis(numbers)
    nao = basis.nao
    # density matrix: occupy first AO fully (2e) simple test
    P = torch.zeros((nao,nao), dtype=torch.float64)
    P[0,0] = 2.0
    S = torch.eye(nao, dtype=torch.float64)
    from gxtb.hamiltonian.second_order_tb import compute_reference_shell_populations
    ref = compute_reference_shell_populations(numbers, basis)
    q_shell = compute_shell_charges(P, S, basis, ref)
    total_pop = P.diag().sum()
    expected_sum = total_pop - ref.sum()
    assert torch.allclose(q_shell.sum(), expected_sum)
    # Build params broadcasting eta
    max_z = int(numbers.max().item())
    eta_like = torch.zeros(max_z+1, dtype=torch.float64)
    eta_like[1]=1.0; eta_like[8]=1.2
    params = build_shell_second_order_params(max_z, eta_like, kexp=0.1)
    cn = torch.zeros(len(numbers), dtype=torch.float64)
    gamma_shell = compute_gamma2_shell(numbers, positions, basis, params, cn)
    # Symmetry
    assert torch.allclose(gamma_shell, gamma_shell.T)
    # On-site equals U (broadcast eta)
    assert pytest.approx(gamma_shell[0,0].item(), rel=1e-12) == eta_like[1]
    # Distance decay: larger R -> smaller gamma (compare with no damping by reducing R artificially)
    # Create second gamma with increased distance
    positions_far = positions.clone(); positions_far[1,0] = 3.0
    gamma_far = compute_gamma2_shell(numbers, positions_far, basis, params, cn)
    assert gamma_far[0,1] < gamma_shell[0,1]

def test_cn_scaling_increases_U():
    numbers = torch.tensor([6,6], dtype=torch.int64)
    positions = torch.tensor([[0.0,0.0,0.0],[1.4,0.0,0.0]], dtype=torch.float64)
    basis = build_dummy_basis(numbers)
    max_z = int(numbers.max().item())
    eta_like = torch.zeros(max_z+1, dtype=torch.float64)
    eta_like[6] = 1.0
    # No CN scaling
    params0 = build_shell_second_order_params(max_z, eta_like, kexp=0.0)
    cn_zero = torch.zeros(len(numbers), dtype=torch.float64)
    g0 = compute_gamma2_shell(numbers, positions, basis, params0, cn_zero)
    # Add kU scaling
    kU = torch.zeros(max_z+1, dtype=torch.float64)
    kU[6] = 0.5
    params_scaled = build_shell_second_order_params(max_z, eta_like, kexp=0.0, kU=kU)
    # Large CN to amplify
    cn_large = torch.full((len(numbers),), 4.0, dtype=torch.float64)
    g_scaled = compute_gamma2_shell(numbers, positions, basis, params_scaled, cn_large)
    # On-site element should increase (U larger) means diagonal gamma increases (Ui)
    assert g_scaled[0,0] > g0[0,0]
    # Off-diagonal denominator includes 1/U terms so increase U lowers 1/U => lowers denom slightly => gamma increases or stays similar
    assert g_scaled[0,1] >= g0[0,1]
