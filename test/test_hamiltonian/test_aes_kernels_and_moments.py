import torch

from gxtb.params.loader import load_basisq
from gxtb.basis.qvszp import build_atom_basis
from gxtb.hamiltonian.moments_builder import build_moment_matrices
from gxtb.hamiltonian.aes import AESParams, compute_multipole_moments, compute_atomic_moments, _pairwise_kernels, aes_energy_and_fock, _third_derivative_tensor, _fourth_derivative_tensor


def _simple_basis(numbers):
    # Load provided q-vSZP basis
    import pathlib
    param_dir = pathlib.Path(__file__).resolve().parents[2] / 'parameters'
    bq = load_basisq(param_dir / 'basisq')
    return build_atom_basis(numbers, bq)


def test_pairwise_kernel_asymptotics_and_damping():
    numbers = torch.tensor([1, 1], dtype=torch.int64)
    dtype = torch.float64
    # Positions at R1 and R2
    R1, R2 = 2.0, 4.0
    pos1 = torch.tensor([[0.0, 0.0, 0.0], [R1, 0.0, 0.0]], dtype=dtype)
    pos2 = torch.tensor([[0.0, 0.0, 0.0], [R2, 0.0, 0.0]], dtype=dtype)
    # mrad zero => fdmp ≈ 1
    mrad = torch.zeros(2, dtype=dtype)
    numbers = torch.tensor([1,1], dtype=torch.long)
    rij1, f3_1, Hess1, mask = _pairwise_kernels(numbers, pos1, mrad, dmp3=6.0, dmp5=6.0)
    rij2, f3_2, Hess2, _ = _pairwise_kernels(numbers, pos2, mrad, dmp3=6.0, dmp5=6.0)
    i, j = 0, 1
    ratio_f3 = (f3_1[i, j] / f3_2[i, j]).item()
    assert abs(ratio_f3 - (R2 / R1) ** 3) < 1e-6
    # Hessian norm ratio ~ (R2/R1)^3 (since ∇∇(1/R) ~ R^{-3})
    h1 = Hess1[i, j]
    h2 = Hess2[i, j]
    norm1 = torch.linalg.norm(h1)
    norm2 = torch.linalg.norm(h2)
    ratio_h = (norm1 / norm2).item()
    assert abs(ratio_h - (R2 / R1) ** 3) < 1e-5
    # Damping reduces magnitudes when mrad>0
    mrad_damped = torch.full((2,), 2.0, dtype=dtype)
    _, f3_d, Hess_d, _ = _pairwise_kernels(numbers, pos1, mrad_damped, dmp3=6.0, dmp5=6.0)
    assert f3_d[i, j] < f3_1[i, j]
    assert torch.linalg.norm(Hess_d[i, j]) < torch.linalg.norm(h1)


def test_si_damping_eq117_behavior():
    # SI damping f_n = 0.5 k_n (1 - erf(-k_s (R - R0))) with R0 from rcov
    dtype = torch.float64
    numbers = torch.tensor([1, 1], dtype=torch.long)
    R1, R2 = 0.5, 5.0
    pos1 = torch.tensor([[0.0, 0.0, 0.0], [R1, 0.0, 0.0]], dtype=dtype)
    pos2 = torch.tensor([[0.0, 0.0, 0.0], [R2, 0.0, 0.0]], dtype=dtype)
    mrad = torch.zeros(2, dtype=dtype)
    si = {'si_k3': 1.0, 'si_ks3': 2.0, 'si_k5': 1.0, 'si_ks5': 2.0, 'si_R0_mode': 'rcov', 'si_R0_scale': 1.0}
    r_cov = torch.zeros(10, dtype=dtype)
    r_cov[1] = 0.5  # H
    rij1, f3_1, _, _ = _pairwise_kernels(numbers, pos1, mrad, dmp3=6.0, dmp5=6.0, si_params=si, r_cov=r_cov)
    rij2, f3_2, _, _ = _pairwise_kernels(numbers, pos2, mrad, dmp3=6.0, dmp5=6.0, si_params=si, r_cov=r_cov)
    # Recover fdmp3 by dividing by R^-3
    invR1 = 1.0 / R1
    invR2 = 1.0 / R2
    g3_1 = invR1 ** 3
    g3_2 = invR2 ** 3
    fd1 = (f3_1[0,1] / g3_1).item()
    fd2 = (f3_2[0,1] / g3_2).item()
    # fd increases towards k3=1 as R grows beyond R0=1.0
    assert fd2 > fd1
    assert abs(fd2 - 1.0) < 1e-6


def test_t3_t4_scaling():
    # Verify T3/T4 scaling for r aligned with x-axis
    dtype = torch.float64
    R1, R2 = 1.0, 2.0
    rij1 = torch.tensor([[[0.0,0.0,0.0],[R1,0.0,0.0]],[[ -R1,0.0,0.0],[0.0,0.0,0.0]]], dtype=dtype)
    rij2 = torch.tensor([[[0.0,0.0,0.0],[R2,0.0,0.0]],[[ -R2,0.0,0.0],[0.0,0.0,0.0]]], dtype=dtype)
    invR1 = torch.zeros((2,2), dtype=dtype); invR1[0,1] = invR1[1,0] = 1.0/R1
    invR2 = torch.zeros((2,2), dtype=dtype); invR2[0,1] = invR2[1,0] = 1.0/R2
    fd = torch.ones((2,2), dtype=dtype)
    T3_1 = _third_derivative_tensor(rij1, invR1, fd)
    T3_2 = _third_derivative_tensor(rij2, invR2, fd)
    T4_1 = _fourth_derivative_tensor(rij1, invR1, fd)
    T4_2 = _fourth_derivative_tensor(rij2, invR2, fd)
    # take norms over off-diagonal block (0,1)
    n31 = torch.linalg.norm(T3_1[0,1])
    n32 = torch.linalg.norm(T3_2[0,1])
    n41 = torch.linalg.norm(T4_1[0,1])
    n42 = torch.linalg.norm(T4_2[0,1])
    ratio3 = (n31 / n32).item()
    ratio4 = (n41 / n42).item()
    assert abs(ratio3 - (R2 / R1) ** 4) < 1e-5
    assert abs(ratio4 - (R2 / R1) ** 5) < 1e-5


def test_atomic_moment_partition_consistency_and_energy_decay():
    numbers = torch.tensor([1, 1], dtype=torch.int64)
    dtype = torch.float64
    # Build simple basis and AO moments
    basis = _simple_basis(numbers)
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=dtype)
    S, D, Q = build_moment_matrices(numbers.to(dtype=dtype), pos, basis)
    # Simple symmetric density matrix (not physical, but deterministic)
    nao = basis.nao
    P = torch.eye(nao, dtype=dtype)
    # Global vs atomic moment sums
    glob = compute_multipole_moments(P, S, D, Q)
    # Build ao->atom map
    ao_atoms = []
    for ish, off in enumerate(basis.ao_offsets):
        ao_atoms.extend([basis.shells[ish].atom_index] * basis.ao_counts[ish])
    atoms = compute_atomic_moments(P, S, D, Q, torch.tensor(ao_atoms, dtype=torch.long))
    assert torch.isclose(atoms['q'].sum(), glob['S'])
    mu_sum = atoms['mu'].sum(dim=0)
    assert torch.allclose(mu_sum, torch.tensor([glob['Dx'], glob['Dy'], glob['Dz']], dtype=dtype))
    Qsum = atoms['Q'].sum(dim=0)
    comps = [glob['Qxx'], glob['Qxy'], glob['Qxz'], glob['Qyy'], glob['Qyz'], glob['Qzz']]
    target = torch.tensor([[comps[0], comps[1], comps[2]], [comps[1], comps[3], comps[4]], [comps[2], comps[4], comps[5]]], dtype=dtype)
    assert torch.allclose(Qsum, target)
    # AES energy decays with distance
    params = AESParams(dmp3=6.0, dmp5=6.0, mprad=torch.zeros(10), mpvcn=torch.zeros(10))
    # r_cov, k_cn for mrad (here zero radii, zero CN)
    r_cov = torch.zeros(10, dtype=dtype)
    E1, _ = aes_energy_and_fock(numbers.to(dtype=dtype), pos, basis, P, S, D, Q, params, r_cov=r_cov, k_cn=0.0)
    pos_far = torch.tensor([[0.0, 0.0, 0.0], [2.4, 0.0, 0.0]], dtype=dtype)
    S2, D2, Q2 = build_moment_matrices(numbers.to(dtype=dtype), pos_far, basis)
    E2, _ = aes_energy_and_fock(numbers.to(dtype=dtype), pos_far, basis, P, S2, D2, Q2, params, r_cov=r_cov, k_cn=0.0)
    assert abs(E2) < abs(E1)
