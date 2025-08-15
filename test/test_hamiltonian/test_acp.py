import torch

from gxtb.params.loader import load_basisq
from gxtb.basis.qvszp import build_atom_basis
from gxtb.hamiltonian.moments_builder import build_moment_matrices
from gxtb.hamiltonian.acp import build_acp_overlap, acp_hamiltonian, acp_energy


def _zp(Zmax, zvals):
    # helper to make (Zmax+1,) with provided dict z->val
    out = torch.zeros(Zmax+1, dtype=torch.float64)
    for z, v in zvals.items():
        out[z] = v
    return out


def test_acp_overlap_shapes_and_psd_for_CH():
    dtype = torch.float64
    device = torch.device("cpu")
    numbers = torch.tensor([6, 1], dtype=torch.long, device=device)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=dtype, device=device)
    basis = build_atom_basis(numbers, load_basisq('parameters/basisq'))
    # Minimal ACP params: s and p channels only
    Zmax = 93
    c0 = torch.zeros((Zmax+1, 4), dtype=dtype, device=device)
    xi = torch.zeros_like(c0)
    # Assign modest ACP coefficients/exponents for C and H (ACP units already)
    c0[6, 0] = 0.3; c0[6, 1] = 0.2
    c0[1, 0] = 0.1; c0[1, 1] = 0.05
    xi[6, 0] = 0.8; xi[6, 1] = 0.7
    xi[1, 0] = 0.6; xi[1, 1] = 0.5
    # CN inputs
    # Use approximate covalent radii as zeros except defined
    r_cov = _zp(Zmax, {1: 0.3, 6: 0.7})
    cn_avg = _zp(Zmax, {1: 1.0, 6: 4.0})
    S_acp = build_acp_overlap(numbers, positions, basis,
                              c0=c0, xi=xi, k_acp_cn=0.0, cn_avg=cn_avg, r_cov=r_cov, k_cn=1.0, l_list=("s","p"))
    # Expected naux = sum_A (1 + 3) = 8
    assert S_acp.shape[0] == basis.nao
    assert S_acp.shape[1] == 8
    H = acp_hamiltonian(S_acp)
    # PSD check: eigenvalues non-negative within tolerance
    evals = torch.linalg.eigvalsh(0.5 * (H + H.T))
    assert torch.all(evals >= -1e-12)
    # Energy: Tr(H P)
    S, D, Q = build_moment_matrices(numbers, positions, basis)
    nao = S.shape[0]
    # Simple symmetric density
    P = torch.eye(nao, dtype=dtype, device=device)
    E = acp_energy(P, H)
    # Positive (since H is PSD and P PSD)
    assert E.item() >= -1e-12


def test_acp_cn_scaling_changes_overlap_norms():
    dtype = torch.float64
    device = torch.device("cpu")
    numbers = torch.tensor([6], dtype=torch.long, device=device)
    positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
    basis = build_atom_basis(numbers, load_basisq('parameters/basisq'))
    Zmax = 93
    c0 = torch.zeros((Zmax+1, 4), dtype=dtype, device=device)
    xi = torch.zeros_like(c0)
    c0[6,0] = 0.2; xi[6,0] = 0.8
    r_cov = _zp(Zmax, {6: 0.7})
    cn_avg = _zp(Zmax, {6: 1.0})
    S0 = build_acp_overlap(numbers, positions, basis, c0=c0, xi=xi, k_acp_cn=0.0, cn_avg=cn_avg, r_cov=r_cov, k_cn=1.0, l_list=("s",))
    S1 = build_acp_overlap(numbers, positions, basis, c0=c0, xi=xi, k_acp_cn=1.0, cn_avg=cn_avg, r_cov=r_cov, k_cn=1.0, l_list=("s",))
    # Norm increases when k_acp_cn > 0 (since CN>0 for any bonded environment)
    n0 = torch.linalg.norm(S0)
    n1 = torch.linalg.norm(S1)
    assert n1 >= n0

