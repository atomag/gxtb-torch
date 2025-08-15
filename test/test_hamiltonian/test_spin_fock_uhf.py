import torch

from gxtb.params.loader import load_basisq
from gxtb.basis.qvszp import build_atom_basis
from gxtb.hamiltonian.moments_builder import build_moment_matrices
from gxtb.hamiltonian.spin import SpinParams, add_spin_fock_uhf, spin_energy, compute_shell_magnetizations


def test_spin_fock_uhf_signs_and_zero_magnetization():
    dtype = torch.float64
    device = torch.device("cpu")
    numbers = torch.tensor([6], dtype=torch.long, device=device)
    positions = torch.tensor([[0.0,0.0,0.0]], dtype=dtype, device=device)
    bq = load_basisq('parameters/basisq')
    basis = build_atom_basis(numbers, bq)
    S, D, Q = build_moment_matrices(numbers, positions, basis)
    nao = S.shape[0]
    # Build Pa, Pb with zero magnetization: identical projectors
    evals, evecs = torch.linalg.eigh(S)
    X = evecs @ torch.diag(evals.clamp_min(1e-12).rsqrt()) @ evecs.T
    nocc = max(1, nao // 2)
    C = X
    P0 = C[:, :nocc] @ C[:, :nocc].T
    Pa = P0.clone(); Pb = P0.clone()
    # kW and W0 simple constants
    Zmax = 93
    kW = torch.zeros(Zmax+1, dtype=dtype)
    kW[6] = 1.0
    W0 = torch.eye(4, dtype=dtype)  # s,p,d,f identity
    params = SpinParams(kW_elem=kW, W0=W0)
    Ha = torch.zeros_like(S); Hb = torch.zeros_like(S)
    add_spin_fock_uhf(Ha, Hb, S, numbers, basis, Pa, Pb, params)
    # Zero magnetization yields zero spin Fock
    assert torch.allclose(Ha, torch.zeros_like(Ha), atol=1e-12)
    assert torch.allclose(Hb, torch.zeros_like(Hb), atol=1e-12)
    # Introduce magnetization: Pa = P0 + dP, Pb = P0 - dP
    dP = torch.randn_like(S) * 1e-5
    dP = 0.5 * (dP + dP.T)
    Pa = P0 + dP; Pb = P0 - dP
    Ha2 = torch.zeros_like(S); Hb2 = torch.zeros_like(S)
    add_spin_fock_uhf(Ha2, Hb2, S, numbers, basis, Pa, Pb, params)
    # Opposite signs
    assert torch.allclose(Ha2, -Hb2, atol=1e-12)

