import torch

from gxtb.params.loader import load_gxtb_params, load_basisq
from gxtb.params.schema import load_schema
from gxtb.basis.qvszp import build_atom_basis
from gxtb.hamiltonian.scf_adapter import build_eht_core
from gxtb.hamiltonian.eht import build_eht_hamiltonian, first_order_energy, eht_energy_gradient


def test_eht_gradient_cn_only_matches_fd_for_diag_P_H2():
    # H2 molecule where we use diagonal P so only diagonal eps(CN) contributes
    dtype = torch.float64
    device = torch.device("cpu")
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    numbers = torch.tensor([1, 1], dtype=torch.long, device=device)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]], dtype=dtype, device=device)
    basis = build_atom_basis(numbers, load_basisq('parameters/basisq'))
    core = build_eht_core(numbers, positions, basis, gparams, schema)
    S = core['S']
    eht = build_eht_hamiltonian(numbers, positions, basis, gparams, schema, wolfsberg_mode='arithmetic')
    H = eht.H
    # Diagonal density (one electron pair in first AO per atom for simplicity)
    nao = H.shape[0]
    P = torch.eye(nao, dtype=dtype, device=device)
    E0 = first_order_energy(P, H)
    # Analytic gradient (Step A, CN and Π terms; Π terms vanish for diag P)
    g = eht_energy_gradient(numbers, positions, basis, gparams, schema, P)
    # Finite difference on both atoms
    eps = 1e-5
    fd = torch.zeros_like(positions)
    for A in range(numbers.shape[0]):
        for k in range(3):
            disp = torch.zeros_like(positions)
            disp[A, k] = eps
            Hp = build_eht_hamiltonian(numbers, positions + disp, basis, gparams, schema, wolfsberg_mode='arithmetic').H
            Hm = build_eht_hamiltonian(numbers, positions - disp, basis, gparams, schema, wolfsberg_mode='arithmetic').H
            Ep = first_order_energy(P, Hp)
            Em = first_order_energy(P, Hm)
            fd[A, k] = (Ep - Em) / (2 * eps)
    # Compare
    assert torch.allclose(g, fd, atol=1e-6, rtol=1e-6)

