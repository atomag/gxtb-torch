import torch

from gxtb.hamiltonian.second_order_tb import add_second_order_fock


def test_atomic_second_order_fock_mapping():
    # Two atoms, three AOs: atom0 has 2 AOs, atom1 has 1 AO
    ao_atoms = torch.tensor([0, 0, 1], dtype=torch.long)
    # Overlap matrix with off-diagonal elements to exercise mapping
    S = torch.tensor([
        [1.0, 0.2, 0.1],
        [0.2, 1.0, 0.3],
        [0.1, 0.3, 1.0],
    ], dtype=torch.float64)
    V_atom = torch.tensor([0.4, -0.2], dtype=torch.float64)  # V_0, V_1
    H = torch.zeros_like(S)
    add_second_order_fock(H, S, ao_atoms, V_atom)
    # Check a few entries: H_{μν} = 0.5 (V_A + V_B) S_{μν}
    def expected(mu, nu):
        A = int(ao_atoms[mu].item()); B = int(ao_atoms[nu].item())
        return 0.5 * (V_atom[A] + V_atom[B]) * S[mu, nu]
    for mu in range(3):
        for nu in range(3):
            assert torch.isclose(H[mu, nu], expected(mu, nu), atol=1e-14)

