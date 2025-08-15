import torch

from gxtb.params.loader import load_basisq
from gxtb.basis.qvszp import build_atom_basis
from gxtb.hamiltonian.overlap_tb import build_overlap


def test_min_eig_S_heavy_pair_not_near_singular():
    # Ar–Pt at 3 Å: previously near-singular; now well-conditioned
    numbers = torch.tensor([18, 78], dtype=torch.long)
    R = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64)
    bq = load_basisq('parameters/basisq')
    basis = build_atom_basis(numbers, bq)
    S = build_overlap(numbers, R, basis)
    evals = torch.linalg.eigvalsh(0.5 * (S + S.T))
    assert torch.all(evals > 1e-10)  # SPD
    assert float(evals.min()) > 1e-6

