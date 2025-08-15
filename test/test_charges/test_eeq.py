import torch
import pytest

from gxtb.charges.eeq import compute_eeq_charges
from gxtb.params.loader import load_eeq_params


def test_single_atom_charge(tmp_path):
    eeq = load_eeq_params('parameters/eeq')
    numbers = torch.tensor([6])  # carbon
    pos = torch.zeros(1,3)
    q = compute_eeq_charges(numbers, pos, eeq, total_charge=1.0)
    assert torch.allclose(q, torch.tensor([1.0], dtype=q.dtype))


def test_neutrality_three_atom():
    eeq = load_eeq_params('parameters/eeq')
    numbers = torch.tensor([6,1,1])
    pos = torch.tensor([[0.0,0.0,0.0],[1.1,0,0],[-1.1,0,0]])
    q = compute_eeq_charges(numbers, pos, eeq, total_charge=0.0)
    assert abs(q.sum().item()) < 1e-10


def test_permutation_invariance():
    eeq = load_eeq_params('parameters/eeq')
    numbers = torch.tensor([8,1,1])
    pos = torch.tensor([[0.0,0.0,0.0],[0.96,0,0],[-0.24,0.93,0]])
    q1 = compute_eeq_charges(numbers, pos, eeq, total_charge=0.0)
    perm = torch.tensor([1,0,2])
    q2 = compute_eeq_charges(numbers[perm], pos[perm], eeq, total_charge=0.0)
    # Reorder q2 back and compare
    q2_back = torch.empty_like(q2)
    q2_back[perm] = q2
    assert torch.allclose(q1, q2_back, atol=1e-8)
