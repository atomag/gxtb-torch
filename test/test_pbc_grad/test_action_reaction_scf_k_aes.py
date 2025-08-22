import numpy as np
import torch

import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
_src = _root / 'src'
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from ase import Atoms
from gxtb.ase_calc import GxTBCalculator


def test_action_reaction_scf_k_aes_forces_sum_to_zero():
    # Simple H2 in a periodic cell; use forward differences for speed
    a = 6.0
    cell = np.array([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]], float)
    pos = np.array([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]], float)
    atoms = Atoms('H2', positions=pos, cell=cell, pbc=True)
    calc = GxTBCalculator(
        parameters_dir='parameters',
        pbc_mode='scf-k',
        mp_grid=(1,1,1), mp_shift=(0.0,0.0,0.0),
        pbc_cutoff=5.0, pbc_cn_cutoff=5.0, pbc_aes_cutoff=6.0, pbc_aes_high_order_cutoff=4.0,
        enable_second_order=True, ewald_eta=0.35, ewald_r_cut=6.0, ewald_g_cut=6.0,
        enable_aes=True,
        enable_dynamic_overlap=False,
        force_diff='central', force_eps=3e-4,
        device='cpu',
    )
    atoms.calc = calc
    F = atoms.get_forces()
    # Smoke: shape and finiteness
    assert F.shape == pos.shape
    assert np.isfinite(F).all()
    # Action–reaction: total force in the cell ~ 0
    Ft = F.sum(axis=0)
    assert np.allclose(Ft, np.zeros(3), atol=5e-3)
