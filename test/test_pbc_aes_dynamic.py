import numpy as np
import torch

import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
_src = _root / 'src'
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from gxtb.ase_calc import GxTBCalculator
from gxtb.params.schema import load_schema
from gxtb.params.loader import load_eeq_params
from gxtb.charges.eeq import compute_eeq_charges


def test_pbc_scf_k_dynamic_overlap_with_aes_runs():
    # Small H2 cell to keep periodic AES fast
    a = 7.0
    cell = np.array([[a, 0, 0], [0, a, 0], [0, 0, a]], float)
    pos = np.array([[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]], float)
    from ase import Atoms
    atoms = Atoms('H2', positions=pos, cell=cell, pbc=True)
    # EEQBC charges via internal EEQ using explicit schema mapping
    schema = load_schema('parameters/gxtb.schema.toml')
    eeq_map = getattr(schema, 'eeq', None)
    assert eeq_map is not None
    mapping = {k: int(v) for k, v in eeq_map.items()}
    q = compute_eeq_charges(torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int64),
                            torch.tensor(atoms.get_positions(), dtype=torch.float64),
                            load_eeq_params('parameters/eeq'),
                            0.0,
                            mapping=mapping,
                            device='cpu', dtype=torch.float64)
    atoms.info['q_eeqbc'] = q.cpu().numpy().tolist()
    # Calculator with dynamic overlap and AES enabled
    calc = GxTBCalculator(
        parameters_dir='parameters',
        pbc_mode='scf-k',
        mp_grid=(1,1,1), mp_shift=(0.0,0.0,0.0),
        pbc_cutoff=5.0, pbc_cn_cutoff=5.0, pbc_aes_cutoff=6.0,
        enable_second_order=True, ewald_eta=0.35, ewald_r_cut=8.0, ewald_g_cut=8.0,
        enable_aes=True,
        enable_dynamic_overlap=True,
        device='cpu',
    )
    atoms.calc = calc
    e = atoms.get_potential_energy()
    assert isinstance(e, float)
    assert np.isfinite(e)

