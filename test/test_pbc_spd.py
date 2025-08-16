import pytest

from ase import Atoms

import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
_src = _root / 'src'
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from gxtb.ase_calc import GxTBCalculator


def test_strict_spd_gamma_he_ok():
    # He in a cubic box should yield SPD S(Γ) with moderate cutoff
    a = 5.0
    atoms = Atoms('He', positions=[[0.0, 0.0, 0.0]], cell=[[a,0,0],[0,a,0],[0,0,a]], pbc=True)
    calc = GxTBCalculator(
        parameters_dir='parameters',
        pbc_mode='eht-gamma',
        pbc_cutoff=4.0,
        pbc_cn_cutoff=4.0,
        enable_second_order=False,
        enable_aes=False,
        s_psd_adapt=False,
        s_strict_spd=True,
        device='cpu',
    )
    atoms.calc = calc
    e = atoms.get_potential_energy()
    assert isinstance(e, float)

