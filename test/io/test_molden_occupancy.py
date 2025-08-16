import re
from pathlib import Path

import numpy as np
import pytest

ase_ok = True
try:
    from ase import Atoms
except Exception:  # pragma: no cover
    ase_ok = False


def _sum_occupancies(text: str) -> list[float]:
    # Return a list of sums of Occup across each [MO] block
    blocks = text.split('[MO]')
    sums = []
    for b in blocks[1:]:
        s = 0.0
        for ln in b.splitlines():
            ln = ln.strip()
            if ln.startswith('Occup='):
                val = float(ln.split('=')[1].strip())
                s += val
        if s != 0.0:
            sums.append(s)
    return sums


@pytest.mark.skipif(not ase_ok, reason="ASE not available")
def test_rhf_occupancy_sum_equals_nelec(tmp_path: Path):
    from gxtb.ase_calc import GxTBCalculator

    atoms = Atoms(numbers=[1, 1], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    atoms.info['q_eeqbc'] = [0.0, 0.0]
    out = tmp_path / 'h2_rhf.molden'
    calc = GxTBCalculator(parameters_dir='parameters', device='cpu', molden_path=str(out), molden_spherical=True)
    atoms.calc = calc
    _ = atoms.get_potential_energy()

    txt = out.read_text()
    sums = _sum_occupancies(txt)
    assert len(sums) == 1
    # Two electrons total in H2 (1 per atom)
    assert np.isclose(sums[0], 2.0)


@pytest.mark.skipif(not ase_ok, reason="ASE not available")
def test_uhf_occupancy_sums_equal_spin_electrons(tmp_path: Path):
    from gxtb.ase_calc import GxTBCalculator

    atoms = Atoms(numbers=[1, 1], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    atoms.info['q_eeqbc'] = [0.0, 0.0]
    out = tmp_path / 'h2_uhf.molden'
    # Enforce 1 alpha and 1 beta electron
    calc = GxTBCalculator(parameters_dir='parameters', device='cpu', molden_path=str(out), molden_spherical=True, uhf=True, nelec_alpha=1, nelec_beta=1)
    atoms.calc = calc
    _ = atoms.get_potential_energy()

    txt = out.read_text()
    sums = _sum_occupancies(txt)
    # Two [MO] blocks: Alpha then Beta
    assert len(sums) == 2
    assert np.isclose(sums[0], 1.0)
    assert np.isclose(sums[1], 1.0)

