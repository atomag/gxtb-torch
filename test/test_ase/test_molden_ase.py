import os
import pytest

ase_ok = True
try:
    from ase import Atoms
except Exception:  # pragma: no cover
    ase_ok = False


@pytest.mark.skipif(not ase_ok, reason="ASE not available")
def test_ase_calculator_writes_molden(tmp_path):
    from gxtb.ase_calc import GxTBCalculator

    atoms = Atoms(numbers=[1, 1], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    atoms.info['q_eeqbc'] = [0.0, 0.0]

    out = tmp_path / 'h2_ase.molden'
    calc = GxTBCalculator(parameters_dir='parameters', device='cpu', molden_path=str(out))
    atoms.calc = calc
    _ = atoms.get_potential_energy()

    assert out.exists(), "Molden file was not written by ASE calculator"
    txt = out.read_text()
    assert "[Molden Format]" in txt and "[MO]" in txt and "[GTO]" in txt

