import os
import pytest

ase_ok = True
try:
    from ase import Atoms
except Exception:  # pragma: no cover
    ase_ok = False


@pytest.mark.skipif(not ase_ok, reason="ASE not available")
def test_ase_calculator_writes_molden_cartesian(tmp_path):
    from gxtb.ase_calc import GxTBCalculator

    atoms = Atoms(numbers=[1, 1], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    atoms.info['q_eeqbc'] = [0.0, 0.0]

    out = tmp_path / 'h2_cart_ase.molden'
    calc = GxTBCalculator(parameters_dir='parameters', device='cpu', molden_path=str(out), molden_spherical=False)
    atoms.calc = calc
    _ = atoms.get_potential_energy()

    assert out.exists(), "Cartesian Molden file was not written by ASE calculator"
    txt = out.read_text()
    assert "[6d]" in txt and "[10f]" in txt and "[15g]" in txt
    assert "[MO]" in txt and "[GTO]" in txt

