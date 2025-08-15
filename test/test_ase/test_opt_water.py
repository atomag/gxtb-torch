import numpy as np
import pytest

ase_ok = True
try:
    from ase import Atoms
    from ase.optimize import BFGS
except Exception:
    ase_ok = False

@pytest.mark.skipif(not ase_ok, reason="ASE not available")
def test_optimize_water_in_box():
    from gxtb.ase_calc import GxTBCalculator

    # Build H2O: O at origin, H along x/y with ~0.96 A bonds, 104.5 deg
    # Positions in Angstrom
    angle = np.deg2rad(104.5)
    OH = 0.96
    O = np.array([0.0, 0.0, 0.0])
    H1 = np.array([OH, 0.0, 0.0])
    H2 = np.array([OH*np.cos(angle), OH*np.sin(angle), 0.0])
    positions = np.vstack([O, H1, H2])
    numbers = [8, 1, 1]
    atoms = Atoms(numbers=numbers, positions=positions)
    # Box 10x10x10 A, not PBC
    atoms.cell = np.diag([10.0, 10.0, 10.0])
    atoms.pbc = False

    calc = GxTBCalculator(parameters_dir='parameters', device='cpu')
    atoms.calc = calc

    # Optimize with few steps using numerical forces
    dyn = BFGS(atoms, logfile=None)
    dyn.run(fmax=0.2, steps=5)

    # Inspect O-H distances and H-O-H angle
    pos = atoms.get_positions()
    O = pos[0]; H1 = pos[1]; H2 = pos[2]
    d1 = np.linalg.norm(H1 - O)
    d2 = np.linalg.norm(H2 - O)
    v1 = (H1 - O) / d1
    v2 = (H2 - O) / d2
    ang = np.rad2deg(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))

    # Loose tolerances to allow method variability
    assert 0.8 < d1 < 1.2
    assert 0.8 < d2 < 1.2
    assert 95.0 < ang < 115.0
