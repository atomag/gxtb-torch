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
import os
import pytest


def _oh_bonds_and_angle(positions: np.ndarray) -> tuple[float, float, float]:
    # Atom order: O (0), H1 (1), H2 (2)
    O = positions[0]
    H1 = positions[1]
    H2 = positions[2]
    r1 = np.linalg.norm(H1 - O)
    r2 = np.linalg.norm(H2 - O)
    v1 = H1 - O
    v2 = H2 - O
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    c = np.clip(c, -1.0, 1.0)
    ang = np.degrees(np.arccos(c))
    return r1, r2, ang


@pytest.mark.skipif(os.environ.get('GXTBRUNOPT','0') != '1', reason="Skip slow numeric-force optimization unless GXTBRUNOPT=1")
def test_optimize_water_nonperiodic_reasonable_geometry(tmp_path):
    # Initial water geometry near experimental
    O = np.array([0.000000, 0.000000, 0.000000])
    H1 = np.array([0.9572, 0.000000, 0.000000])
    H2 = np.array([-0.2399872, 0.927297, 0.000000])
    pos0 = np.vstack([O, H1, H2])
    atoms = Atoms('OH2', positions=pos0)
    # Use forward-diff forces for speed; disable heavy terms
    calc = GxTBCalculator(
        parameters_dir='parameters',
        enable_second_order=False, enable_third_order=False, enable_fourth_order=False,
        enable_aes=False, enable_dispersion=False,
        force_diff='forward', force_eps=8e-4,
        device='cpu',
    )
    atoms.calc = calc
    # Take a single small relaxation step along -F to avoid long runtimes
    F = atoms.get_forces()
    # Take a normalized steepest-descent step with max displacement ~0.02 Å
    fnorm = np.linalg.norm(F, axis=1)
    maxf = float(np.max(fnorm)) if F.size else 0.0
    if maxf > 1e-12:
        alpha = 0.02 / maxf
        atoms.set_positions(atoms.get_positions() - alpha * F)
    r1, r2, ang = _oh_bonds_and_angle(atoms.get_positions())
    print(f"Non-PBC H2O: OH1={r1:.4f} Å, OH2={r2:.4f} Å, HOH={ang:.2f}°")
    # Reasonable ranges
    assert 0.75 <= r1 <= 1.20
    assert 0.75 <= r2 <= 1.20
    assert 90.0 <= ang <= 120.0


@pytest.mark.skipif(os.environ.get('GXTBRUNOPT','0') != '1', reason="Skip slow numeric-force optimization unless GXTBRUNOPT=1")
def test_optimize_water_pbc_gamma_reasonable_geometry(tmp_path):
    # Same molecule in a 5x5x5 Å box with PBC; Γ-only band energy
    a = 5.0
    cell = np.array([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]], float)
    O = np.array([0.000000, 0.000000, 0.000000])
    H1 = np.array([0.9572, 0.000000, 0.000000])
    H2 = np.array([-0.2399872, 0.927297, 0.000000])
    pos0 = np.vstack([O, H1, H2])
    atoms = Atoms('OH2', positions=pos0, cell=cell, pbc=True)
    calc = GxTBCalculator(
        parameters_dir='parameters',
        pbc_mode='eht-gamma',
        pbc_cutoff=4.0, pbc_cn_cutoff=4.0,
        enable_second_order=False, enable_aes=False, enable_dispersion=False,
        force_diff='forward', force_eps=1.0e-3,
        device='cpu',
    )
    atoms.calc = calc
    F = atoms.get_forces()
    fnorm = np.linalg.norm(F, axis=1)
    maxf = float(np.max(fnorm)) if F.size else 0.0
    if maxf > 1e-12:
        alpha = 0.02 / maxf
        atoms.set_positions(atoms.get_positions() - alpha * F)
    r1, r2, ang = _oh_bonds_and_angle(atoms.get_positions())
    print(f"PBC(5Å) H2O: OH1={r1:.4f} Å, OH2={r2:.4f} Å, HOH={ang:.2f}°")
    assert 0.70 <= r1 <= 1.30
    assert 0.70 <= r2 <= 1.30
    assert 85.0 <= ang <= 125.0
