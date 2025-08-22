import os
import numpy as np
import torch
import pytest

import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
_src = _root / 'src'
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from ase import Atoms
from gxtb.ase_calc import GxTBCalculator


def _measure(ps):
    O, H1, H2 = ps[0], ps[1], ps[2]
    r1 = np.linalg.norm(H1-O); r2 = np.linalg.norm(H2-O)
    v1 = H1-O; v2 = H2-O
    c = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    c = np.clip(c, -1.0, 1.0)
    ang = np.degrees(np.arccos(c))
    return r1, r2, ang


@pytest.mark.skipif(os.environ.get('GXTBRUNOPT','0') != '1', reason="Enable with GXTBRUNOPT=1")
def test_water_all_terms_nonpbc_print():
    # Near-equilibrium water
    O = np.array([0.000000, 0.000000, 0.000000])
    H1 = np.array([0.9572, 0.000000, 0.000000])
    H2 = np.array([-0.2399872, 0.927297, 0.000000])
    pos0 = np.vstack([O, H1, H2])
    atoms = Atoms('OH2', positions=pos0)
    # Enable most terms: second order, third, fourth, AES, dispersion
    calc = GxTBCalculator(
        parameters_dir='parameters',
        enable_second_order=True,
        enable_third_order=True,
        enable_fourth_order=True,
        enable_aes=True,
        enable_dispersion=True,
        d4_variant='bj-eeq-atm',
        force_diff='central', force_eps=6e-4,
        device='cpu',
    )
    atoms.calc = calc
    e = atoms.get_potential_energy()  # eV
    r1, r2, ang = _measure(atoms.get_positions())
    # Print energy decomposition keys if present
    print(f"Total energy (eV): {e:.6f}")
    for key in ("E_increment_eV","E_repulsion_eV","E_elec_eV","E2_eV"):
        val = atoms.calc.results.get(key, None)
        if val is not None:
            print(f"  {key} = {val:.6f}")
    # Forces (numeric) and geometry sanity
    F = atoms.get_forces()
    fnorm = float(np.linalg.norm(F))
    print(f"|F| (eV/Å): {fnorm:.6f}")
    print(f"Geometry: OH1={r1:.4f} Å, OH2={r2:.4f} Å, HOH={ang:.2f}°")
    # Sanity ranges
    assert np.isfinite(e)
    assert 0.6 <= r1 <= 1.4
    assert 0.6 <= r2 <= 1.4
    assert 70.0 <= ang <= 140.0

