import re
from pathlib import Path

import numpy as np
import pytest
import torch

ase_ok = True
try:
    from ase import Atoms
except Exception:  # pragma: no cover
    ase_ok = False


def _count_atoms_section(text: str) -> int:
    # Count number of atom lines under [Atoms] until next '[' section
    lines = iter(text.splitlines())
    count = 0
    in_atoms = False
    for ln in lines:
        if ln.strip().startswith('[Atoms]'):
            in_atoms = True
            continue
        if in_atoms:
            if ln.strip().startswith('['):
                break
            if ln.strip():
                count += 1
    return count


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
        if s != 0.0 or ('Occup=' in b):
            sums.append(s)
    return sums


def _count_coeff_lines_in_first_mo(text: str) -> int:
    # Count coefficient lines in the first [MO] block
    parts = text.split('[MO]')
    if len(parts) < 2:
        return 0
    block = parts[1]
    cnt = 0
    for ln in block.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith('Sym=') or ln.startswith('Ene=') or ln.startswith('Spin=') or ln.startswith('Occup='):
            continue
        toks = ln.split()
        if toks and toks[0].isdigit():
            cnt += 1
    return cnt


def _ao_count_from_basis(numbers: list[int]) -> int:
    from gxtb.params.loader import load_basisq
    from gxtb.basis.qvszp import build_atom_basis
    basisq = load_basisq('parameters/basisq')
    tnum = torch.tensor(numbers, dtype=torch.int64)
    ab = build_atom_basis(tnum, basisq)
    return int(ab.nao)


@pytest.mark.skipif(not ase_ok, reason="ASE not available")
def test_molden_sanity_h2_and_h(tmp_path: Path):
    from gxtb.ase_calc import GxTBCalculator

    # H2 RHF
    atoms = Atoms(numbers=[1, 1], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    atoms.info['q_eeqbc'] = [0.0, 0.0]
    out_h2 = tmp_path / 'h2_rhf.molden'
    calc_h2 = GxTBCalculator(parameters_dir='parameters', device='cpu', molden_path=str(out_h2), molden_spherical=True)
    atoms.calc = calc_h2
    _ = atoms.get_potential_energy()
    txt = out_h2.read_text()
    # Atoms count
    assert _count_atoms_section(txt) == 2
    # AO lines in first MO equals AO count from basis
    assert _count_coeff_lines_in_first_mo(txt) == _ao_count_from_basis([1, 1])
    # Occupancy sum equals total electrons (2)
    sums = _sum_occupancies(txt)
    assert len(sums) == 1 and np.isclose(sums[0], 2.0)

    # H atom UHF (1 alpha, 0 beta)
    atom_h = Atoms(numbers=[1], positions=[[0.0, 0.0, 0.0]])
    atom_h.info['q_eeqbc'] = [0.0]
    out_h = tmp_path / 'h_uhf.molden'
    calc_h = GxTBCalculator(parameters_dir='parameters', device='cpu', molden_path=str(out_h), molden_spherical=True, uhf=True, nelec_alpha=1, nelec_beta=0)
    atom_h.calc = calc_h
    _ = atom_h.get_potential_energy()
    txt_h = out_h.read_text()
    assert _count_atoms_section(txt_h) == 1
    assert _count_coeff_lines_in_first_mo(txt_h) == _ao_count_from_basis([1])
    sums_h = _sum_occupancies(txt_h)
    assert len(sums_h) == 2 and np.isclose(sums_h[0], 1.0) and np.isclose(sums_h[1], 0.0)


@pytest.mark.skipif(not ase_ok, reason="ASE not available")
def test_molden_sanity_cl2(tmp_path: Path):
    from gxtb.ase_calc import GxTBCalculator
    from gxtb.scf import _valence_electron_counts
    from gxtb.params.loader import load_basisq
    from gxtb.basis.qvszp import build_atom_basis

    atoms = Atoms(numbers=[17, 17], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    atoms.info['q_eeqbc'] = [0.0, 0.0]
    out = tmp_path / 'cl2.molden'
    calc = GxTBCalculator(parameters_dir='parameters', device='cpu', molden_path=str(out), molden_spherical=True)
    atoms.calc = calc
    _ = atoms.get_potential_energy()
    txt = out.read_text()
    assert _count_atoms_section(txt) == 2
    assert _count_coeff_lines_in_first_mo(txt) == _ao_count_from_basis([17, 17])
    # Occupancy sum equals valence electrons from basis-aware count
    basisq = load_basisq('parameters/basisq')
    numbers = torch.tensor([17, 17], dtype=torch.int64)
    ab = build_atom_basis(numbers, basisq)
    val = _valence_electron_counts(numbers, ab)
    nelec = float(val.sum().item())
    sums = _sum_occupancies(txt)
    assert len(sums) == 1 and np.isclose(sums[0], nelec)


@pytest.mark.skipif(not ase_ok, reason="ASE not available")
def test_molden_sanity_pt_2x2(tmp_path: Path):
    from gxtb.ase_calc import GxTBCalculator
    from gxtb.scf import _valence_electron_counts
    from gxtb.params.loader import load_basisq
    from gxtb.basis.qvszp import build_atom_basis

    # 2x2 Pt cluster (square)
    positions = [
        [0.0, 0.0, 0.0],
        [2.4, 0.0, 0.0],
        [0.0, 2.4, 0.0],
        [2.4, 2.4, 0.0],
    ]
    numbers = [78, 78, 78, 78]
    atoms = Atoms(numbers=numbers, positions=positions)
    atoms.info['q_eeqbc'] = [0.0] * 4
    out = tmp_path / 'pt4.molden'
    calc = GxTBCalculator(parameters_dir='parameters', device='cpu', molden_path=str(out), molden_spherical=True)
    atoms.calc = calc
    _ = atoms.get_potential_energy()
    txt = out.read_text()
    assert _count_atoms_section(txt) == 4
    assert _count_coeff_lines_in_first_mo(txt) == _ao_count_from_basis(numbers)
    # Occupancy sum equals basis-aware valence electrons
    basisq = load_basisq('parameters/basisq')
    tnum = torch.tensor(numbers, dtype=torch.int64)
    ab = build_atom_basis(tnum, basisq)
    val = _valence_electron_counts(tnum, ab)
    nelec = float(val.sum().item())
    sums = _sum_occupancies(txt)
    assert len(sums) == 1 and np.isclose(sums[0], nelec)

