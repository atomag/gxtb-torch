import io
from pathlib import Path

import pytest

from gxtb.params.loader import load_basisq
from gxtb.io.molden import write as write_molden, shells_from_qvszp, MOSet, MOWavefunction


def test_write_molden_h2(tmp_path: Path):
    basisq = load_basisq("parameters/basisq")
    numbers = [1, 1]
    symbols = ["H", "H"]
    coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]
    q_eff = [0.0, 0.0]

    shells = shells_from_qvszp(numbers, basisq, q_eff)
    # Expect two s AO's (one per H)
    nao = 2
    assert sum({0: 1, 1: 3, 2: 5}.get(sh.l, 0) for sh in shells) == nao

    # Minimal RHF-like MO set: 2x2 identity coeffs
    coeff = [[1.0, 0.0], [0.0, 1.0]]
    energy = [0.0, 0.1]
    occ = [1.0, 1.0]
    wf = MOWavefunction(alpha=MOSet(coeff=coeff, energy=energy, occ=occ))

    out = tmp_path / "h2.molden"
    write_molden(
        out,
        numbers=numbers,
        symbols=symbols,
        coords=coords,
        unit="Angs",
        shells=shells,
        wf=wf,
        spherical=True,
        program="gxtb",
        version="test",
    )

    txt = out.read_text()
    # Sanity: headers present
    assert "[Molden Format]" in txt
    assert "[Atoms] (Angs)" in txt
    assert "[GTO]" in txt
    assert "[MO]" in txt
    # Angular momentum flag for spherical
    assert "[5d]" in txt and "[7f]" in txt and "[9g]" in txt
    # Atoms lines
    assert txt.count("\nH   1   1") == 1
    assert txt.count("\nH   2   1") == 1
    # One s-shell per atom, with the correct primitive count
    nprim = len(shells[0].exps)
    assert f" s   {nprim:2d} 1.00" in txt
    # Two AO coefficients per MO
    mo_blocks = txt.split("[MO]")
    assert len(mo_blocks) >= 2  # at least one [MO]
    # Count coefficient lines by pattern of leading index '  1' and '  2'
    # Simpler: ensure both AO indices appear twice (for 2 MOs)
    assert txt.count("\n   1    ") == 2
    assert txt.count("\n   2    ") == 2


def _parse_mo_coeffs(txt: str, ncoeff: int) -> list[float]:
    # Return coefficient list from the first [MO] block
    after = txt.split("[MO]\n", 1)[1]
    lines = after.splitlines()
    vals: list[float] = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if ln.startswith("Sym="):
            continue
        if ln.startswith("Ene="):
            continue
        if ln.startswith("Spin="):
            continue
        if ln.startswith("Occup="):
            continue
        parts = ln.split()
        if len(parts) >= 2 and parts[0].isdigit():
            vals.append(float(parts[1]))
            if len(vals) == ncoeff:
                break
    return vals


def test_write_molden_single_f_shell_order(tmp_path: Path):
    # Single atom with only one f shell; check AO order mapping is applied
    from gxtb.io.molden import ShellRecord
    numbers = [58]
    symbols = ["Ce"]
    coords = [[0.0, 0.0, 0.0]]
    # One contraction with 1 primitive is enough for I/O
    sh = ShellRecord(atom_index=0, l=3, exps=(1.0,), contractions=((1.0,),))
    shells = [sh]
    # Internal AO rows = 7; set coeff row i to value i (0..6) to see permutation clearly
    coeff = [[float(i)] for i in range(7)]  # shape (7,1)
    wf = MOWavefunction(alpha=MOSet(coeff=coeff, energy=[-0.1], occ=[1.0]))
    out = tmp_path / "ce_f.molden"
    write_molden(
        out,
        numbers=numbers,
        symbols=symbols,
        coords=coords,
        unit="AU",
        shells=shells,
        wf=wf,
        spherical=True,
    )
    txt = out.read_text()
    coeffs = _parse_mo_coeffs(txt, 7)
    # Expected internal->Molden mapping for f: [3,4,2,5,1,6,0]
    expected = [coeff[i][0] for i in [3, 4, 2, 5, 1, 6, 0]]
    assert coeffs == expected


def test_write_molden_single_g_shell_order(tmp_path: Path):
    # Synthetic single g shell; verifies mapping without requiring g in basisq
    from gxtb.io.molden import ShellRecord
    numbers = [1]
    symbols = ["H"]
    coords = [[0.0, 0.0, 0.0]]
    sh = ShellRecord(atom_index=0, l=4, exps=(1.5,), contractions=((0.7,),))
    shells = [sh]
    coeff = [[float(i)] for i in range(9)]  # shape (9,1)
    wf = MOWavefunction(alpha=MOSet(coeff=coeff, energy=[0.2], occ=[0.0]))
    out = tmp_path / "g_shell.molden"
    write_molden(
        out,
        numbers=numbers,
        symbols=symbols,
        coords=coords,
        unit="AU",
        shells=shells,
        wf=wf,
        spherical=True,
    )
    txt = out.read_text()
    coeffs = _parse_mo_coeffs(txt, 9)
    # Expected mapping for g: [4,5,3,6,2,7,1,8,0]
    expected = [coeff[i][0] for i in [4, 5, 3, 6, 2, 7, 1, 8, 0]]
    assert coeffs == expected


def test_write_molden_symm_and_scaling(tmp_path: Path):
    # Two s AOs with simple coeffs, custom symmetry labels, and AO row scaling
    from gxtb.io.molden import ShellRecord
    numbers = [1, 1]
    symbols = ["H", "H"]
    coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]
    sh1 = ShellRecord(atom_index=0, l=0, exps=(1.0,), contractions=((1.0,),))
    sh2 = ShellRecord(atom_index=1, l=0, exps=(1.5,), contractions=((1.0,),))
    shells = [sh1, sh2]
    # coeff shape (2,2)
    coeff = [[1.0, 0.0], [0.0, 1.0]]
    energy = [-0.5, 0.1]
    occ = [2.0, 0.0]
    symm = ["A1", "B1"]
    wf = MOWavefunction(alpha=MOSet(coeff=coeff, energy=energy, occ=occ, symm=symm))
    out = tmp_path / "symscale.molden"
    write_molden(
        out,
        numbers=numbers,
        symbols=symbols,
        coords=coords,
        unit="Angs",
        shells=shells,
        wf=wf,
        spherical=True,
        ao_row_scale=[2.0, 3.0],
    )
    txt = out.read_text()
    # Symmetry labels present
    assert "Sym= A1" in txt and "Sym= B1" in txt
    # Coefficients scaled: first MO has (AO1=1*2, AO2=0*3) => 2.0 at index 1; second MO -> 3.0 at index 2
    assert "\n   1    2" in txt  # rough check that 2.xx appears for AO 1, first MO
    assert "\n   2    3" in txt  # 3.xx for AO 2, second MO


def test_write_molden_uhf_two_blocks(tmp_path: Path):
    # Synthetic UHF: separate alpha/beta coeffs
    from gxtb.io.molden import ShellRecord
    numbers = [1]
    symbols = ["H"]
    coords = [[0.0, 0.0, 0.0]]
    shells = [ShellRecord(atom_index=0, l=0, exps=(1.0,), contractions=((1.0,),))]
    coeff_a = [[1.0]]
    coeff_b = [[-1.0]]
    wf = MOWavefunction(
        alpha=MOSet(coeff=coeff_a, energy=[-0.1], occ=[1.0], symm=["A1"]),
        beta=MOSet(coeff=coeff_b, energy=[-0.2], occ=[1.0], symm=["A1"]),
    )
    out = tmp_path / "uhf.molden"
    write_molden(
        out,
        numbers=numbers,
        symbols=symbols,
        coords=coords,
        unit="AU",
        shells=shells,
        wf=wf,
        spherical=True,
    )
    txt = out.read_text()
    # Should contain two [MO] sections, one with Spin= Alpha and one with Spin= Beta
    assert txt.count("[MO]") == 2
    assert txt.count("Spin= Alpha") == 1
    assert txt.count("Spin= Beta") == 1


def test_write_molden_cartesian_p_shell(tmp_path: Path):
    # Single p shell; internal spherical order is (py,pz,px). Set coefficients to select px only.
    from gxtb.io.molden import ShellRecord
    numbers = [6]
    symbols = ["C"]
    coords = [[0.0, 0.0, 0.0]]
    shells = [ShellRecord(atom_index=0, l=1, exps=(1.0,), contractions=((1.0,),))]
    # One MO; spherical coeffs: [py=0, pz=0, px=1]
    coeff = [[0.0], [0.0], [1.0]]
    wf = MOWavefunction(alpha=MOSet(coeff=coeff, energy=[-0.1], occ=[1.0]))
    out = tmp_path / "p_cart.molden"
    write_molden(
        out,
        numbers=numbers,
        symbols=symbols,
        coords=coords,
        unit="AU",
        shells=shells,
        wf=wf,
        spherical=False,
    )
    txt = out.read_text()
    # Cartesian flags present
    assert "[6d]" in txt and "[10f]" in txt and "[15g]" in txt
    # First three AO coeff lines should reflect px, py, pz in that order; expect |px|>>0 and py,pz≈0
    lines = [ln for ln in txt.splitlines() if ln.strip().split()[0] in ('1','2','3')][:3]
    vals = [float(ln.split()[1].replace('D','e')) for ln in lines]
    assert abs(vals[0]) > 1e-6 and abs(vals[1]) < 1e-12 and abs(vals[2]) < 1e-12
