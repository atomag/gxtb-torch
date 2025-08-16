import re
from pathlib import Path

import torch

from gxtb.io.molden import write as write_molden, ShellRecord, MOSet, MOWavefunction
from gxtb.basis.md_overlap import _metric_orthonormal_sph_transform, _overlap_cart_block


def _parse_first_mo_coeffs_cartesian(text: str, ncart: int) -> list[float]:
    # Find first [MO] block and return first ncart coefficient values (floats)
    mo_start = text.index("[MO]")
    lines = text[mo_start:].splitlines()
    coeffs = []
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("Sym=") or ln.startswith("Ene=") or ln.startswith("Spin=") or ln.startswith("Occup="):
            continue
        parts = ln.split()
        if parts and parts[0].isdigit() and len(parts) >= 2:
            # Molden uses plain floats; be tolerant
            coeffs.append(float(parts[1].replace('D','e')))
            if len(coeffs) == ncart:
                break
    return coeffs


def _molden_cart_order(l: int) -> list[int]:
    if l == 1:
        return [0, 1, 2]
    if l == 2:
        return [0, 3, 5, 1, 2, 4]
    if l == 3:
        return [0, 6, 9, 3, 1, 2, 5, 8, 7, 4]
    if l == 4:
        return [0, 10, 14, 1, 2, 6, 11, 9, 13, 3, 5, 12, 4, 7, 8]
    raise ValueError


def _n_sph(l: int) -> int:
    return {0: 1, 1: 3, 2: 5, 3: 7, 4: 9}[l]


def _n_cart(l: int) -> int:
    return (l + 1) * (l + 2) // 2


def test_cartesian_transform_matches_T_single_shell(tmp_path: Path):
    # Validate that the writer’s spherical->Cartesian mapping matches T^T from md_overlap
    for l in (1, 2, 3):
        # Build a single-shell atom, random exps/coeffs
        torch.manual_seed(1234 + l)
        nprim = 3
        exps = torch.rand(nprim, dtype=torch.float64) * 1.2 + 0.3
        cvec = torch.rand(nprim, dtype=torch.float64)
        sh = ShellRecord(atom_index=0, l=l, exps=tuple(float(x) for x in exps.tolist()), contractions=(tuple(float(x) for x in cvec.tolist()),))
        shells = [sh]
        nsph = _n_sph(l)
        ncart = _n_cart(l)
        # First MO: spherical e0 = [1,0,...]
        coeff = [[0.0] for _ in range(nsph)]
        coeff[0][0] = 1.0
        wf = MOWavefunction(alpha=MOSet(coeff=coeff, energy=[-0.1], occ=[1.0]))
        out = tmp_path / f"single_l{l}_cart.molden"
        write_molden(
            out,
            numbers=[6],
            symbols=["C"],
            coords=[[0.0, 0.0, 0.0]],
            unit="AU",
            shells=shells,
            wf=wf,
            spherical=False,
        )
        txt = out.read_text()
        got = _parse_first_mo_coeffs_cartesian(txt, ncart)
        # Compute expected using T from md_overlap and reorder to Molden order
        Scc = _overlap_cart_block(l, l, exps, cvec, exps, cvec, exps.new_zeros(3))
        T = _metric_orthonormal_sph_transform(l, Scc)  # nsph x ncart
        exp_cart = T.T[:, 0]  # ncart
        order = _molden_cart_order(l)
        exp_cart_molden = [float(exp_cart[i].item()) for i in order]
        # Compare within tight tolerance
        assert len(got) == len(exp_cart_molden)
        for a, b in zip(got, exp_cart_molden):
            assert abs(a - b) < 1e-10


def test_cartesian_transform_matches_T_multi_shell_dfg(tmp_path: Path):
    # Build a stack of d,f,g shells on one atom; MO has 1.0 on first spherical basis of each shell.
    shells = []
    exp_list = []
    c_list = []
    l_list = [2, 3, 4]
    torch.manual_seed(77)
    for l in l_list:
        nprim = 3
        exps = torch.rand(nprim, dtype=torch.float64) * 1.0 + 0.4
        cvec = torch.rand(nprim, dtype=torch.float64)
        exp_list.append(exps)
        c_list.append(cvec)
        shells.append(
            ShellRecord(
                atom_index=0,
                l=l,
                exps=tuple(float(x) for x in exps.tolist()),
                contractions=(tuple(float(x) for x in cvec.tolist()),),
            )
        )
    # Build MO spherical coefficients: concatenate [1,0,..] for each shell
    ns_total = sum({0:1,1:3,2:5,3:7,4:9}[l] for l in l_list)
    coeff = [[0.0] for _ in range(ns_total)]
    off = 0
    for l in l_list:
        coeff[off][0] = 1.0
        off += {0:1,1:3,2:5,3:7,4:9}[l]
    wf = MOWavefunction(alpha=MOSet(coeff=coeff, energy=[-0.2], occ=[1.0]))
    out = tmp_path / "stack_dfg_cart.molden"
    write_molden(
        out,
        numbers=[10],
        symbols=["Ne"],
        coords=[[0.0, 0.0, 0.0]],
        unit="AU",
        shells=shells,
        wf=wf,
        spherical=False,
    )
    txt = out.read_text()
    # Expected concatenation of per-shell T^T column 0, reordered to Molden
    expected = []
    for l, exps, cvec in zip(l_list, exp_list, c_list):
        Scc = _overlap_cart_block(l, l, exps, cvec, exps, cvec, exps.new_zeros(3))
        T = _metric_orthonormal_sph_transform(l, Scc)
        v = T.T[:, 0]
        order = _molden_cart_order(l)
        expected.extend([float(v[i].item()) for i in order])
    got = _parse_first_mo_coeffs_cartesian(txt, len(expected))
    assert len(got) == len(expected)
    for a, b in zip(got, expected):
        assert abs(a - b) < 1e-10
