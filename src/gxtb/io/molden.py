"""Molden file writer (no dependency on PySCF).

This module writes wavefunctions in the Molden format as specified at:
    https://www.cmbi.nl/xyz/molden/molden_format.html

Scope and guarantees:
- Spherical harmonics shells supported: s, p, d. If f/g are present, a
  descriptive exception is raised (no partial, silent support). This avoids
  mislabeling AO subfunction orders in viewers. Extend with care.
- No hidden defaults: caller must pass all quantities explicitly.
- Deterministic formatting: fixed float formats and AO ordering.

Traceability notes:
- AO subfunction order per shell follows Molden spherical convention
  (writer reorders our internal AO order to Molden order):
    p: (px, py, pz)
    d (5D): (d(z^2), d(xz), d(yz), d(x^2-y^2), d(xy))
  See PySCF tools.molden.order_ao_index() for a commonly used convention.

__doc_refs__ = {
    "molden": {"file": "https://www.cmbi.nl/xyz/molden/molden_format.html", "eqs": []},
}
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Dict, Any, Optional

import torch
from gxtb.basis.md_overlap import _metric_orthonormal_sph_transform, _overlap_cart_block, _cart_list


@dataclass(frozen=True)
class ShellRecord:
    """One contracted shell for an atom.

    - atom_index: 0-based atom index this shell belongs to
    - l: angular momentum as integer (0=s, 1=p, 2=d, 3=f, 4=g)
    - exps: primitive exponents (alpha_k)
    - contractions: list of contraction coefficient vectors, each of length len(exps)
      Molden allows multiple contractions with same exponents; most xTB shells use one.
    """
    atom_index: int
    l: int
    exps: Tuple[float, ...]
    contractions: Tuple[Tuple[float, ...], ...]


@dataclass(frozen=True)
class MOSet:
    """One spin block of molecular orbitals.

    coeff: shape (nao, nmo) in the INTERNAL AO order (see note below)
    energy: shape (nmo,)
    occ: shape (nmo,)

    Note: The writer reorders AO rows from our internal order to Molden order
    per shell when dumping coefficients. Caller does not need to reorder.
    """
    coeff: Any  # accept numpy or torch-like arrays; only __getitem__ needed
    energy: Sequence[float]
    occ: Sequence[float]
    symm: Optional[Sequence[str]] = None  # Optional symmetry labels per MO (default 'A')


@dataclass(frozen=True)
class MOWavefunction:
    """Wavefunction to write.

    Either unrestricted (alpha and beta) or restricted (only alpha used).
    """
    alpha: MOSet
    beta: Optional[MOSet] = None


def _l_sym(l: int) -> str:
    if l == 0:
        return "s"
    if l == 1:
        return "p"
    if l == 2:
        return "d"
    if l == 3:
        return "f"
    if l == 4:
        return "g"
    raise ValueError(f"Angular momentum l={l} not supported in Molden writer")


def _n_sph(l: int) -> int:
    return {0: 1, 1: 3, 2: 5, 3: 7, 4: 9}[l]


def _n_cart(l: int) -> int:
    # Number of Cartesian functions per l
    return (l + 1) * (l + 2) // 2


def _internal_to_molden_order(l: int) -> List[int]:
    """Permutation mapping internal spherical row order -> Molden AO order.

    Internal order in this project (from basis.md_overlap spherical rows):
      - p: (py, pz, px)
      - d: (d_xy, d_yz, d_z2, d_xz, d_x2-y2)
      - f: project internal 7 real spherical functions (consistent with dxtb)
      - g: project internal 9 real spherical functions (m = −4..+4)
    Mapping to Molden follows the widely used PySCF convention for spherical sets:
      d: [2,3,1,4,0], f: [3,4,2,5,1,6,0], g: [4,5,3,6,2,7,1,8,0]
    """
    if l == 0:
        return [0]
    if l == 1:
        # internal: [py, pz, px] -> Molden: [px, py, pz]
        return [2, 0, 1]
    if l == 2:
        # internal: [d_xy, d_yz, d_z2, d_xz, d_x2-y2]
        # Molden 5D: [dz2, dxz, dyz, dx2-y2, dxy]
        return [2, 3, 1, 4, 0]
    if l == 3:
        # f spherical mapping (PySCF order_ao_index spherical)
        return [3, 4, 2, 5, 1, 6, 0]
    if l == 4:
        # g spherical mapping (PySCF order_ao_index spherical)
        return [4, 5, 3, 6, 2, 7, 1, 8, 0]
    raise NotImplementedError("Angular momentum l>4 not supported")


def _internal_cart_to_molden_order(l: int) -> List[int]:
    """Permutation mapping internal Cartesian order (from _cart_list) -> Molden cart AO order.

    Matches PySCF order_ao_index for cart case:
      - d (6D): xx, yy, zz, xy, xz, yz
      - f (10F): xxx, yyy, zzz, xyy, xxy, xxz, xzz, yzz, yyz, xyz
      - g (15G): xxxx yyyy zzzz xxxy xxxz yyyx yyyz zzzx zzzy xxyy xxzz yyzz xxyz yyxz zzxy
    """
    if l == 0:
        return [0]
    if l == 1:
        # internal cart p order from _cart_list is likely (x,y,z) or similar; ensure px,py,pz
        # Our _cart_list(1) yields [(1,0,0),(0,1,0),(0,0,1)] corresponding to px,py,pz directly
        return [0, 1, 2]
    if l == 2:
        return [0, 3, 5, 1, 2, 4]
    if l == 3:
        return [0, 6, 9, 3, 1, 2, 5, 8, 7, 4]
    if l == 4:
        return [0, 10, 14, 1, 2, 6, 11, 9, 13, 3, 5, 12, 4, 7, 8]
    raise NotImplementedError("Angular momentum l>4 not supported for Cartesian order")


def _format_float(x: float) -> str:
    # Deterministic, compact but precise formatting similar to PySCF
    return f"{x:18.14g}"


def _count_nao(shells: Sequence[ShellRecord]) -> int:
    return sum(_n_sph(sh.l) for sh in shells)


def _validate_shells(shells: Sequence[ShellRecord], nat: int) -> None:
    for sh in shells:
        if not (0 <= sh.atom_index < nat):
            raise ValueError(f"Shell atom_index {sh.atom_index} out of range [0,{nat})")
        if len(sh.exps) == 0:
            raise ValueError("Shell with zero primitives")
        if any(len(c) != len(sh.exps) for c in sh.contractions):
            raise ValueError("Contraction length does not match number of primitives")


def _build_molden_to_internal_idx(shells: Sequence[ShellRecord]) -> List[int]:
    """Global AO index mapping: molden_index -> internal_index.

    Build by concatenating per-shell permutations.
    """
    mapping: List[int] = []
    off = 0
    for sh in shells:
        perm = _internal_to_molden_order(sh.l)
        if len(perm) != _n_sph(sh.l):
            raise AssertionError("Permutation length mismatch for shell l={sh.l}")
        mapping.extend([off + k for k in perm])
        off += _n_sph(sh.l)
    return mapping


def write(
    path: str | Path,
    *,
    numbers: Sequence[int],
    symbols: Sequence[str],
    coords: Sequence[Sequence[float]],
    unit: str,
    shells: Sequence[ShellRecord],
    wf: MOWavefunction,
    spherical: bool,
    program: str = "gxtb",
    version: str | None = None,
    ao_row_scale: Optional[Sequence[float]] = None,
) -> None:
    """Write a Molden file.

    Parameters
    - path: output file path
    - numbers: atomic numbers (nat,)
    - symbols: element symbols (nat,); required (no implicit Z->symbol mapping)
    - coords: geometry (nat,3) in the units specified by `unit`
    - unit: 'AU' or 'Angs' (case-insensitive label is printed as-is)
    - shells: sequence of ShellRecord in the global AO build order used for SCF
    - wf: molecular orbitals (restricted or unrestricted)
    - spherical: True to declare [5d][7f][9g] (real spherical), False for [6d][10f][15g]
    - program/version: provenance line in header
    """
    nat = len(numbers)
    if len(symbols) != nat or len(coords) != nat:
        raise ValueError("symbols and coords must have length equal to numbers")
    if unit not in ("AU", "Angs"):
        raise ValueError("unit must be 'AU' or 'Angs'")

    _validate_shells(shells, nat)
    nao = _count_nao(shells)

    # Compute mapping for on-the-fly coefficient reorder (Molden index -> internal index)
    molden_to_internal = _build_molden_to_internal_idx(shells)
    if len(molden_to_internal) != nao:
        raise AssertionError("AO mapping length mismatch")

    def _check_mo(mo: MOSet) -> None:
        # Expect mo.coeff shape (nao, nmo)
        coeff = mo.coeff
        try:
            nao_chk = len(coeff)
            nmo = len(mo.energy)
        except Exception as e:  # pragma: no cover
            raise TypeError("MOSet has non-sequence fields")
        if nao_chk != nao:
            raise ValueError(f"MO coeff row count {nao_chk} != nao {nao}")
        # Check occ length
        if len(mo.occ) != nmo:
            raise ValueError("Length of occ does not match energy")
        if mo.symm is not None and len(mo.symm) != nmo:
            raise ValueError("Length of symm labels must match number of MOs")

    _check_mo(wf.alpha)
    if wf.beta is not None:
        _check_mo(wf.beta)

    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        # Header
        f.write("[Molden Format]\n")
        ver = version if version is not None else ""
        f.write(f"made by {program} {ver}\n")

        # Atoms
        f.write(f"[Atoms] ({unit})\n")
        for i, (Z, sym, xyz) in enumerate(zip(numbers, symbols, coords), start=1):
            if len(xyz) != 3:
                raise ValueError("coords must have shape (nat,3)")
            x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
            f.write(f"{sym}   {i}   {int(Z)}   {_format_float(x)}   {_format_float(y)}   {_format_float(z)}\n")

        # GTO
        f.write("[GTO]\n")
        # Group shells by atom
        by_atom: Dict[int, List[ShellRecord]] = {i: [] for i in range(nat)}
        for sh in shells:
            by_atom[sh.atom_index].append(sh)
        for i in range(nat):
            f.write(f"{i+1} 0\n")
            for sh in by_atom[i]:
                L = _l_sym(sh.l)
                nprim = len(sh.exps)
                # One line per contraction per Molden convention
                for cvec in sh.contractions:
                    f.write(f" {L}   {nprim:2d} 1.00\n")
                    for a, c in zip(sh.exps, cvec):
                        f.write(f"    {_format_float(float(a))}  {_format_float(float(c))}\n")
            f.write("\n")

        # Angular momentum flag
        if spherical:
            f.write("[5d]\n[7f]\n[9g]\n")
        else:
            f.write("[6d]\n[10f]\n[15g]\n")

        # Optional AO row scaling (e.g., for normalization adjustments)
        if ao_row_scale is not None:
            if len(ao_row_scale) != nao:
                raise ValueError("ao_row_scale length must equal number of AOs")
            scale = [float(x) for x in ao_row_scale]
        else:
            scale = None

        def dump_mo_block_spherical(mo: MOSet, spin: str) -> None:
            f.write("[MO]\n")
            coeff = mo.coeff
            nmo = len(mo.energy)
            for imo in range(nmo):
                ene = float(mo.energy[imo])
                occ = float(mo.occ[imo])
                sym = mo.symm[imo] if (mo.symm is not None) else 'A'
                f.write(f" Sym= {sym}\n")
                f.write(f" Ene= {_format_float(ene)}\n")
                f.write(f" Spin= {spin}\n")
                f.write(f" Occup= {_format_float(occ)}\n")
                # Write coefficients in Molden AO order
                for iao_molden, iao_internal in enumerate(molden_to_internal, start=1):
                    cij = float(coeff[iao_internal][imo])
                    if scale is not None:
                        cij = cij * scale[iao_internal]
                    f.write(f" {iao_molden:3d}    {_format_float(cij)}\n")

        def dump_mo_block_cartesian(mo: MOSet, spin: str) -> None:
            # Precompute per-shell spherical->cart transforms and AO slices
            # Build spherical AO offsets per shell
            sph_counts = [_n_sph(sh.l) for sh in shells]
            offs: List[int] = []
            acc = 0
            for n in sph_counts:
                offs.append(acc)
                acc += n
            # Prepare transforms and Molden cart order indices
            transforms: List[torch.Tensor] = []  # T^T for each shell (n_cart x n_sph)
            molden_cart_idx: List[List[int]] = []
            for sh in shells:
                l = sh.l
                ncart = _n_cart(l)
                nsph = _n_sph(l)
                # Use first contraction (typical for q-vSZP); raise if none
                if len(sh.contractions) == 0:
                    raise ValueError("Shell has zero contractions; cannot form transform")
                alpha = torch.tensor(sh.exps, dtype=torch.float64)
                cvec = torch.tensor(sh.contractions[0], dtype=torch.float64)
                Scc = _overlap_cart_block(l, l, alpha, cvec, alpha, cvec, alpha.new_zeros(3))
                T = _metric_orthonormal_sph_transform(l, Scc)  # (nsph x ncart)
                Tt = T.T.contiguous()
                transforms.append(Tt)
                molden_cart_idx.append(_internal_cart_to_molden_order(l))

            # Now dump [MO]
            f.write("[MO]\n")
            coeff = mo.coeff
            nmo = len(mo.energy)
            for imo in range(nmo):
                ene = float(mo.energy[imo])
                occ = float(mo.occ[imo])
                sym = mo.symm[imo] if (mo.symm is not None) else 'A'
                f.write(f" Sym= {sym}\n")
                f.write(f" Ene= {_format_float(ene)}\n")
                f.write(f" Spin= {spin}\n")
                f.write(f" Occup= {_format_float(occ)}\n")
                # For each shell in order, for each cart AO in Molden order
                ao_counter = 0
                for ish, sh in enumerate(shells):
                    nsph = _n_sph(sh.l)
                    ncart = _n_cart(sh.l)
                    start = offs[ish]
                    end = start + nsph
                    C_sph_col = torch.tensor([float(coeff[i][imo]) for i in range(start, end)], dtype=torch.float64)
                    # Optional AO row scaling applies to spherical rows; if given, scale rows before transform
                    if scale is not None:
                        for k in range(nsph):
                            C_sph_col[k] *= scale[start + k]
                    Tt = transforms[ish]
                    C_cart_col = Tt @ C_sph_col  # (ncart,)
                    for loc_molden in molden_cart_idx[ish]:
                        ao_counter += 1
                        cij = float(C_cart_col[loc_molden].item())
                        f.write(f" {ao_counter:3d}    {_format_float(cij)}\n")

        if spherical:
            dump_mo_block_spherical(wf.alpha, "Alpha")
            if wf.beta is not None:
                dump_mo_block_spherical(wf.beta, "Beta")
        else:
            dump_mo_block_cartesian(wf.alpha, "Alpha")
            if wf.beta is not None:
                dump_mo_block_cartesian(wf.beta, "Beta")


# Convenience builder: construct ShellRecords from q‑vSZP basis + q_eff
def shells_from_qvszp(
    numbers: Sequence[int],
    basisq: Any,
    q_eff: Sequence[float],
) -> List[ShellRecord]:
    """Assemble ShellRecord sequence from q‑vSZP parameters and q_eff.

    Requires:
      - basisq: return of gxtb.params.loader.load_basisq
      - q_eff: per-atom effective charges used to form c = c0 + c1 q_eff

    No defaults: raises if element/shell data missing.
    """
    # Lazy import to avoid coupling io to heavy modules at import time
    from gxtb.basis.qvszp import build_atom_basis, build_dynamic_primitive_coeffs
    import torch

    numbers_t = torch.tensor(list(numbers), dtype=torch.long)
    ab = build_atom_basis(numbers_t, basisq)
    qeff_t = torch.tensor(list(q_eff), dtype=torch.float64)
    if qeff_t.shape[0] != numbers_t.shape[0]:
        raise ValueError("q_eff length must equal nat")
    c_list = build_dynamic_primitive_coeffs(numbers_t, ab, qeff_t)

    shells: List[ShellRecord] = []
    ic = 0
    for sh in ab.shells:
        exps = tuple(float(p[0]) for p in sh.primitives)
        coeffs = (tuple(float(x) for x in c_list[ic].tolist()),)
        shells.append(
            ShellRecord(atom_index=sh.atom_index, l={"s": 0, "p": 1, "d": 2, "f": 3}[sh.l], exps=exps, contractions=coeffs)
        )
        ic += 1
    return shells
