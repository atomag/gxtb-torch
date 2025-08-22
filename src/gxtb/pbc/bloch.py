from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import math
import torch

from ..basis.qvszp import AtomBasis
from ..basis.md_overlap import overlap_shell_pair
from ..basis.overlap import scale_diatomic_overlap
from ..params.loader import GxTBParameters
from ..params.schema import GxTBSchema, map_diatomic_params, map_eht_params, map_cn_params
from ..hamiltonian.distance_tb import distance_factor, en_penalty
from ..hamiltonian.onsite_tb import build_onsite_with_cn
from .cell import build_lattice_translations
from .cn_pbc import coordination_number_pbc

__all__ = [
    "eht_lattice_blocks",
    "assemble_k_matrices",
]

__doc_refs__ = {
    "file": "doc/theory/25_periodic_boundary_conditions.md",
    "eqs": ["Bloch sums for S(k), H(k)", "k-space generalized eigenproblem"],
}


def _shell_meta(basis: AtomBasis):
    meta = []
    for idx, sh in enumerate(basis.shells):
        meta.append({
            'index': idx,
            'atom': sh.atom_index,
            'element': sh.element,
            'l': sh.l,
            'ao_offset': basis.ao_offsets[idx],
            'n_ao': basis.ao_counts[idx],
            'nprims': sh.nprims,
            'primitives': sh.primitives,
        })
    return meta


def eht_lattice_blocks(
    numbers: torch.Tensor,
    positions: torch.Tensor,
    basis: AtomBasis,
    gparams: GxTBParameters,
    schema: GxTBSchema,
    cell: torch.Tensor,
    cutoff: float,
    cn_cutoff: float,
    wolfsberg_mode: str = "arithmetic",
    coeff_override: List[torch.Tensor] | None = None,
    translations: List[Tuple[int,int,int]] | None = None,
    ao_atoms_opt: torch.Tensor | None = None,
) -> Dict[str, object]:
    """Build real-space lattice blocks S(0R), H(0R) up to |R|<=cutoff for EHT first-order.

    - Uses diatomic-frame scaled overlaps (doc/theory/8, Eqs. 31–32) and modified Eq. 64 for off-diagonals.
    - Onsite ε_lA includes CN dependence evaluated under PBC with real-space cutoff cn_cutoff (doc/theory/9 and 25).
    - Returns dict with keys: 'translations' (list of (i,j,k)), 'S_blocks' (list[tensor]), 'H_blocks' (list[tensor]).
    """
    device = positions.device
    dtype = positions.dtype
    meta = _shell_meta(basis)
    nshell = len(meta)
    nao = basis.nao
    # EHT and diatomic parameters
    eht = map_eht_params(gparams, schema)
    diat = map_diatomic_params(gparams, schema)
    # Map Wolfsberg per-shell scalings
    def kW_of(shell_l: str, Z: int) -> torch.Tensor:
        key = f'k_w_{shell_l}'
        if key in eht:
            return eht[key][Z].to(device=device, dtype=dtype)
        return torch.tensor(1.0, dtype=dtype, device=device)
    has_en = 'en' in eht and 'k_en' in eht
    en = eht['en'] if has_en else None
    k_en = float(eht['k_en'][0].item()) if has_en else None

    # Primitive exponents and contraction coefficients
    # eq: doc/theory/7_q-vSZP_basis_set.md (Eqs. 27–28) for c = c0 + c1*q_eff; here allow explicit override.
    alpha = [torch.tensor([p[0] for p in sh['primitives']], dtype=dtype, device=device) for sh in meta]
    if coeff_override is None:
        coeff = [torch.tensor([p[1] for p in sh['primitives']], dtype=dtype, device=device) for sh in meta]
    else:
        if len(coeff_override) != len(meta):
            raise ValueError("coeff_override length must match number of shells in basis")
        coeff = [c.to(device=device, dtype=dtype) for c in coeff_override]

    # CN under PBC and onsite eps per AO
    cn_map = map_cn_params(gparams, schema)
    cn = coordination_number_pbc(positions, numbers, cn_map['r_cov'].to(device=device, dtype=dtype), float(cn_map['k_cn']), cell, float(cn_cutoff))
    eps_ao = build_onsite_with_cn(numbers, cn, basis, gparams, schema)

    # Helper for diatomic scaling tensors for elements
    def k_channels_for_Z(Zval: int) -> dict:
        return {
            "sigma": float(diat["sigma"][Zval].item()),
            "pi": float(diat["pi"][Zval].item()),
            "delta": float(diat["delta"][Zval].item()),
        }

    # Lattice translations within cutoff (include origin)
    if translations is None:
        translations = build_lattice_translations(float(cutoff), cell)
    S_blocks_raw: List[torch.Tensor] = []
    S_blocks_scaled: List[torch.Tensor] = []
    H_blocks: List[torch.Tensor] = []

    # Precompute shell eps averages used for off-diagonal blocks
    def shell_eps(ish: int) -> torch.Tensor:
        off = basis.ao_offsets[ish]; n = basis.ao_counts[ish]
        return eps_ao[off:off+n].mean()

    # Main loop over translations
    for (ti, tj, tk) in translations:
        R = ti * cell[0] + tj * cell[1] + tk * cell[2]
        # Initialize blocks for this translation
        S_R_raw = torch.zeros((nao, nao), dtype=dtype, device=device)
        S_R_scaled = torch.zeros((nao, nao), dtype=dtype, device=device)
        H_R = torch.zeros((nao, nao), dtype=dtype, device=device)

        # For origin translation, include onsite diagonal eps on H
        if ti == 0 and tj == 0 and tk == 0:
            H_R.diagonal().copy_(eps_ao)

        # Build off-center shell pair contributions (i: in home cell; j: in cell translated by R)
        for i, shi in enumerate(meta):
            A = int(shi['atom']); ZA = int(shi['element']); lA = shi['l']
            oi, ni = int(shi['ao_offset']), int(shi['n_ao'])
            ai, ci = alpha[i], coeff[i]
            epsA = shell_eps(i)
            kWA = kW_of(lA, ZA)
            for j, shj in enumerate(meta):
                B = int(shj['atom']); ZB = int(shj['element']); lB = shj['l']
                oj, nj = int(shj['ao_offset']), int(shj['n_ao'])
                aj, cj = alpha[j], coeff[j]
                epsB = shell_eps(j)
                kWB = kW_of(lB, ZB)
                # Displacement including lattice translation
                dR = (positions[B] + R) - positions[A]
                # Identify origin translation
                is_origin = (ti == 0 and tj == 0 and tk == 0)
                # Unscaled overlap block for this pair across cells (raw MD)
                li = {'s':0,'p':1,'d':2,'f':3}[lA]
                lj = {'s':0,'p':1,'d':2,'f':3}[lB]
                block = overlap_shell_pair(li, lj, ai, ci, aj, cj, dR)
                # Diatomic scaling (Eqs. 31–32) for off-center pairs only.
                # For on-center origin pairs (A==B and R==0), keep raw block to preserve on-center normalization.
                if is_origin and A == B:
                    S_raw = block
                    Ssc = block
                else:
                    kA_ch = k_channels_for_Z(ZA)
                    kB_ch = k_channels_for_Z(ZB)
                    S_raw = block
                    Ssc = scale_diatomic_overlap(block, dR, lA, lB, kA_ch, kB_ch)
                S_R_raw[oi:oi+ni, oj:oj+nj] = S_raw
                S_R_scaled[oi:oi+ni, oj:oj+nj] = Ssc
                # EHT off-diagonal H via modified Eq. 64 (skip on-site diagonal; already covered by eps_ao)
                if is_origin and A == B and i == j:
                    pass
                else:
                    if wolfsberg_mode == "geometric":
                        kpair = torch.sqrt(torch.clamp(kWA, min=0.0) * torch.clamp(kWB, min=0.0))
                    else:
                        kpair = 0.5 * (kWA + kWB)
                    R_AB = torch.linalg.norm(dR)
                    Pi_R = distance_factor(lA, lB, ZA, ZB, R_AB, eht)
                    X = en_penalty(ZA, ZB, en, k_en)
                    avg_eps = 0.5 * (epsA + epsB)
                    Hblk = kpair * avg_eps * Pi_R * X * Ssc
                    H_R[oi:oi+ni, oj:oj+nj] = Hblk

        # Symmetrize within block (explicitly real)
        S_blocks_raw.append(S_R_raw)
        S_blocks_scaled.append(S_R_scaled)
        H_blocks.append(H_R)

    # AO->atom map for downstream Mulliken charges
    if ao_atoms_opt is not None:
        ao_atoms = ao_atoms_opt.to(device=device)
    else:
        ao_atoms_list: list[int] = []
        for ish, sh in enumerate(basis.shells):
            ao_atoms_list.extend([sh.atom_index] * basis.ao_counts[ish])
        ao_atoms = torch.tensor(ao_atoms_list, dtype=torch.long, device=device)

    return {"translations": translations, "S_blocks_raw": S_blocks_raw, "S_blocks_scaled": S_blocks_scaled, "H_blocks": H_blocks, "ao_atoms": ao_atoms}


def assemble_k_matrices(
    translations: Sequence[Tuple[int,int,int]],
    S_blocks: Sequence[torch.Tensor],
    H_blocks: Sequence[torch.Tensor],
    kpoints: torch.Tensor,
) -> Dict[str, List[torch.Tensor]]:
    """Assemble S(k), H(k) by Bloch sums from real-space blocks (doc/theory/25).

    - kpoints: (nk,3) fractional coordinates.
    - phase(k, n) = exp(i 2π k·n) with integer translation n.
    Returns dict with lists of complex Hermitian matrices S_k, H_k.
    """
    if len(translations) != len(S_blocks) or len(S_blocks) != len(H_blocks):
        raise ValueError("translations, S_blocks, H_blocks must have matching lengths")
    nk = int(kpoints.shape[0])
    nao = int(S_blocks[0].shape[0])
    two_pi = 2.0 * math.pi
    Sks: List[torch.Tensor] = []
    Hks: List[torch.Tensor] = []
    # Precompute integer triplets as tensor
    N = torch.tensor(list(translations), dtype=kpoints.dtype, device=kpoints.device)  # (nR,3)
    for ik in range(nk):
        k = kpoints[ik]  # (3,)
        phase = torch.exp(1j * two_pi * (N @ k))  # (nR,)
        S_k = torch.zeros((nao, nao), dtype=torch.complex128)
        H_k = torch.zeros((nao, nao), dtype=torch.complex128)
        for p, (S_R, H_R) in enumerate(zip(S_blocks, H_blocks)):
            ph = phase[p].to(dtype=S_k.dtype)
            S_k = S_k + ph * S_R.to(dtype=S_k.real.dtype)
            H_k = H_k + ph * H_R.to(dtype=H_k.real.dtype)
        # Hermitian symmetrization (numerical)
        S_k = 0.5 * (S_k + S_k.conj().T)
        H_k = 0.5 * (H_k + H_k.conj().T)
        Sks.append(S_k)
        Hks.append(H_k)
    return {"S_k": Sks, "H_k": Hks}
