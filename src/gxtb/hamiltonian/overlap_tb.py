from __future__ import annotations
"""Overlap matrix construction and diatomic channel scaling.

Implements:
    - Raw Gaussian overlap S via McMurchie–Davidson integrals (standard Gaussian overlap; theory Sec. 1.3 preceding Eq. 31 – no explicit number for primitive accumulation; diagonal enforced to 1 post‑contraction consistent with orthonormal on‑center assumption feeding Eq. 64).
    - Diatomic frame rotation and channel scaling factors k^{L}_{A}, L ∈ {σ, π, δ} applied to off‑center blocks (Eqs. 31–32):
                Eq. 31: S̃^{diat} = O_A S O_B^T (rotation into diatomic frame)
                Eq. 32: S̃^{sc}_{μν} = k^{L}_{AB} S̃^{diat}_{μν} per angular channel L (here k^{L}_{AB} taken as harmonic mean of per‑element k^{L}_A, k^{L}_B)

Design notes:
    - Harmonic mean choice assures symmetry and suppresses large disparities (see utils.harmonic_mean comment).
    - f-shell handling enabled (li, lj up to 3). Channel scaling treats f blocks with unity scaling (no σ/π/δ classification currently).
"""
import torch
from typing import Dict
from ..basis.md_overlap import overlap_shell_pair
from ..basis.qvszp import AtomBasis
from ..basis.overlap import rotation_matrices, _bond_type_indices
from ..params.schema import map_diatomic_params, GxTBSchema
from ..params.loader import GxTBParameters
from .utils_tb import harmonic_mean

Tensor = torch.Tensor

__all__ = [
    "build_overlap",
    "apply_diatomic_scaling",
    "build_scaled_overlap",
    "build_overlap_dynamic",
    "build_scaled_overlap_dynamic",
]


def _collect_shell_meta(basis: AtomBasis):
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


def build_overlap(numbers: Tensor, positions: Tensor, basis: AtomBasis) -> Tensor:
    """Raw AO overlap S (pre-scaling) used in Eq. 64; diagonal normalized to 1 (on-site orthonormal assumption)."""
    meta = _collect_shell_meta(basis)
    nao = basis.nao
    dtype = positions.dtype
    device = positions.device
    S = torch.zeros((nao, nao), dtype=dtype, device=device)
    # Precompute per-shell primitive tensors once per geometry/device/dtype
    alpha_list = []
    coeff_list = []
    for sh in meta:
        alpha_list.append(torch.tensor([p[0] for p in sh['primitives']], dtype=dtype, device=device))
        # Static q‑vSZP baseline uses c = c0 (doc/theory/7 Eq. 27 with q_eff=0)
        coeff_list.append(torch.tensor([p[1] for p in sh['primitives']], dtype=dtype, device=device))
    for i, shi in enumerate(meta):
        for j, shj in enumerate(meta):
            if j < i:
                continue
            li = {'s':0,'p':1,'d':2,'f':3,'g':4}[shi['l']]
            lj = {'s':0,'p':1,'d':2,'f':3,'g':4}[shj['l']]
            if li > 4 or lj > 4:
                raise ValueError("Angular momentum > g not supported")
            ai = alpha_list[i]
            ci = coeff_list[i]
            aj = alpha_list[j]
            cj = coeff_list[j]
            R = positions[shi['atom']] - positions[shj['atom']]
            block = overlap_shell_pair(li, lj, ai, ci, aj, cj, R)
            oi, oj = shi['ao_offset'], shj['ao_offset']
            ni, nj = shi['n_ao'], shj['n_ao']
            S[oi:oi+ni, oj:oj+nj] = block
            if j != i:
                S[oj:oj+nj, oi:oi+ni] = block.T
    # Do not override diagonal; on-center overlaps are computed analytically via MD (respect theory Eq. 31 inputs)
    return 0.5 * (S + S.T)


def build_overlap_dynamic(numbers: Tensor, positions: Tensor, basis: AtomBasis, coeffs: Dict[int, Tensor]) -> Tensor:
    """AO overlap S using dynamic contraction coefficients per shell.

    coeffs: mapping from shell index -> tensor of primitive coefficients c (len=nprims for that shell).
    """
    meta = _collect_shell_meta(basis)
    nao = basis.nao
    dtype = positions.dtype
    device = positions.device
    S = torch.zeros((nao, nao), dtype=dtype, device=device)
    alpha_list = [torch.tensor([p[0] for p in sh['primitives']], dtype=dtype, device=device) for sh in meta]
    for i, shi in enumerate(meta):
        for j, shj in enumerate(meta):
            if j < i:
                continue
            li = {'s':0,'p':1,'d':2,'f':3,'g':4}[shi['l']]
            lj = {'s':0,'p':1,'d':2,'f':3,'g':4}[shj['l']]
            if li > 4 or lj > 4:
                raise ValueError("Angular momentum > g not supported")
            ai = alpha_list[i]
            aj = alpha_list[j]
            ci = coeffs[i].to(device=device, dtype=dtype)
            cj = coeffs[j].to(device=device, dtype=dtype)
            R = positions[shi['atom']] - positions[shj['atom']]
            block = overlap_shell_pair(li, lj, ai, ci, aj, cj, R)
            oi, oj = shi['ao_offset'], shj['ao_offset']
            ni, nj = shi['n_ao'], shj['n_ao']
            S[oi:oi+ni, oj:oj+nj] = block
            if j != i:
                S[oj:oj+nj, oi:oi+ni] = block.T
    # Preserve analytically computed on-center values; enforce symmetry
    return 0.5 * (S + S.T)


def apply_diatomic_scaling(S: Tensor, numbers: Tensor, positions: Tensor, basis: AtomBasis, diat_params: Dict[str, Tensor]) -> Tensor:
    """Apply σ/π/δ channel scaling (Eqs. 31–32) to off-center overlap blocks in diatomic frame.

    Optimisations:
        - Precompute shell metadata (ℓ, AO offsets/counts, atoms, elements) as tensors.
        - Precompute per-(ℓ_i,ℓ_j) boolean masks for σ/π/δ channels once and reuse.
        - Build K for each pair via mask composition instead of nested Python loops.
        - Keep math in torch (batched-friendly), only iterating over shell pairs for slicing/rotations.

    Behavioural parity:
        - f-shells (> d) remain skipped.
        - If k^L_A == 0 or k^L_B == 0 for a channel L, scaling for that channel is skipped (K entries stay 1),
          matching the previous logic.
    """
    # --- Setup & precomputation
    meta = _collect_shell_meta(basis)
    dtype = S.dtype
    device = S.device

    lmap = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
    nshell = len(meta)

    # Tensors describing shells
    l_idx = torch.tensor([lmap[m['l']] for m in meta], device=device, dtype=torch.long)
    ao_off = torch.tensor([m['ao_offset'] for m in meta], device=device, dtype=torch.long)
    ao_cnt = torch.tensor([m['n_ao'] for m in meta], device=device, dtype=torch.long)
    atom_idx = torch.tensor([m['atom'] for m in meta], device=device, dtype=torch.long)
    elements = [m['element'] for m in meta]  # keep Python list for dict indexing of params

    Sout = S.clone()

    # Diatomic parameters per channel
    k_sigma = diat_params['sigma']
    k_pi = diat_params['pi']
    k_delta = diat_params['delta']

    # AO dimensions per ℓ (s,p,d,f)
    dim_for_l = {0: 1, 1: 3, 2: 5, 3: 7}

    # Precompute index masks for each (ℓ_i, ℓ_j) once
    masks_cache: Dict[tuple, Dict[str, torch.Tensor]] = {}

    inv_lmap = {v: k for k, v in lmap.items()}

    def _channel_masks(li: int, lj: int) -> Dict[str, torch.Tensor]:
        key = (li, lj)
        if key in masks_cache:
            return masks_cache[key]
        # Bond-type indices for each ℓ (use string labels expected by helper)
        typesA = _bond_type_indices(inv_lmap[li])
        typesB = _bond_type_indices(inv_lmap[lj])
        ni, nj = dim_for_l[li], dim_for_l[lj]
        m_sigma = torch.zeros((ni, nj), dtype=torch.bool, device=device)
        m_pi = torch.zeros((ni, nj), dtype=torch.bool, device=device)
        m_delta = torch.zeros((ni, nj), dtype=torch.bool, device=device)
        for L, m in (('sigma', m_sigma), ('pi', m_pi), ('delta', m_delta)):
            rows = typesA[L]
            cols = typesB[L]
            if len(rows) == 0 or len(cols) == 0:
                continue
            # rows/cols are tuples of slice objects; assign directly
            for sA in rows:
                for sB in cols:
                    m[sA, sB] = True
        out = {'sigma': m_sigma, 'pi': m_pi, 'delta': m_delta}
        masks_cache[key] = out
        return out

    def _scale_or_one(kvals, eA: str, eB: str) -> float:
        """Return harmonic mean(kA,kB) if both > 0 else 1.0. Works with dict or tensor-like values."""
        # Robustly extract scalars from dict/Mapping/tensor-like containers
        try:
            kA = float(kvals[eA])
        except Exception:
            kA = 0.0
        try:
            kB = float(kvals[eB])
        except Exception:
            kB = 0.0
        if kA <= 0.0 or kB <= 0.0:
            return 1.0
        return float((2.0 * kA * kB) / (kA + kB))  # harmonic mean

    # --- Main loop over unique shell pairs (i < j)
    for i in range(nshell):
        li = int(l_idx[i].item())
        if li > 3:
            continue  # only up to f supported
        oi = int(ao_off[i].item())
        ni = int(ao_cnt[i].item())
        ei = elements[i]
        ai = int(atom_idx[i].item())

        for j in range(i + 1, nshell):
            lj = int(l_idx[j].item())
            if lj > 3:
                continue
            oj = int(ao_off[j].item())
            nj = int(ao_cnt[j].item())
            ej = elements[j]
            aj = int(atom_idx[j].item())

            # Rotation to diatomic frame for this pair
            R = positions[aj] - positions[ai]
            O = rotation_matrices(R)
            OA = O[inv_lmap[li]]
            OB = O[inv_lmap[lj]]

            # Local block
            block = Sout[oi:oi + ni, oj:oj + nj]
            S_loc = OA @ block @ OB.T

            # Channel masks and per-channel scales
            masks = _channel_masks(li, lj)
            s_sig = _scale_or_one(k_sigma, ei, ej)
            s_pi = _scale_or_one(k_pi, ei, ej)
            s_del = _scale_or_one(k_delta, ei, ej)

            # Compose K via masks (all torch ops)
            K = torch.ones((ni, nj), dtype=dtype, device=device)
            if s_sig != 1.0 and masks['sigma'].any():
                K = torch.where(masks['sigma'], K.new_full((ni, nj), s_sig), K)
            if s_pi != 1.0 and masks['pi'].any():
                K = torch.where(masks['pi'], K.new_full((ni, nj), s_pi), K)
            if s_del != 1.0 and masks['delta'].any():
                K = torch.where(masks['delta'], K.new_full((ni, nj), s_del), K)

            # Rotate back with scaled local overlap
            S_scaled = OA.T @ (S_loc * K) @ OB
            Sout[oi:oi + ni, oj:oj + nj] = S_scaled
            Sout[oj:oj + nj, oi:oi + ni] = S_scaled.T

    return Sout


def build_scaled_overlap(numbers: Tensor, positions: Tensor, basis: AtomBasis, gparams: GxTBParameters, schema: GxTBSchema) -> Tensor:
    """Convenience wrapper returning S or S^{sc} (Eq. 32) depending on diatomic parameters availability."""
    S = build_overlap(numbers, positions, basis)
    diat = map_diatomic_params(gparams, schema) if schema.diatomic else None
    if diat:
        return apply_diatomic_scaling(S, numbers, positions, basis, diat)
    return S


def build_scaled_overlap_dynamic(
    numbers: Tensor,
    positions: Tensor,
    basis: AtomBasis,
    coeffs: Dict[int, Tensor],
    gparams: GxTBParameters,
    schema: GxTBSchema,
) -> Tensor:
    """Build S^{sc} using dynamic primitive coefficients (Eq. 27) and diatomic scaling (Eqs. 31–32).

    Traceability:
      - Contraction coefficients: c = c0 + c1 q_eff(A) (doc/theory/7_q-vSZP_basis_set.md, Eq. 27)
      - Diatomic scaling in σ/π/δ channels (doc/theory/8_diatomic_frame_scaled_overlap.md, Eqs. 31–32)
    """
    S = build_overlap_dynamic(numbers, positions, basis, coeffs)
    diat = map_diatomic_params(gparams, schema) if schema.diatomic else None
    if diat:
        return apply_diatomic_scaling(S, numbers, positions, basis, diat)
    return S
