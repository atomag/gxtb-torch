from __future__ import annotations
"""Adapter utilities to bridge EHT Hamiltonian and SCF driver.

Functions:
    - build_eht_core: constructs first-order H0, eps_ao, S, ao_atoms.
    - make_core_builder: closure matching scf() expected build_h_core signature.

Equation context:
    - H0 corresponds to H^{EHT} (Eq. 64 modified form) assembled in build_eht_hamiltonian.
    - eps_ao holds onsite ε values (Eq. 65 or polynomial extension) used for Hubbard / charge updates elsewhere.
    - S is the scaled overlap S^{sc} (Eqs. 31–32) needed for Löwdin orthogonalization in SCF.
    - first-order energy later computed as Tr{ P H0 } (Eq. 63).
"""
from typing import Dict, Callable
import torch
from .eht import build_eht_hamiltonian
from ..params.loader import GxTBParameters
from ..params.schema import GxTBSchema, map_cn_params
from .overlap_tb import build_scaled_overlap_dynamic
from ..basis.qvszp import AtomBasis


def build_eht_core(numbers: torch.Tensor, positions: torch.Tensor, basis: AtomBasis, gparams: GxTBParameters, schema: GxTBSchema, wolfsberg_mode: str = "arithmetic") -> Dict[str, torch.Tensor]:
    cn_map = map_cn_params(gparams, schema) if schema.cn else None
    r_cov = cn_map['r_cov'] if cn_map else None
    k_cn = cn_map['k_cn'] if cn_map else None
    res = build_eht_hamiltonian(numbers, positions, basis, gparams, schema, r_cov=r_cov, k_cn=k_cn, wolfsberg_mode=wolfsberg_mode)
    ao_atoms: list[int] = []
    for idx, sh in enumerate(basis.shells):
        offs = basis.ao_offsets[idx]
        n = basis.ao_counts[idx]
        ao_atoms.extend([sh.atom_index] * n)
    ao_atoms_t = torch.tensor(ao_atoms, dtype=torch.long, device=positions.device)
    # Provide both raw and scaled overlaps; SCF orthogonalization must use S_raw (doc/theory/5)
    return {'H0': res.H, 'eps_ao': res.eps, 'S': res.S_raw, 'S_raw': res.S_raw, 'S_scaled': res.S_scaled, 'ao_atoms': ao_atoms_t}


def make_core_builder(basis: AtomBasis, gparams: GxTBParameters, schema: GxTBSchema, wolfsberg_mode: str = "arithmetic") -> Callable[[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """Return closure building core Hamiltonian with chosen wolfsberg_mode ('arithmetic' or 'geometric')."""
    def _builder(numbers: torch.Tensor, positions: torch.Tensor, ctx: Dict[str, torch.Tensor]):
        # Optional: allow SCF to override S via dynamic q‑vSZP coefficients or explicit S
        S_raw_override = ctx.get('S_raw', None)
        S_scaled_override = ctx.get('S_scaled', None)
        if S_scaled_override is None and 'coeffs' in ctx:
            # Build S^{sc} from dynamic primitive coefficients (doc/theory/7 Eq. 27 + Eqs. 31–32)
            coeffs = ctx['coeffs']  # mapping idx->tensor
            S_scaled_override = build_scaled_overlap_dynamic(numbers, positions, basis, coeffs, gparams, schema)
        cn_map = map_cn_params(gparams, schema)
        res = build_eht_hamiltonian(
            numbers, positions, basis, gparams, schema,
            r_cov=cn_map['r_cov'], k_cn=cn_map['k_cn'], wolfsberg_mode=wolfsberg_mode,
            S_raw_override=S_raw_override, S_scaled_override=S_scaled_override,
        )
        # Build AO->atom map (consistent with build_eht_core)
        ao_atoms: list[int] = []
        for idx, sh in enumerate(basis.shells):
            offs = basis.ao_offsets[idx]
            n = basis.ao_counts[idx]
            ao_atoms.extend([sh.atom_index] * n)
        ao_atoms_t = torch.tensor(ao_atoms, dtype=torch.long, device=positions.device)
        return {'H0': res.H, 'eps_ao': res.eps, 'S_raw': res.S_raw, 'S_scaled': res.S_scaled, 'ao_atoms': ao_atoms_t}
    return _builder
