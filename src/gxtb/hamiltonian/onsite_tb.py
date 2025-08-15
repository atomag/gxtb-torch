from __future__ import annotations
"""Onsite energy construction ε_{lA} including CN polynomial shifts.

Equations:
    - Linear CN dependence (fallback) ε_{lA} = h_{l_A} - k^{H0,CN}_{l_A} * CN_A (Eq. 65)
    - Polynomial extension ε_{lA} = ε^0_{lA} + Π^{CN}_l(CN_A) (design extension; Π^{CN}_l Horner-evaluated like distance polynomials Eqs. 66–67 but over CN instead of R).

Notes:
    - If polynomial coefficients (pi_cn_l) exist we treat ε^0_{lA} analogous to h_{l_A}, adding the evaluated polynomial shift.
    - Otherwise we attempt linear form with k_ho_l (mapped to k^{H0,CN}_{l_A}).
    - CN evaluation depends on covalent radii and scaling constant (Sec. 1.4, earlier Eq. 48 for dCN/dR used later for gradients).
"""
import torch
from typing import Dict, List
from ..cn import coordination_number
from ..basis.qvszp import AtomBasis
from ..params.schema import map_eht_params, GxTBSchema
from ..params.loader import GxTBParameters
from .utils_tb import poly_eval

Tensor = torch.Tensor

__all__ = ["build_onsite"]


def build_onsite(numbers: Tensor, positions: Tensor, basis: AtomBasis, gparams: GxTBParameters, schema: GxTBSchema, r_cov: Tensor | None, k_cn: float | None) -> torch.Tensor:
    """Return per-AO onsite energies ε used for H_{μμ} and averaging in modified Eq. 64."""
    eht = map_eht_params(gparams, schema)
    if r_cov is not None and k_cn is not None:
        cn = coordination_number(positions, numbers, r_cov, k_cn)
    else:
        cn = torch.zeros(numbers.shape[0], dtype=positions.dtype, device=positions.device)
    nao = basis.nao
    eps_ao = torch.zeros(nao, dtype=positions.dtype, device=positions.device)
    for idx, sh in enumerate(basis.shells):
        offs = basis.ao_offsets[idx]
        n = basis.ao_counts[idx]
        z = sh.element
        l = sh.l
        # Onsite base level: prefer eps_* (schema uses eps_l); allow legacy h_* as fallback.
        # eq: doc/theory/12_eht_hamiltonian.md Eq. 65 (ε_{lA} base level)
        eps_key = f"eps_{l}"
        h_key = f"h_{l}"
        kho_key = f"k_ho_{l}"
        eps0 = eht.get(eps_key, None)
        if eps0 is None:
            eps0 = eht.get(h_key, None)
        if eps0 is None:
            raise KeyError(
                f"Missing onsite base parameter for shell '{l}' (expected '{eps_key}' or legacy '{h_key}') per Eq. 65; check parameters/gxtb.schema.toml [eht] mapping."
            )
        # Base linear CN shift
        if kho_key in eht:
            eps_lin = eps0[z] - eht[kho_key][z] * cn[sh.atom_index]
        else:
            eps_lin = eps0[z]
        # Override with CN polynomial if available (design extension)
        poly_key = f"pi_cn_{l}"
        if poly_key in eht:
            coeff = eht[poly_key][z]
            eps_poly = poly_eval(coeff, cn[sh.atom_index])
            eps_val = eps0[z] + eps_poly
        else:
            eps_val = eps_lin
        eps_ao[offs:offs+n] = eps_val
    return eps_ao


def build_onsite_with_cn(numbers: Tensor, cn: Tensor, basis: AtomBasis, gparams: GxTBParameters, schema: GxTBSchema) -> torch.Tensor:
    """Return per-AO onsite energies ε using a precomputed CN vector (PBC‑aware CN or custom).

    This mirrors build_onsite but takes CN explicitly to allow periodic CN evaluation per doc/theory/25_periodic_boundary_conditions.md.
    """
    eht = map_eht_params(gparams, schema)
    nao = basis.nao
    eps_ao = torch.zeros(nao, dtype=cn.dtype, device=cn.device)
    for idx, sh in enumerate(basis.shells):
        offs = basis.ao_offsets[idx]
        n = basis.ao_counts[idx]
        z = sh.element
        l = sh.l
        eps_key = f"eps_{l}"
        h_key = f"h_{l}"
        kho_key = f"k_ho_{l}"
        eps0 = eht.get(eps_key, None)
        if eps0 is None:
            eps0 = eht.get(h_key, None)
        if eps0 is None:
            raise KeyError(
                f"Missing onsite base parameter for shell '{l}' (expected '{eps_key}' or legacy '{h_key}') per Eq. 65; check parameters/gxtb.schema.toml [eht] mapping."
            )
        if kho_key in eht:
            eps_lin = eps0[z].to(device=cn.device, dtype=cn.dtype) - eht[kho_key][z].to(device=cn.device, dtype=cn.dtype) * cn[sh.atom_index]
        else:
            eps_lin = eps0[z].to(device=cn.device, dtype=cn.dtype)
        poly_key = f"pi_cn_{l}"
        if poly_key in eht:
            coeff = eht[poly_key][z].to(device=cn.device, dtype=cn.dtype)
            eps_poly = poly_eval(coeff, cn[sh.atom_index])
            eps_val = eps0[z].to(device=cn.device, dtype=cn.dtype) + eps_poly
        else:
            eps_val = eps_lin
        eps_ao[offs:offs+n] = eps_val
    return eps_ao
