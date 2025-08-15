from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple

import torch

Shell = Literal["s", "p", "d", "f"]


@dataclass(frozen=True)
class GxTBGlobal:
    lines: Tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class GxTBElementBlock:
    z: int
    lines: Tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class GxTBParameters:
    global_lines: GxTBGlobal
    elements: Dict[int, GxTBElementBlock]


@dataclass(frozen=True)
class EEQElement:
    z: int
    values: torch.Tensor  # shape (10,)


@dataclass(frozen=True)
class EEQParameters:
    elements: Dict[int, EEQElement]


@dataclass(frozen=True)
class Primitive:
    alpha: float
    c1: float
    c2: float


@dataclass(frozen=True)
class ShellPrimitives:
    nprims: int
    primitives: Tuple[Primitive, ...]


@dataclass(frozen=True)
class ElementBasis:
    z: int
    header: Tuple[float, float, float]
    shells: Dict[Shell, Tuple[ShellPrimitives, ...]]


@dataclass(frozen=True)
class BasisQParameters:
    elements: Dict[int, ElementBasis]


@dataclass(frozen=True)
class D4Parameters:
    raw: dict


@dataclass(frozen=True)
class DiatomicScaling:
    """Diatomic scaling parameters (k_diat) for overlap matrix. (doc/theory/8)"""

    k_sigma: float  # scaling for s-s, s-p_sigma, p-p_sigma
    k_pi: float  # scaling for p-p_pi
    k_delta: float  # scaling for d-d_delta, etc.


@dataclass(frozen=True)
class EHTParameters:
    """EHT parameters (h_l, k^W_l, etc.) for an element. (doc/theory/12)"""

    h_s: float  # H_s,eff shell energy
    h_p: float  # H_p,eff shell energy
    h_d: float  # H_d,eff shell energy
    k_w: float  # Wolfsberg factor
    k_h: float  # Hubbard factor for third-order
    # Coefficients for coordination-dependent polynomial Π_l(R)
    # (c0, c1, c2, c3, c4, c5)
    pi_s: torch.Tensor
    pi_p: torch.Tensor
    pi_d: torch.Tensor
    eps_f: torch.Tensor
    k_ho_f: torch.Tensor
    k_w_f: torch.Tensor
    pi_f: torch.Tensor
    # CN polynomial coefficients per shell
    pi_cn_s: torch.Tensor
    pi_cn_p: torch.Tensor
    pi_cn_d: torch.Tensor
    pi_cn_f: torch.Tensor
    # distance polynomial coefficients per shell-pair class (s-s, s-p, p-p, ...)
    pi_r_ss: torch.Tensor
    pi_r_sp: torch.Tensor
    pi_r_pp: torch.Tensor
    pi_r_sd: torch.Tensor
    pi_r_pd: torch.Tensor
    pi_r_dd: torch.Tensor
    pi_r_sf: torch.Tensor
    pi_r_pf: torch.Tensor
    pi_r_df: torch.Tensor
    pi_r_ff: torch.Tensor
    # electronegativity and global scaling
    en: torch.Tensor  # per element electronegativity
    k_en: torch.Tensor  # global scalar (broadcast)

# NOTE: Do not extend GxTBElementBlock in-place; we map interpreted parameters
# separately to keep raw ingestion lossless. Provide a container for mapped
# EHT tables (per-element tensors) after schema interpretation.
@dataclass(frozen=True)
class EHTTables:
    h_s: torch.Tensor
    h_p: torch.Tensor
    h_d: torch.Tensor
    k_w_s: torch.Tensor
    k_w_p: torch.Tensor
    k_w_d: torch.Tensor
    # future: f-shell support
    # Optional higher-order / CN polynomials could be added later.


@dataclass(frozen=True)
class RevD4RefData:
    """Reference dataset for revD4/D4S dispersion construction.

    Contains:
    - refalpha: (Z, R, W) neutral reference polarizabilities α^0_{Z,r}(iω)
    - refcovcn: (Z, R) reference CNs per element/reference
    - refc: (Z, R) reference multiplicity Ns ∈ {0,1,3}
    - r4r2: (Z,) atomic expectation values used to build C8 from C6
    """

    refalpha: torch.Tensor
    refcovcn: torch.Tensor
    refc: torch.Tensor
    r4r2: torch.Tensor


@dataclass(frozen=True)
class D4SData:
    """Hardness-based charge-scaling data for D4S ζ path.

    Contains:
    - refq: (Z, R) per-element, per-reference charges
    - gam: (Z,) chemical hardness table
    - zeff: (Z,) effective charges
    - ga, gc: global scalars controlling exponent heights and steepness
    """

    refq: torch.Tensor
    gam: torch.Tensor
    zeff: torch.Tensor
    ga: float
    gc: float
